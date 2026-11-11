#ifndef NERAULGRAPHICS_RENDER_CORE_H
#define NERAULGRAPHICS_RENDER_CORE_H

// The renderer itself, shared by both backends (see backend_api.h): the scene as the trace
// reads it, intersection, shading, and the work for one pixel. A backend only has to call
// render_pixel for every pixel: the CUDA kernel does it with one thread each, the CPU backend
// with a loop spread over threads. Because both run this same code, they draw the same image.

#include "device_scene.h"
#include "mesh_dev.h"

namespace neraulgraphics::NG_NS {
namespace {

constexpr int MAX_DEPTH = 4;
constexpr float MIN_THROUGHPUT = 1e-3f;
constexpr float SURFACE_EPS = 1e-3f;

// Everything a pixel's trace reads besides the frame buffers. On the GPU it is passed to the
// kernel by value, and kernel parameters live in constant memory, so every thread in a warp
// reading the same sphere still gets it in one broadcast. Each Renderer launches with its own
// copy, which is what lets several coexist.
// The sky, ready to look up: a gradient, or a map with the tables that choose directions on it in proportion to its brightness.
struct EnvParams {
    float3 right, up, forward;      // a frame for the up axis; a direction is looked up as (right, up, forward)
    float3 color_up, color_down;
    float intensity;
    float rotation;                 // radians about up
    const float4 *map;              // null for the gradient
    int map_width, map_height;
    const float *marginal;          // map_height + 1
    const float *conditional;       // map_height * (map_width + 1)
    const float *density;           // map_width * map_height
};

// What path tracing is asked to do.
struct PathParams {
    int bounces;           // light bounces before a path ends
    float max_radiance;    // a sample brighter than this is scaled down to it, which keeps rare very bright paths (fireflies) from speckling the image
    float exposure;
    int tonemap;           // nonzero: compress bright light to what a screen can show, rather than clipping it
};

struct SceneParams {
    Sphere spheres[MAX_SPHERES];
    Light lights[MAX_LIGHTS];
    Plane floor;
    MeshScene mesh;
    EnvParams env;
    PathParams path;
    float ambient;
    int sphere_count;
    int light_count;
};

struct Hit {
    float3 point;
    float3 normal;
    Material material;
    float t;
    int object;  // OBJECT_SPHERE + i, OBJECT_FLOOR or OBJECT_INSTANCE + i
};

// Analytic ray-sphere test. t is the distance along dir, which is normalized.
NG_FN bool sphere_hit(const Sphere &sphere, const float3 &origin, const float3 &dir, float &t) {
    const float3 L = sphere.center - origin;
    const float tca = dot(L, dir);
    const float r2 = sphere.radius * sphere.radius;
    const float d2 = dot(L, L) - tca * tca;
    if (d2 > r2) return false;
    const float thc = sqrtf(r2 - d2);
    t = tca - thc;                // front of the sphere
    if (t < 0.0f) t = tca + thc;  // back, only when the origin is inside
    return t >= 0.0f;
}

// Closest-hit query. Records an index and resolves the surface once at the end.
NG_FN bool scene_intersect(const SceneParams &s, const int world, const float3 &origin, const float3 &dir, Hit &hit) {
    float best = 1e30f;
    int best_sphere = -1;

    for (int i = 0; i < s.sphere_count; ++i) {
        float t;
        if (sphere_hit(s.spheres[i], origin, dir, t) && t < best) {
            best = t;
            best_sphere = i;
        }
    }

    bool hit_floor = false;
    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (s.floor.y - origin.y) / dir.y; d > 0.0f && d < best) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < s.floor.half_extent_x && pt.z > s.floor.z_min && pt.z < s.floor.z_max) {
                best = d;
                hit_floor = true;
            }
        }
    }

    // Meshes go last so the BVH walk starts with the closest hit so far as its
    // limit, and skips every box beyond it.
    int best_instance = -1;
    int best_triangle = -1;
    float best_u = 0.0f, best_v = 0.0f;
    if (s.mesh.instance_count > 0 &&
        mesh_intersect(s.mesh, world, origin, dir, best, best_instance, best_triangle, best_u, best_v)) {
        best_sphere = -1;
        hit_floor = false;
    } else {
        best_instance = -1;
    }

    if (best_sphere < 0 && !hit_floor && best_instance < 0) return false;

    hit.t = best;
    hit.point = origin + dir * best;

    if (best_instance >= 0) {
        float3 tint;
        mesh_surface(s.mesh, best_instance, best_triangle, best_u, best_v, dir, hit.normal, tint);
        hit.material = s.mesh.instances[best_instance].material;
        hit.material.diffuse_color = hit.material.diffuse_color * tint;
        hit.object = OBJECT_INSTANCE + (best_instance - world * s.mesh.instances_per_world);  // the same in every world
    } else if (hit_floor) {
        hit.object = OBJECT_FLOOR;
        hit.normal = make_float3(0.0f, 1.0f, 0.0f);
        hit.material = s.floor.material;
        const int cx = static_cast<int>(floorf(hit.point.x * s.floor.checker_scale));
        const int cz = static_cast<int>(floorf(hit.point.z * s.floor.checker_scale));
        const float3 tint = ((cx + cz) & 1) ? make_float3(1.0f, 1.0f, 1.0f) : make_float3(1.0f, 0.7f, 0.3f);
        hit.material.diffuse_color = tint * 0.3f;
    } else {
        hit.normal = normalize(hit.point - s.spheres[best_sphere].center);
        hit.material = s.spheres[best_sphere].material;
        hit.object = OBJECT_SPHERE + best_sphere;
    }
    return true;
}

// Any-hit query for shadow rays: returns on the first occluder found.
NG_FN bool scene_occluded(const SceneParams &s, const int world, const float3 &origin, const float3 &dir, const float max_dist) {
    for (int i = 0; i < s.sphere_count; ++i) {
        float t;
        if (sphere_hit(s.spheres[i], origin, dir, t) && t < max_dist) return true;
    }

    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (s.floor.y - origin.y) / dir.y; d > 0.0f && d < max_dist) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < s.floor.half_extent_x && pt.z > s.floor.z_min && pt.z < s.floor.z_max)
                return true;
        }
    }

    return s.mesh.instance_count > 0 && mesh_occluded(s.mesh, world, origin, dir, max_dist);
}

// The map's colour in a direction: equirectangular, bilinear, wrapping round the up axis.
NG_FN float3 environment_map_lookup(const EnvParams &e, const float3 &dir) {
    const float x = dot(dir, e.right), y = dot(dir, e.up), z = dot(dir, e.forward);
    const float two_pi = 6.28318530718f;
    float u = atan2f(z, x) / two_pi + e.rotation / two_pi;
    u -= floorf(u);
    const float v = acosf(fminf(fmaxf(y, -1.0f), 1.0f)) * 0.31830988618f;  // angle from up, over pi
    const float fx = u * static_cast<float>(e.map_width) - 0.5f, fy = v * static_cast<float>(e.map_height) - 0.5f;
    const float x0f = floorf(fx), y0f = floorf(fy);
    const float tx = fx - x0f, ty = fy - y0f;
    const int w = e.map_width, h = e.map_height;
    int x0 = static_cast<int>(x0f), x1 = x0 + 1, y0 = static_cast<int>(y0f), y1 = y0 + 1;
    x0 = (x0 % w + w) % w;  // wraps round the up axis
    x1 = (x1 % w + w) % w;
    y0 = y0 < 0 ? 0 : (y0 > h - 1 ? h - 1 : y0);  // and stops at the poles
    y1 = y1 < 0 ? 0 : (y1 > h - 1 ? h - 1 : y1);
    const float4 a = e.map[y0 * w + x0], b = e.map[y0 * w + x1], c = e.map[y1 * w + x0], d = e.map[y1 * w + x1];
    return make_float3((a.x + (b.x - a.x) * tx) * (1 - ty) + (c.x + (d.x - c.x) * tx) * ty,
        (a.y + (b.y - a.y) * tx) * (1 - ty) + (c.y + (d.y - c.y) * tx) * ty,
        (a.z + (b.z - a.z) * tx) * (1 - ty) + (c.z + (d.z - c.z) * tx) * ty);
}

// What the sky looks like in a direction: its radiance. The classic gradient by default, brighter or darker along the up axis.
NG_FN float3 environment_radiance(const EnvParams &e, const float3 &dir) {
    if (e.map != nullptr) return environment_map_lookup(e, dir) * e.intensity;
    const float t = saturate(dot(dir, e.up) * 0.5f + 0.5f);
    return lerp(e.color_down, e.color_up, t) * e.intensity;
}

// Vertical gradient background.
NG_FN float3 sky(const SceneParams &s, const float3 &dir) { return environment_radiance(s.env, dir); }

// Nudges a new ray off the surface it started on to avoid self-intersection.
NG_FN float3 offset_origin(const float3 &point, const float3 &normal, const float3 &dir) {
    return point + normal * (dot(dir, normal) < 0.0f ? -SURFACE_EPS : SURFACE_EPS);
}

// Iterative path loop. `throughput` is the running product of reflectivities, so
// the loop can stop once further bounces cannot affect the result.
NG_FN float3 trace(const SceneParams &s, const int world, float3 origin, float3 dir, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float throughput = 1.0f;

    for (int depth = 0;; ++depth) {
        Hit hit;
        if (depth > MAX_DEPTH || !scene_intersect(s, world, origin, dir, hit)) {
            radiance += sky(s, dir) * throughput;
            break;
        }

        float diffuse = 0.0f;
        float specular = 0.0f;
        for (int i = 0; i < s.light_count; ++i) {
            // Sampling a random point on the light is what softens the shadow.
            const float3 lp = sample_sphere_light(s.lights[i].position, s.lights[i].radius, rng);
            const float3 to_light = lp - hit.point;
            const float light_distance = length(to_light);
            const float3 light_dir = to_light * (1.0f / light_distance);

            if (scene_occluded(s, world, offset_origin(hit.point, hit.normal, light_dir), light_dir,
                light_distance))
                continue;

            diffuse += s.lights[i].intensity * fmaxf(0.0f, dot(light_dir, hit.normal));
            specular += fast_pow(fmaxf(0.0f, dot(reflect(light_dir, hit.normal), dir)),
                            hit.material.specular_exponent) * s.lights[i].intensity;
        }

        const float3 direct = hit.material.diffuse_color * (diffuse * hit.material.diffuse_albedo) +
            make_float3(1.0f, 1.0f, 1.0f) * (specular * hit.material.specular_albedo);
        radiance += (direct + hit.material.diffuse_color * (s.ambient * hit.material.diffuse_albedo)) * throughput;

        throughput *= hit.material.reflectivity;
        if (throughput < MIN_THROUGHPUT) break;

        dir = reflect(dir, hit.normal);
        origin = offset_origin(hit.point, hit.normal, dir);
    }
    return radiance;
}

// The picture's value at one pixel: spp jittered rays traced and averaged, in linear light.
NG_FN float3 sample_pixel(const SceneParams &s, const int world, const int x, const int y, const int width,
    const int height, const Camera &cam, const int frame, const int spp) {
    RNG rng = make_rng(y * width + x, frame);
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < spp; ++sample) {
        const float jx = rand_float(rng);
        const float jy = rand_float(rng);
        sum += trace(s, world, cam.position,
            camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height), rng);
    }
    return sum * (1.0f / static_cast<float>(spp));
}

// A camera from a camera-to-world matrix: the top three rows of the 4x4, row-major, in the OpenGL
// convention (x right, y up, looking along -z). The axes are renormalized, so a matrix that carries a
// little scale or drift still gives a unit basis.
NG_FN Camera camera_from_matrix(const float *m, const float fov, const int width, const int height) {
    Camera c;
    c.position = make_float3(m[3], m[7], m[11]);
    c.right = normalize(make_float3(m[0], m[4], m[8]));
    c.up = normalize(make_float3(m[1], m[5], m[9]));
    c.forward = normalize(make_float3(-m[2], -m[6], -m[10]));
    c.fov_scale = tanf(fov * 0.5f);
    c.aspect = static_cast<float>(width) / static_cast<float>(height);
    return c;
}

// The picture for one pixel of a batch, as red, green, blue bytes: no accumulation, so no buffer of sums.
NG_FN void render_pixel_rgb8(const SceneParams &s, const int world, uint8_t *rgb, const int x, const int y,
    const int width, const int height, const Camera &cam, const int frame, const int spp) {
    const float3 c = linear_to_srgb(sample_pixel(s, world, x, y, width, height, cam, frame, spp));
    const int idx = y * width + x;
    rgb[3 * idx] = static_cast<uint8_t>(saturate(c.x) * 255.0f + 0.5f);
    rgb[3 * idx + 1] = static_cast<uint8_t>(saturate(c.y) * 255.0f + 0.5f);
    rgb[3 * idx + 2] = static_cast<uint8_t>(saturate(c.z) * 255.0f + 0.5f);
}

// The work for one pixel: jitter, trace, accumulate, convert, pack.
// s is a reference so the GPU can pass it as a __grid_constant__ without first copying the whole
// scene from the parameter bank into local memory.
NG_FN void render_pixel(const SceneParams &s, const int world, float4 *accum, uint32_t *out, const int x, const int y,
    const int width, const int height, const Camera &cam, const int frame, const int spp) {
    const int idx = y * width + x;
    const float3 sum = sample_pixel(s, world, x, y, width, height, cam, frame, spp);

    // Running average across frames. .w holds the sample count.
    const float4 acc = frame == 0 ? to_float4(sum, 1.0f) : accum[idx] + to_float4(sum, 1.0f);
    accum[idx] = acc;

    // Gamma only. This scene has little range above 1.0, so a tone map would
    // just flatten contrast; reinstate one if the lighting starts to clip.
    const float3 average = xyz(acc) * (1.0f / acc.w);
    out[idx] = pack_argb8888(linear_to_srgb(average));
}

// The images a frame can carry besides the picture. A null pointer is an image not wanted.
struct AovBuffers {
    float *depth;
    float *normal;
    int32_t *object_id;
};

// Depth, normal and object ID for one pixel: a single ray through its middle, with no jitter and no
// shading, so every value is exact and nothing is averaged across an object's edge.
// Depth is measured along the camera's view axis, as a depth camera reports it, not along the ray.
NG_FN void aov_pixel(const SceneParams &s, const int world, const AovBuffers &out, const int x, const int y, const int width,
    const int height, const Camera &cam) {
    const int idx = y * width + x;
    const float3 dir = camera_ray(cam, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, width, height);
    Hit hit;
    const bool found = scene_intersect(s, world, cam.position, dir, hit);
    if (out.depth) out.depth[idx] = found ? hit.t * dot(dir, cam.forward) : ng_infinity();
    if (out.normal) {
        const float3 n = found ? hit.normal : make_float3(0.0f, 0.0f, 0.0f);
        out.normal[3 * idx] = n.x;
        out.normal[3 * idx + 1] = n.y;
        out.normal[3 * idx + 2] = n.z;
    }
    if (out.object_id) out.object_id[idx] = found ? hit.object : OBJECT_NONE;
}

// One ray of a ray query. Distance is along the (normalized) direction. A ray with a zero or
// non-finite direction, or whose first hit is beyond max_distance, reports a miss.
NG_FN void cast_ray(const SceneParams &s, const int world, const float *origins, const float *directions, const float max_distance,
    const int i, float *distance, float *normal, int32_t *object_id) {
    const float3 o = make_float3(origins[3 * i], origins[3 * i + 1], origins[3 * i + 2]);
    float3 d = make_float3(directions[3 * i], directions[3 * i + 1], directions[3 * i + 2]);
    const float len2 = dot(d, d);
    Hit hit;
    // len2 is compared with >, so a NaN direction is a miss too (the host rejects NaN origins).
    bool found = len2 > 1e-30f && len2 < 1e30f;
    if (found) {
        d = d * (1.0f / sqrtf(len2));
        found = scene_intersect(s, world, o, d, hit) && hit.t <= max_distance;
    }
    distance[i] = found ? hit.t : ng_infinity();
    const float3 n = found ? hit.normal : make_float3(0.0f, 0.0f, 0.0f);
    normal[3 * i] = n.x;
    normal[3 * i + 1] = n.y;
    normal[3 * i + 2] = n.z;
    object_id[i] = found ? hit.object : OBJECT_NONE;
}

}  // namespace
}  // namespace neraulgraphics::NG_NS

#endif  // NERAULGRAPHICS_RENDER_CORE_H
