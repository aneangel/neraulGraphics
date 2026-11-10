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
struct SceneParams {
    Sphere spheres[MAX_SPHERES];
    Light lights[MAX_LIGHTS];
    Plane floor;
    MeshScene mesh;
    float ambient;
    int sphere_count;
    int light_count;
};

struct Hit {
    float3 point;
    float3 normal;
    Material material;
    float t;
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
NG_FN bool scene_intersect(const SceneParams &s, const float3 &origin, const float3 &dir, Hit &hit) {
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
        mesh_intersect(s.mesh, origin, dir, best, best_instance, best_triangle, best_u, best_v)) {
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
    } else if (hit_floor) {
        hit.normal = make_float3(0.0f, 1.0f, 0.0f);
        hit.material = s.floor.material;
        const int cx = static_cast<int>(floorf(hit.point.x * s.floor.checker_scale));
        const int cz = static_cast<int>(floorf(hit.point.z * s.floor.checker_scale));
        const float3 tint = ((cx + cz) & 1) ? make_float3(1.0f, 1.0f, 1.0f) : make_float3(1.0f, 0.7f, 0.3f);
        hit.material.diffuse_color = tint * 0.3f;
    } else {
        hit.normal = normalize(hit.point - s.spheres[best_sphere].center);
        hit.material = s.spheres[best_sphere].material;
    }
    return true;
}

// Any-hit query for shadow rays: returns on the first occluder found.
NG_FN bool scene_occluded(const SceneParams &s, const float3 &origin, const float3 &dir, const float max_dist) {
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

    return s.mesh.instance_count > 0 && mesh_occluded(s.mesh, origin, dir, max_dist);
}

// Vertical gradient background.
NG_FN float3 sky(const float3 &dir) {
    const float t = saturate(dir.y * 0.5f + 0.5f);
    return lerp(make_float3(0.35f, 0.52f, 0.70f), make_float3(0.10f, 0.26f, 0.60f), t);
}

// Nudges a new ray off the surface it started on to avoid self-intersection.
NG_FN float3 offset_origin(const float3 &point, const float3 &normal, const float3 &dir) {
    return point + normal * (dot(dir, normal) < 0.0f ? -SURFACE_EPS : SURFACE_EPS);
}

// Iterative path loop. `throughput` is the running product of reflectivities, so
// the loop can stop once further bounces cannot affect the result.
NG_FN float3 trace(const SceneParams &s, float3 origin, float3 dir, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float throughput = 1.0f;

    for (int depth = 0;; ++depth) {
        Hit hit;
        if (depth > MAX_DEPTH || !scene_intersect(s, origin, dir, hit)) {
            radiance += sky(dir) * throughput;
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

            if (scene_occluded(s, offset_origin(hit.point, hit.normal, light_dir), light_dir,
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

// The work for one pixel: jitter, trace, accumulate, convert, pack.
// s is a reference so the GPU can pass it as a __grid_constant__ without first copying the whole
// scene from the parameter bank into local memory.
NG_FN void render_pixel(const SceneParams &s, float4 *accum, uint32_t *out, const int x, const int y,
    const int width, const int height, const Camera &cam, const int frame, const int spp) {
    const int idx = y * width + x;
    RNG rng = make_rng(idx, frame);

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < spp; ++sample) {
        const float jx = rand_float(rng);
        const float jy = rand_float(rng);
        sum += trace(s, cam.position,
            camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height), rng);
    }
    sum = sum * (1.0f / static_cast<float>(spp));

    // Running average across frames. .w holds the sample count.
    const float4 acc = frame == 0 ? to_float4(sum, 1.0f) : accum[idx] + to_float4(sum, 1.0f);
    accum[idx] = acc;

    // Gamma only. This scene has little range above 1.0, so a tone map would
    // just flatten contrast; reinstate one if the lighting starts to clip.
    const float3 average = xyz(acc) * (1.0f / acc.w);
    out[idx] = pack_argb8888(linear_to_srgb(average));
}

}  // namespace
}  // namespace neraulgraphics::NG_NS

#endif  // NERAULGRAPHICS_RENDER_CORE_H
