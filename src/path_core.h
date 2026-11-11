#ifndef NERAULGRAPHICS_PATH_CORE_H
#define NERAULGRAPHICS_PATH_CORE_H

// The path tracer, shared by both backends like the rest of the rendering code (see backend_api.h).
//
// A path tracer finds the light reaching a pixel by following random paths: from the camera to a surface, then
// onward in a direction chosen at random in proportion to how the material scatters light, and so on, adding the
// light each surface receives directly from the lights and the sky along the way. Averaged over many paths this
// converges to the true answer; a few paths give a noisy picture, which denoise_core.h cleans up.
//
// Each surface is a mix of Lambertian diffuse and a GGX specular lobe: the standard physically based material.
// The lights are sampled directly at every surface (next-event estimation), which is what makes their soft
// shadows cheap, and a map for the sky is sampled in proportion to its brightness and combined with
// the surface's own sampling by multiple importance sampling, so a small, bright sun costs no more noise
// than a broad, dim sky. See the headers named above for the units a Light's intensity is in.

#include "render_core.h"

namespace neraulgraphics::NG_NS {
namespace {

constexpr float PI = 3.14159265359f;

NG_FN int imin(const int a, const int b) { return a < b ? a : b; }

// ---------------------------------------------------------------- sampling --

// An orthonormal basis around the unit vector n (Duff et al. 2017): t and b with n make a right-handed frame.
NG_FN void basis(const float3 &n, float3 &t, float3 &b) {
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = make_float3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = make_float3(c, sign + n.y * n.y * a, -n.y);
}

NG_FN float3 cosine_hemisphere_sample(const float u1, const float u2) {
    const float r = sqrtf(u1), phi = 2.0f * PI * u2;
    return make_float3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

// ----------------------------------------------------------------- material --

// What one bounce needs to know of a material.
struct Surface {
    float3 albedo;   // diffuse reflectance (none for a metal)
    float3 f0;       // specular reflectance looking straight at the surface
    float roughness; // perceptual roughness, for the multiple-scattering compensation
    float alpha;     // GGX roughness squared: 0 is a mirror
    float spec_prob; // how often a bounce is sampled from the specular lobe, rather than the diffuse
};

// The classic material fields give every existing material a physical meaning: reflectivity is how metallic it is,
// and the specular exponent sets the roughness (a Blinn-Phong exponent n matches GGX roughness sqrt(2 / (n + 2))).
NG_FN Surface make_surface(const Material &m) {
    Surface s;
    const bool explicit_pbr = m.roughness >= 0.0f || m.metallic >= 0.0f;
    const float metallic = m.metallic >= 0.0f ? saturate(m.metallic) : saturate(m.reflectivity);
    const float roughness = m.roughness >= 0.0f ? saturate(m.roughness)
        : fminf(fmaxf(sqrtf(2.0f / (m.specular_exponent + 2.0f)), 0.03f), 1.0f);
    // A reflectance above 1 would create light: bring the brightest channel back to 1, keeping the hue.
    float3 tint = m.diffuse_color;
    const float peak = max_component(tint);
    if (peak > 1.0f) tint = tint * (1.0f / peak);
    // What a classic material scatters diffusely is its colour times diffuse_albedo; a metal's reflection is tinted by
    // the colour alone (a mirror with diffuse_albedo 0 still reflects, in its colour).
    const float3 base = tint * (explicit_pbr ? 1.0f : saturate(m.diffuse_albedo));
    const float dielectric = (explicit_pbr || m.specular_albedo > 0.0f) ? 0.04f : 0.0f;
    s.f0 = lerp(make_float3(dielectric, dielectric, dielectric), tint, metallic);
    s.albedo = base * (1.0f - metallic);
    s.roughness = roughness;
    s.alpha = fmaxf(roughness * roughness, 0.002f);  // a perfectly smooth lobe has no finite sampling density
    const float spec_w = max_component(s.f0), diff_w = max_component(s.albedo);
    if (spec_w < 1e-4f) s.spec_prob = 0.0f;
    else if (diff_w < 1e-4f) s.spec_prob = 1.0f;
    else s.spec_prob = fminf(fmaxf(spec_w / (spec_w + diff_w), 0.05f), 0.95f);
    return s;
}

NG_FN float ggx_distribution(const float noh, const float a2) {
    const float d = noh * noh * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d + 1e-20f);
}

// Smith's masking and shadowing for GGX, already divided by 4 N.L N.V (the "visibility" term of the specular BRDF).
NG_FN float ggx_visibility(const float nol, const float nov, const float a2) {
    const float ggxv = nol * sqrtf(nov * nov * (1.0f - a2) + a2);
    const float ggxl = nov * sqrtf(nol * nol * (1.0f - a2) + a2);
    return 0.5f / (ggxv + ggxl + 1e-20f);
}

NG_FN float ggx_g1(const float nov, const float a2) { return 2.0f * nov / (nov + sqrtf(a2 + (1.0f - a2) * nov * nov)); }

// The fraction of light a white GGX surface (F0 = 1) reflects towards the viewer at cosine nov, counting only light that scatters once off
// the microfacets: a polynomial fit (4 x 4, within 0.019 of the integral everywhere and 0.003 on average) to the expectation of G2 / G1
// over visible-normal samples of this very BRDF, computed with tools/rendering/single_scatter_albedo.py. It is below 1 for a rough surface
// because light that bounces a second time between microfacets is not counted, which is what makes single-scattering rough metals too dark.
NG_FN float single_scatter_albedo(const float nov, const float roughness) {
    const float m = fminf(fmaxf(nov, 0.05f), 1.0f);
    const float r = fminf(fmaxf(roughness, 0.1f), 1.0f);          // below 0.1 the lobe is nearly a mirror and the albedo nearly 1
    const float p0 = 1.1588385f + m * (-0.2434899f + m * (-1.9937967f + m * (4.4454591f + m * -2.3415755f)));
    const float p1 = -3.1011066f + m * (15.9027169f + m * (-14.3536701f + m * (-9.7754813f + m * 10.7972369f)));
    const float p2 = 11.1633026f + m * (-80.4498132f + m * (143.2939255f + m * (-79.8227469f + m * 8.9758152f)));
    const float p3 = -14.4848248f + m * (117.5458419f + m * (-252.2666243f + m * (190.6101952f + m * -48.2041842f)));
    const float p4 = 6.2187773f + m * (-55.0192745f + m * (129.4373330f + m * (-109.3843620f + m * 32.2038979f)));
    const float e = p0 + r * (p1 + r * (p2 + r * (p3 + r * p4)));
    return fminf(fmaxf(e, 0.05f), 1.0f);
}

// The material's BRDF for light arriving from wi and leaving towards wo (both unit, away from the surface), and
// the probability density with which brdf_sample would have chosen wi. Zero for directions below the surface.
NG_FN float3 brdf_eval(const Surface &s, const float3 &n, const float3 &wo, const float3 &wi, float &pdf) {
    pdf = 0.0f;
    const float nol = dot(n, wi), nov = dot(n, wo);
    if (nol <= 0.0f || nov <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    const float3 h = normalize(wi + wo);
    const float noh = saturate(dot(n, h)), voh = saturate(dot(wo, h));
    const float a2 = s.alpha * s.alpha;
    const float d = ggx_distribution(noh, a2), vis = ggx_visibility(nol, nov, a2);
    const float f1 = 1.0f - voh;
    const float fw = f1 * f1 * f1 * f1 * f1;
    const float3 fresnel = s.f0 + (make_float3(1.0f, 1.0f, 1.0f) - s.f0) * fw;
    // Light that reflects specularly does not also scatter diffusely.
    const float3 diffuse = s.albedo * (make_float3(1.0f, 1.0f, 1.0f) - fresnel) * (1.0f / PI);
    pdf = s.spec_prob * (ggx_g1(nov, a2) * d / (4.0f * nov)) + (1.0f - s.spec_prob) * nol * (1.0f / PI);
    // Multiple scattering (Turquin 2019): the light that bounces more than once between microfacets is put back by scaling the lobe by
    // 1 + F0 (1 / E - 1), E being the single-scattering albedo; it is exact for a white metal and small for a dielectric.
    const float inv_e = 1.0f / single_scatter_albedo(nov, s.roughness) - 1.0f;
    const float3 compensation = make_float3(1.0f + s.f0.x * inv_e, 1.0f + s.f0.y * inv_e, 1.0f + s.f0.z * inv_e);
    return diffuse + fresnel * compensation * (d * vis);
}

// Picks a direction wi for light leaving towards wo, from the diffuse or the specular lobe. The specular lobe is
// sampled by its visible normals (Heitz 2018), which never picks a microfacet the viewer cannot see.
NG_FN bool brdf_sample(const Surface &s, const float3 &n, const float3 &wo, RNG &rng, float3 &wi) {
    const float u1 = rand_float(rng), u2 = rand_float(rng), u3 = rand_float(rng);
    float3 t, b;
    basis(n, t, b);
    if (u3 < s.spec_prob) {
        const float3 ve = make_float3(dot(wo, t), dot(wo, b), dot(wo, n));
        const float3 vh = normalize(make_float3(s.alpha * ve.x, s.alpha * ve.y, ve.z));
        const float len2 = vh.x * vh.x + vh.y * vh.y;
        const float3 t1 = len2 > 1e-12f ? make_float3(-vh.y, vh.x, 0.0f) * (1.0f / sqrtf(len2)) : make_float3(1.0f, 0.0f, 0.0f);
        const float3 t2 = cross(vh, t1);
        const float r = sqrtf(u1), phi = 2.0f * PI * u2;
        const float p1 = r * cosf(phi);
        float p2 = r * sinf(phi);
        const float k = 0.5f * (1.0f + vh.z);
        p2 = (1.0f - k) * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1)) + k * p2;
        const float3 nh = t1 * p1 + t2 * p2 + vh * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2));
        const float3 hl = normalize(make_float3(s.alpha * nh.x, s.alpha * nh.y, fmaxf(0.0f, nh.z)));
        const float3 h = t * hl.x + b * hl.y + n * hl.z;
        wi = reflect(-wo, h);
    } else {
        const float3 l = cosine_hemisphere_sample(u1, u2);
        wi = t * l.x + b * l.y + n * l.z;
    }
    return dot(n, wi) > 0.0f;
}

// ------------------------------------------------------------- environment --

// The probability density, per unit of solid angle, that sample_environment would pick this direction.
NG_FN float environment_pdf(const EnvParams &e, const float3 &dir) {
    const float x = dot(dir, e.right), y = dot(dir, e.up), z = dot(dir, e.forward);
    const float two_pi = 6.28318530718f;
    float u = atan2f(z, x) / two_pi + e.rotation / two_pi;
    u -= floorf(u);
    const float theta = acosf(fminf(fmaxf(y, -1.0f), 1.0f));
    const float sin_theta = sinf(theta);
    if (sin_theta < 1e-5f) return 0.0f;
    const int px = imin(static_cast<int>(u * static_cast<float>(e.map_width)), e.map_width - 1);
    const int py = imin(static_cast<int>(theta * 0.31830988618f * static_cast<float>(e.map_height)), e.map_height - 1);
    return e.density[py * e.map_width + px] / (2.0f * PI * PI * sin_theta);
}

// Which of the n cells a value u in [0, 1) falls in, given the running totals cdf[0..n] (cdf[0] = 0, cdf[n] = 1).
NG_FN int cdf_search(const float *cdf, const int n, const float u) {
    int lo = 0, hi = n;  // the answer is in [lo, hi)
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Picks a direction on the sky in proportion to its brightness (and the area of the sphere each pixel covers).
NG_FN bool sample_environment(const EnvParams &e, RNG &rng, float3 &dir, float3 &radiance, float &pdf) {
    const float u1 = rand_float(rng), u2 = rand_float(rng), u3 = rand_float(rng), u4 = rand_float(rng);
    const int y = cdf_search(e.marginal, e.map_height, u1);
    const int x = cdf_search(e.conditional + static_cast<size_t>(y) * (e.map_width + 1), e.map_width, u2);
    const float u = (static_cast<float>(x) + u3) / static_cast<float>(e.map_width);
    const float v = (static_cast<float>(y) + u4) / static_cast<float>(e.map_height);
    const float theta = PI * v, phi = 2.0f * PI * u - e.rotation;
    const float sin_theta = sinf(theta);
    if (sin_theta < 1e-5f) return false;
    dir = e.right * (sin_theta * cosf(phi)) + e.up * cosf(theta) + e.forward * (sin_theta * sinf(phi));
    pdf = e.density[y * e.map_width + x] / (2.0f * PI * PI * sin_theta);
    radiance = environment_radiance(e, dir);
    return pdf > 0.0f;
}

// ----------------------------------------------------------- direct lighting --

// How much of the light along a shadow ray gets through, per channel: 1 when nothing is in the way, 0 when something opaque is, and in
// between when only glass is, which lets the light through tinted and slightly weakened (the Fresnel reflection off each face). The ray is
// not bent by the glass, so this is a soft shadow without caustics, not a physical one: the focusing of light by a lens needs paths from
// the light that this renderer does not trace. The any-hit test comes first, so an unshadowed point pays nothing for it.
NG_FN float3 shadow_visibility(const SceneParams &s, const int world, float3 origin, const float3 &dir, const float reach) {
    if (!scene_occluded(s, world, origin, dir, reach)) return make_float3(1.0f, 1.0f, 1.0f);
    float3 visible = make_float3(1.0f, 1.0f, 1.0f);
    float remaining = reach;
    for (int crossing = 0; crossing < 8; ++crossing) {
        Hit h;
        if (!scene_intersect(s, world, origin, dir, h) || h.t >= remaining) return visible;
        const Material &m = h.material;
        if (m.transmission <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        // the glass part of the surface passes light, minus the Fresnel reflection (about 4 % at each face), tinted by the glass
        const float pass = m.transmission * 0.92f;
        visible *= make_float3(pass * sqrtf(fmaxf(m.diffuse_color.x, 0.0f)), pass * sqrtf(fmaxf(m.diffuse_color.y, 0.0f)),
            pass * sqrtf(fmaxf(m.diffuse_color.z, 0.0f)));
        if (max_component(visible) < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        remaining -= h.t + 2.0e-3f;
        origin = h.point + dir * 2.0e-3f;
    }
    return visible;
}

// The power heuristic, which weighs two ways of sampling the same light by how well each suits the direction.
NG_FN float power_heuristic(const float a, const float b) { return a * a / (a * a + b * b + 1e-30f); }

// The light a surface point receives straight from the lights, reflected towards wo. A light's intensity is the
// irradiance a surface facing it receives (not diminished by distance), so a white diffuse surface facing a light of
// intensity 1 shows radiance 1: this is what the classic renderer's lights always meant. A light with a radius is a
// small sphere, sampled across the cone it covers, which is what gives the soft shadow.
NG_FN float3 direct_from_lights(const SceneParams &s, const int world, const Hit &hit, const Surface &surf, const float3 &wo, RNG &rng) {
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    const float3 n = hit.normal;
    for (int i = 0; i < s.light_count; ++i) {
        const Light &l = s.lights[i];
        const float3 to_light = l.position - hit.point;
        const float d2 = dot(to_light, to_light);
        const float d = sqrtf(d2);
        const float3 axis = to_light * (1.0f / d);
        const float strength = l.intensity * light_factor(l, axis, d);
        if (strength <= 0.0f) continue;
        float3 wi = axis;
        float reach = d;  // how far a shadow ray must be clear
        if (l.radius > 1e-4f && l.radius < d * 0.999f) {
            const float cos_max = sqrtf(1.0f - l.radius * l.radius / d2);
            const float u1 = rand_float(rng), u2 = rand_float(rng);
            const float cos_t = 1.0f - u1 * (1.0f - cos_max);
            const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t)), phi = 2.0f * PI * u2;
            float3 t, b;
            basis(axis, t, b);
            wi = t * (sin_t * cosf(phi)) + b * (sin_t * sinf(phi)) + axis * cos_t;
            reach = d - l.radius;  // the near side of the sphere
        }
        const float nol = dot(n, wi);
        if (nol <= 0.0f) continue;
        float pdf;
        const float3 f = brdf_eval(surf, n, wo, wi, pdf);
        if (max_component(f) <= 0.0f) continue;
        const float3 visible = shadow_visibility(s, world, offset_origin(hit.point, n, wi), wi, reach);
        if (max_component(visible) <= 0.0f) continue;
        sum += f * l.color * visible * (nol * PI * strength);
    }
    return sum;
}

// ----------------------------------------------------------------- the path --

// What the first surface a pixel sees looks like, for the denoiser.
struct FirstHit {
    float3 normal;
    float depth;     // along the camera's view axis; negative if nothing was hit
    float3 albedo;   // the colour to divide out of the radiance before filtering, and put back after
};

// The first surface seen through the middle of a pixel. This is taken with a ray that is not jittered, on purpose: it
// describes the pixel, not one random sample of it, so it is the same from frame to frame, which is what lets the
// denoiser tell a surface that has not changed from one that has. (A jittered ray's depth changes by tens of percent per
// pixel near the horizon, and would make every frame look like a new surface.) Anti-aliasing is left to the paths,
// which are jittered, and to their average over time.

NG_FN float3 clamp_radiance(const float3 &c, const float limit) {
    // Zero a NaN or negative value, and scale a very bright sample down to the limit while keeping its colour.
    float3 r = make_float3(fmaxf(c.x, 0.0f), fmaxf(c.y, 0.0f), fmaxf(c.z, 0.0f));
    const float peak = max_component(r);
    if (limit > 0.0f && peak > limit) r = r * (limit / peak);
    return r;
}

// The light arriving along one path from the camera: origin and dir (unit) are the camera ray.
NG_FN float3 path_radiance(const SceneParams &s, const int world, float3 origin, float3 dir, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float prev_pdf = 0.0f;
    const bool sky_map = s.env.map != nullptr;
    bool inside = false;            // the ray is inside a transmissive solid: it entered one and has not yet left
    bool specular_bounce = false;   // the last bounce was a delta (glass): the sky was not sampled directly at it
    float3 medium_tint = make_float3(1.0f, 1.0f, 1.0f);   // the glass the ray is travelling in, for Beer-Lambert absorption
    int glass_events = 0;           // glass crossings do not count against the bounce budget, up to a limit

    for (int bounce = 0; bounce <= s.path.bounces; ++bounce) {
        Hit hit;
        if (!scene_intersect(s, world, origin, dir, hit)) {
            const float3 le = environment_radiance(s.env, dir);
            // With a map, the sky has also been sampled directly at each surface, so a path that happens to find it must not count it in full.
            const float w = (bounce == 0 || !sky_map || specular_bounce) ? 1.0f : power_heuristic(prev_pdf, environment_pdf(s.env, dir));
            radiance += throughput * le * w;
            break;
        }
        // Light travelling through glass is absorbed along the way: Beer-Lambert, the tint raised to the distance in metres.
        if (inside) {
            throughput *= make_float3(powf(fmaxf(medium_tint.x, 1e-4f), hit.t), powf(fmaxf(medium_tint.y, 1e-4f), hit.t),
                powf(fmaxf(medium_tint.z, 1e-4f), hit.t));
        }
        const Surface surf = make_surface(hit.material);
        radiance += throughput * hit.material.emission;

        const float3 wo = -dir;

        // Glass: a smooth dielectric, chosen with probability `transmission` (and the opaque material otherwise). A delta lobe cannot
        // be sampled from the lights, so there is no direct lighting here; the path reflects or refracts and goes on. Each choice
        // carries the energy it should in its probability, so the throughput is unchanged.
        if (hit.material.transmission > 0.0f && rand_float(rng) < hit.material.transmission) {
            // A sphere's normal always points out; a mesh's is turned to face the viewer. Turn it here, to the side the ray arrives from.
            const float3 n = dot(hit.normal, wo) >= 0.0f ? hit.normal : -hit.normal;
            const float eta = inside ? hit.material.ior : 1.0f / hit.material.ior;   // index of the side the ray comes from over the other
            const float cos_i = fminf(fmaxf(dot(n, wo), 0.0f), 1.0f);
            const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
            float fresnel = 1.0f;       // total internal reflection unless there is a transmitted ray
            float cos_t = 0.0f;
            if (sin2_t < 1.0f) {
                cos_t = sqrtf(1.0f - sin2_t);
                const float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
                const float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
                fresnel = 0.5f * (rs * rs + rp * rp);
            }
            if (rand_float(rng) < fresnel) {
                dir = reflect(dir, n);
                origin = offset_origin(hit.point, n, dir);
            } else {
                dir = normalize(dir * eta + n * (eta * cos_i - cos_t));
                origin = offset_origin(hit.point, n, dir);
                inside = !inside;
                medium_tint = hit.material.diffuse_color;
            }
            specular_bounce = true;
            if (glass_events < 12) { ++glass_events; --bounce; }    // not a bounce of the budget: the loop's increment puts it back
            continue;
        }
        specular_bounce = false;
        float3 direct = direct_from_lights(s, world, hit, surf, wo, rng);
        if (sky_map) {
            float3 wi, le;
            float pe;
            if (sample_environment(s.env, rng, wi, le, pe) && dot(hit.normal, wi) > 0.0f) {
                float pb;
                const float3 f = brdf_eval(surf, hit.normal, wo, wi, pb);
                const float3 visible = max_component(f) > 0.0f ? shadow_visibility(s, world, offset_origin(hit.point, hit.normal, wi), wi, 1e30f)
                                                               : make_float3(0.0f, 0.0f, 0.0f);
                if (max_component(visible) > 0.0f)
                    // At the last vertex the path is not continued, so sampling the sky directly is the only strategy left: its weight is 1.
                    direct += f * le * visible * (dot(hit.normal, wi) / pe * (bounce == s.path.bounces ? 1.0f : power_heuristic(pe, pb)));
            }
        }
        radiance += throughput * direct;

        if (bounce == s.path.bounces) break;
        float3 wi;
        if (!brdf_sample(surf, hit.normal, wo, rng, wi)) break;
        float pdf;
        const float3 f = brdf_eval(surf, hit.normal, wo, wi, pdf);
        if (pdf <= 1e-8f) break;
        throughput *= f * (dot(hit.normal, wi) / pdf);
        prev_pdf = pdf;

        // Russian roulette: a path that has become dim is likely to contribute little, so end it at random, and make up for it by brightening the ones that continue.
        if (bounce >= 2) {
            const float keep = fminf(fmaxf(max_component(throughput), 0.05f), 0.95f);
            if (rand_float(rng) > keep) break;
            throughput *= 1.0f / keep;
        }
        origin = offset_origin(hit.point, hit.normal, wi);
        dir = wi;
    }
    return radiance;
}

// -------------------------------------------------------------- the pixel --

// What the first surface in a pixel looks like, by an unjittered ray through its middle.
NG_FN FirstHit first_hit(const SceneParams &s, const int world, const int x, const int y, const int width, const int height, const Camera &cam) {
    FirstHit first;
    first.normal = make_float3(0.0f, 0.0f, 0.0f);
    first.depth = -1.0f;
    first.albedo = make_float3(1.0f, 1.0f, 1.0f);
    const float3 dir = camera_ray(cam, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, width, height);
    Hit hit;
    if (!scene_intersect(s, world, cam.position, dir, hit)) return first;
    first.normal = hit.normal;
    first.depth = hit.t * dot(dir, cam.forward);
    // The colour that is divided out and put back. It is averaged over four fixed points of the pixel, so that a pixel
    // straddling two colours (a checker's edge) is divided by their blend, as the paths' average over the pixel is; the
    // points do not change from frame to frame, which keeps the colour steady. What it is exactly does not matter, only that
    // it is the same at both ends. Sky and light sources count as 1: they are not coloured by anything.
    float3 albedo = make_float3(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 4; ++k) {
        const float3 d = camera_ray(cam, static_cast<float>(x) + 0.25f + 0.5f * static_cast<float>(k & 1),
            static_cast<float>(y) + 0.25f + 0.5f * static_cast<float>(k >> 1), width, height);
        Hit h;
        float3 a = make_float3(1.0f, 1.0f, 1.0f);
        if (scene_intersect(s, world, cam.position, d, h) && max_component(h.material.emission) <= 0.0f) {
            const Surface sf = make_surface(h.material);
            a = make_float3(fmaxf(sf.albedo.x + sf.f0.x, 0.04f), fmaxf(sf.albedo.y + sf.f0.y, 0.04f), fmaxf(sf.albedo.z + sf.f0.z, 0.04f));
        }
        albedo += a * 0.25f;
    }
    first.albedo = albedo;
    return first;
}

// The picture's value at a pixel by path tracing, in linear light, averaged over spp jittered paths. first describes the
// pixel's surface, and irradiance is the same average with first's colour divided out.
NG_FN float3 path_pixel(const SceneParams &s, const int world, const int x, const int y, const int width, const int height,
    const Camera &cam, const int frame, const int spp, FirstHit &first, float3 &irradiance) {
    first = first_hit(s, world, x, y, width, height, cam);
    RNG rng = make_rng(y * width + x, frame);
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < spp; ++sample) {
        const float jx = rand_float(rng), jy = rand_float(rng);
        const float3 dir = camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height);
        sum += clamp_radiance(path_radiance(s, world, cam.position, dir, rng), s.path.max_radiance);
    }
    sum = sum * (1.0f / static_cast<float>(spp));
    irradiance = sum / first.albedo;
    return sum;
}

}  // namespace
}  // namespace neraulgraphics::NG_NS

#endif  // NERAULGRAPHICS_PATH_CORE_H
