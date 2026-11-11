#ifndef NERAULGRAPHICS_DEVICE_SCENE_H
#define NERAULGRAPHICS_DEVICE_SCENE_H

// Device-side scene types: the public types of scene.h with float3 in place of Vec3
// (CUDA's on the GPU, backend_api.h's on the CPU). Private to the library; the backends convert
// at the boundary. They
// share names with the public types so the device code, which sits in this
// namespace, reads the same as before the split.

#include <cmath>
#include <neraulgraphics/scene.h>
#include "geom_dev.h"

namespace neraulgraphics::NG_NS {

struct Material {
    float3 diffuse_color;
    float diffuse_albedo;
    float specular_albedo;
    float reflectivity;
    float specular_exponent;
    float roughness;  // < 0: derive from specular_exponent
    float metallic;   // < 0: derive from reflectivity
    float3 emission;
    float transmission;  // the fraction that is smooth glass
    float ior;           // its index of refraction
};

struct Sphere {
    float3 center;
    float radius;
    Material material;
};

struct Light {
    float3 position;
    float intensity;
    float radius;
    float falloff;      // 0: the intensity does not fall with distance
    float3 direction;   // unit, where a spotlight points
    float cos_outer;    // cosine of the cone's half-angle; below -1.5 means no cone
    float cos_inner;    // the cosine inside which the light is at full strength
    float3 color;       // multiplies intensity
};

struct Plane {
    float y;
    float half_extent_x;
    float z_min;
    float z_max;
    float checker_scale;
    Material material;
};

struct Camera {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;
    float fov_scale;
    float aspect;
};

inline float3 to_device(const Vec3 &v) { return make_float3(v.x, v.y, v.z); }

inline Vec3 to_public(const float3 &v) { return Vec3{v.x, v.y, v.z}; }

inline Material to_device(const neraulgraphics::Material &m) {
    return Material{to_device(m.diffuse_color), m.diffuse_albedo, m.specular_albedo,
        m.reflectivity, m.specular_exponent, m.roughness, m.metallic, to_device(m.emission), m.transmission, m.ior};
}

inline Sphere to_device(const neraulgraphics::Sphere &s) {
    return Sphere{to_device(s.center), s.radius, to_device(s.material)};
}

inline Light to_device(const neraulgraphics::Light &l) {
    float3 d = to_device(l.direction);
    const float n = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    d = n > 0.0f ? make_float3(d.x / n, d.y / n, d.z / n) : make_float3(0.0f, 0.0f, -1.0f);
    const bool spot = l.spot_angle > 0.0f;
    return Light{to_device(l.position), l.intensity, l.radius, l.falloff, d,
        spot ? std::cos(l.spot_angle) : -2.0f, spot ? std::cos(l.spot_angle * (1.0f - l.spot_softness)) : -2.0f, to_device(l.color)};
}

inline Plane to_device(const neraulgraphics::Plane &p) {
    return Plane{p.y, p.half_extent_x, p.z_min, p.z_max, p.checker_scale, to_device(p.material)};
}

inline Camera to_device(const neraulgraphics::Camera &c) {
    return Camera{to_device(c.position), to_device(c.right), to_device(c.up), to_device(c.forward),
        c.fov_scale, c.aspect};
}

// How much of a light's intensity reaches a point d away from it, towards_light being the unit vector from the point to the light: a
// lamp's falloff with distance and a spotlight's cone (smooth from full strength inside cos_inner to nothing at cos_outer). 1 for the
// plain lights that have neither.
GPU_FN float light_factor(const Light &l, const float3 &towards_light, const float d) {
    float f = 1.0f;
    if (l.falloff > 0.0f) {
        const float q = d / l.falloff;
        f = 1.0f / (1.0f + q * q);
    }
    if (l.cos_outer > -1.5f) {
        // the light points along direction; the point is in the cone when the direction from the light to it agrees
        const float c = -(l.direction.x * towards_light.x + l.direction.y * towards_light.y + l.direction.z * towards_light.z);
        const float t = fminf(fmaxf((c - l.cos_outer) / fmaxf(l.cos_inner - l.cos_outer, 1e-6f), 0.0f), 1.0f);
        f *= t * t * (3.0f - 2.0f * t);
    }
    return f;
}

// Primary ray direction. px/py are continuous, so passing x + jitter anti-aliases.
GPU_FN float3 camera_ray(const Camera &c, const float px, const float py, const int width,
    const int height) {
    const float sx = (2.0f * px / static_cast<float>(width) - 1.0f) * c.fov_scale * c.aspect;
    const float sy = -(2.0f * py / static_cast<float>(height) - 1.0f) * c.fov_scale;
    return normalize(c.right * sx + c.up * sy + c.forward);
}

}  // namespace neraulgraphics::NG_NS

#endif  // NERAULGRAPHICS_DEVICE_SCENE_H
