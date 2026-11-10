#ifndef NERAULGRAPHICS_DEVICE_SCENE_CUH
#define NERAULGRAPHICS_DEVICE_SCENE_CUH

// Device-side scene types: the public types of scene.h with CUDA's float3 in place
// of Vec3. Private to the library; render_gpu.cu converts at the boundary. They
// share names with the public types so the device code, which sits in this
// namespace, reads the same as before the split.

#include <neraulgraphics/scene.h>
#include "geom_gpu.cuh"

namespace neraulgraphics::device {

struct Material {
    float3 diffuse_color;
    float diffuse_albedo;
    float specular_albedo;
    float reflectivity;
    float specular_exponent;
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
        m.reflectivity, m.specular_exponent};
}

inline Sphere to_device(const neraulgraphics::Sphere &s) {
    return Sphere{to_device(s.center), s.radius, to_device(s.material)};
}

inline Light to_device(const neraulgraphics::Light &l) {
    return Light{to_device(l.position), l.intensity, l.radius};
}

inline Plane to_device(const neraulgraphics::Plane &p) {
    return Plane{p.y, p.half_extent_x, p.z_min, p.z_max, p.checker_scale, to_device(p.material)};
}

inline Camera to_device(const neraulgraphics::Camera &c) {
    return Camera{to_device(c.position), to_device(c.right), to_device(c.up), to_device(c.forward),
        c.fov_scale, c.aspect};
}

// Primary ray direction. px/py are continuous, so passing x + jitter anti-aliases.
GPU_FN float3 camera_ray(const Camera &c, const float px, const float py, const int width,
    const int height) {
    const float sx = (2.0f * px / static_cast<float>(width) - 1.0f) * c.fov_scale * c.aspect;
    const float sy = -(2.0f * py / static_cast<float>(height) - 1.0f) * c.fov_scale;
    return normalize(c.right * sx + c.up * sy + c.forward);
}

}  // namespace neraulgraphics::device

#endif  // NERAULGRAPHICS_DEVICE_SCENE_CUH
