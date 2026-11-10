#ifndef NERAULGRAPHICS_SCENE_H
#define NERAULGRAPHICS_SCENE_H

// Scene description types. Plain data with no CUDA types, so a program using the
// library needs neither nvcc nor the CUDA headers. The renderer converts these
// to its own device layout when a Renderer is created.

namespace neraulgraphics {

// A scene is passed to the kernel by value, and kernel parameters have a fixed
// size, so these capacities are fixed at compile time.
constexpr int MAX_SPHERES = 32;
constexpr int MAX_LIGHTS = 8;

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec2 {
    float u;
    float v;
};

struct Material {
    Vec3 diffuse_color;
    float diffuse_albedo;
    float specular_albedo;
    float reflectivity;  // 0 = matte, 1 = perfect mirror
    float specular_exponent;
};

inline Material make_material(const Vec3 color, const float kd, const float ks, const float kr,
    const float exponent) {
    return Material{color, kd, ks, kr, exponent};
}

struct Sphere {
    Vec3 center;
    float radius;
    Material material;
};

// radius 0 is a point light (hard shadows); above 0 gives soft shadows.
struct Light {
    Vec3 position;
    float intensity;
    float radius;
};

// Finite axis-aligned checkerboard floor at constant y. A zero half_extent_x
// gives no floor at all, which is what a value-initialized Plane{} is.
struct Plane {
    float y;
    float half_extent_x;
    float z_min;
    float z_max;
    float checker_scale;
    Material material;
};

// Build with make_camera. The basis and fov_scale are precomputed there so the
// kernel does not redo them per pixel.
struct Camera {
    Vec3 position;
    Vec3 right;
    Vec3 up;
    Vec3 forward;
    float fov_scale;
    float aspect;
};

// A camera at position looking at target. fov is the vertical field of view in
// radians; width and height give the aspect ratio. Computed inside the library,
// so the result does not depend on the caller's floating point flags.
Camera make_camera(const Vec3 &position, const Vec3 &target, const Vec3 &world_up, float fov,
    int width, int height);

}  // namespace neraulgraphics

#endif  // NERAULGRAPHICS_SCENE_H
