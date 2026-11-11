#ifndef NERAULGRAPHICS_SCENE_H
#define NERAULGRAPHICS_SCENE_H

// Scene description types. Plain data with no CUDA types, so a program using the
// library needs neither nvcc nor the CUDA headers. The renderer converts these
// to its own device layout when a Renderer is created.

#include <vector>

namespace neraulgraphics {

// A scene is passed to the kernel by value, and kernel parameters have a fixed
// size, so these capacities are fixed at compile time.
constexpr int MAX_SPHERES = 32;
constexpr int MAX_LIGHTS = 8;

// What an object-ID (segmentation) image holds at a pixel, or what a ray query reports: which
// object was hit. Fixed numbers, so an ID means the same thing whatever else is in the scene.
constexpr int OBJECT_NONE = 0;                             // nothing: the ray left the scene
constexpr int OBJECT_SPHERE = 1;                           // sphere i is OBJECT_SPHERE + i
constexpr int OBJECT_FLOOR = OBJECT_SPHERE + MAX_SPHERES;  // the floor
constexpr int OBJECT_INSTANCE = OBJECT_FLOOR + 1;          // mesh instance i is OBJECT_INSTANCE + i

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

    // Physically based parameters, which path tracing uses (the classic renderer ignores them). A
    // negative value means: work it out from the classic fields above, so every existing material
    // has a sensible physical meaning. Metals take their colour from diffuse_color.
    float roughness = -1.0f;  // 0 is mirror-smooth and 1 fully rough. Classic: from specular_exponent
    float metallic = -1.0f;   // 0 is a dielectric (paint, plastic) and 1 a metal. Classic: reflectivity
    Vec3 emission = {0.0f, 0.0f, 0.0f};  // light the surface gives off, as radiance in linear units
    // Glass, water, clear plastic: the fraction of the surface that is a smooth dielectric, which reflects and refracts light by the Fresnel
    // equations and Snell's law (path tracing only; the classic renderer ignores it). 0 is opaque. A path chooses the glass at random in
    // this proportion and otherwise scatters as the opaque material does, so 0.5 is a half-silvered pane. ior is the index of refraction
    // (1.5 glass, 1.33 water, 2.4 diamond). Inside a closed mesh or a sphere the light is tinted by diffuse_color per metre travelled
    // (Beer-Lambert), so (1, 1, 1) is clear and (0.6, 0.9, 0.7) a green bottle. The path needs bounces to spare: entering and leaving a
    // ball costs two.
    float transmission = 0.0f;
    float ior = 1.5f;
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
//
// By default a light's intensity does not fall with distance, like the sun's. falloff makes it a lamp: the intensity is divided by
// 1 + (distance / falloff)^2, so it has fallen to half at a distance of falloff. spot_angle (the half-angle of the cone, in radians,
// below pi) makes it a spotlight pointing along direction, fading from full strength to nothing over the outer spot_softness of the cone.
// With falloff 0 and spot_angle 0, which is what a plain Light(position, intensity, radius) has, nothing changes.
struct Light {
    Vec3 position;
    float intensity;
    float radius;
    float falloff = 0.0f;
    Vec3 direction = {0.0f, 0.0f, -1.0f};
    float spot_angle = 0.0f;
    float spot_softness = 0.25f;
    // The colour of the light, multiplying intensity: (1, 1, 1) is white, (1, 0.7, 0.4) a warm torch or a flame.
    Vec3 color = {1.0f, 1.0f, 1.0f};
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

// The light that comes from the sky: what a ray that leaves the scene sees, and what lights the surfaces it
// leaves behind. A gradient by default (the sky of the classic renderer). A map replaces the gradient with a
// picture of the whole sky, such as a photographed HDRI, which is also what lights the scene: a path tracer
// samples it in proportion to its brightness, so even a small, bright sun is found at once.
struct Environment {
    Vec3 up = {0.0f, 1.0f, 0.0f};            // the world's up axis, which the sky is brighter or darker along
    Vec3 color_up = {0.10f, 0.26f, 0.60f};   // the gradient's colour straight up
    Vec3 color_down = {0.35f, 0.52f, 0.70f}; // and straight down; halfway between at the horizon
    float intensity = 1.0f;                  // scales the gradient's or the map's radiance
    // An equirectangular map: map_width x map_height pixels of 3 floats (red, green, blue, in linear
    // radiance), rows from the top (straight up) to the bottom (straight down), the columns going round
    // the up axis. Empty for no map.
    int map_width = 0;
    int map_height = 0;
    std::vector<float> map;
    float rotation = 0.0f;                   // radians the map is turned about the up axis
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
