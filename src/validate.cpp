#include "validate.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace neraulgraphics::detail {


void require(const bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument(message);
}

bool finite(const float v) { return std::isfinite(v); }

bool finite(const Vec3 &v) { return finite(v.x) && finite(v.y) && finite(v.z); }

bool finite(const Material &m) {
    return finite(m.diffuse_color) && finite(m.diffuse_albedo) && finite(m.specular_albedo) &&
        finite(m.reflectivity) && finite(m.specular_exponent);
}

void check_frame_size(const int width, const int height) {
    require(width > 0 && height > 0, "frame size must be positive, got " + std::to_string(width) +
        "x" + std::to_string(height));
    // Pixels are indexed with an int.
    require(static_cast<long long>(width) * height <= INT32_MAX,
        "frame of " + std::to_string(width) + "x" + std::to_string(height) + " pixels is too large");
}

void check_scene(const std::vector<Sphere> &spheres, const std::vector<Light> &lights,
    const Plane &floor) {
    require(spheres.size() <= MAX_SPHERES, "scene has " + std::to_string(spheres.size()) +
        " spheres, at most " + std::to_string(MAX_SPHERES) + " fit");
    require(lights.size() <= MAX_LIGHTS, "scene has " + std::to_string(lights.size()) +
        " lights, at most " + std::to_string(MAX_LIGHTS) + " fit");

    for (size_t i = 0; i < spheres.size(); ++i) {
        const Sphere &sp = spheres[i];
        const std::string name = "sphere " + std::to_string(i);
        require(finite(sp.center) && finite(sp.radius) && finite(sp.material),
            name + " has a NaN or infinite value");
        // A zero radius makes the hit normal 0/0.
        require(sp.radius > 0.0f, name + " radius must be positive, got " +
            std::to_string(sp.radius));
    }
    for (size_t i = 0; i < lights.size(); ++i) {
        const Light &l = lights[i];
        const std::string name = "light " + std::to_string(i);
        require(finite(l.position) && finite(l.intensity) && finite(l.radius),
            name + " has a NaN or infinite value");
        require(l.radius >= 0.0f, name + " radius must be 0 or more, got " +
            std::to_string(l.radius));
    }
    require(finite(floor.y) && finite(floor.half_extent_x) && finite(floor.z_min) &&
        finite(floor.z_max) && finite(floor.checker_scale) && finite(floor.material),
        "floor has a NaN or infinite value");
}

void check_render_args(const Camera &c, const int frame, const int samples_per_frame) {
    require(frame >= 0, "frame must be 0 or more, got " + std::to_string(frame));
    require(samples_per_frame >= 1, "samples_per_frame must be 1 or more, got " +
        std::to_string(samples_per_frame));
    require(finite(c.position) && finite(c.right) && finite(c.up) && finite(c.forward) &&
        finite(c.fov_scale) && finite(c.aspect),
        "camera has a NaN or infinite value (is its target on the view axis, or up parallel to it?)");
}

}  // namespace neraulgraphics::detail
