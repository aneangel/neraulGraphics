// The Renderer class: checks its input, chooses a backend, and hands the work to it.
// Nothing here touches CUDA, so this file builds with or without it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <neraulgraphics/renderer.h>
#include "backend.h"
#include "mesh_build.h"

namespace neraulgraphics {
namespace {

// Input checks. They live here rather than in a header because main.cpp is built
// with -ffast-math, which lets the compiler assume NaN never occurs and delete
// std::isfinite. This file is built without it.

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

// What Auto should mean right now: NERAULGRAPHICS_DEVICE if it says, else CUDA when it can.
Device resolve(const Device requested) {
    if (requested != Device::Auto) return requested;
    if (const char *env = std::getenv("NERAULGRAPHICS_DEVICE")) {
        if (std::strcmp(env, "cpu") == 0) return Device::Cpu;
        if (std::strcmp(env, "cuda") == 0) return Device::Cuda;
    }
    return detail::cuda_usable() ? Device::Cuda : Device::Cpu;
}

}  // namespace

bool cuda_available() { return resolve(Device::Auto) == Device::Cuda; }

Camera make_camera(const Vec3 &position, const Vec3 &target, const Vec3 &world_up, const float fov,
    const int width, const int height) {
    // In double, then rounded: the basis does not depend on the caller's floating point flags
    // or on the backend, and a near-degenerate camera loses less precision before it is rejected.
    struct V {
        double x, y, z;
    };
    auto sub = [](const V &a, const V &b) { return V{a.x - b.x, a.y - b.y, a.z - b.z}; };
    auto cross = [](const V &a, const V &b) { return V{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; };
    auto normalize = [](const V &a) {
        const double n = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        return V{a.x / n, a.y / n, a.z / n};
    };
    auto pub = [](const V &a) { return Vec3{static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)}; };

    const V forward = normalize(sub(V{target.x, target.y, target.z}, V{position.x, position.y, position.z}));
    const V right = normalize(cross(forward, V{world_up.x, world_up.y, world_up.z}));
    const V up = cross(right, forward);
    return Camera{position, pub(right), pub(up), pub(forward), std::tan(fov * 0.5f),
        static_cast<float>(width) / static_cast<float>(height)};
}

struct Renderer::Impl {
    std::unique_ptr<detail::Backend> backend;
    int width = 0;
    int height = 0;
    int instances = 0;
};

Renderer::Renderer(const int width, const int height, const std::vector<Sphere> &spheres,
    const std::vector<Light> &lights, const Plane &floor, const std::vector<Mesh> &meshes,
    const std::vector<Instance> &instances, const std::vector<Texture> &textures, const Device device) {
    check_frame_size(width, height);
    check_scene(spheres, lights, floor);
    detail::check_meshes(meshes, instances, textures);

    // With no instances there is nothing to build, so a meshless scene pays nothing.
    detail::BuiltMeshes built;
    if (!instances.empty()) built = detail::build_meshes(meshes);
    const detail::SceneInput scene{width, height, spheres, lights, floor, textures, instances,
        instances.empty() ? nullptr : &built};

    auto r = std::make_unique<Impl>();
    r->width = width;
    r->height = height;
    r->instances = static_cast<int>(instances.size());
    r->backend = resolve(device) == Device::Cuda ? detail::make_cuda_backend(scene) : detail::make_cpu_backend(scene);
    impl_ = std::move(r);
}

// Defined here, where Impl is complete, because unique_ptr must see it to delete it.
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer &&) noexcept = default;
Renderer &Renderer::operator=(Renderer &&) noexcept = default;

Renderer::Impl &Renderer::state() {
    if (!impl_) throw std::logic_error("neraulgraphics::Renderer used after being moved from");
    return *impl_;
}

const Renderer::Impl &Renderer::state() const {
    if (!impl_) throw std::logic_error("neraulgraphics::Renderer used after being moved from");
    return *impl_;
}

const uint32_t *Renderer::render(const Camera &camera, const int frame, const int samples_per_frame) {
    Impl &r = state();
    check_render_args(camera, frame, samples_per_frame);
    return r.backend->render(camera, frame, samples_per_frame);
}

void Renderer::set_instance_transforms(const std::vector<Transform> &transforms) {
    Impl &r = state();
    if (transforms.size() != static_cast<size_t>(r.instances))
        throw std::invalid_argument("expected " + std::to_string(r.instances) +
            " transforms, one per instance, got " + std::to_string(transforms.size()));
    for (size_t i = 0; i < transforms.size(); ++i)
        detail::check_transform(transforms[i], "transform " + std::to_string(i));
    if (!transforms.empty()) r.backend->set_instance_transforms(transforms);
}

int Renderer::instance_count() const { return state().instances; }

void Renderer::set_ambient(const float ambient) {
    Impl &r = state();
    require(finite(ambient) && ambient >= 0.0f, "ambient must be 0 or more, got " + std::to_string(ambient));
    r.backend->set_ambient(ambient);
}

float Renderer::ambient() const { return state().backend->ambient(); }

Device Renderer::device() const { return state().backend->device(); }

float Renderer::last_kernel_ms() const { return state().backend->last_kernel_ms(); }

int Renderer::width() const { return state().width; }

int Renderer::height() const { return state().height; }

}  // namespace neraulgraphics
