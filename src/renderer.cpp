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

Frame Renderer::render_frame(const Camera &camera, const int frame, const int samples_per_frame,
    const unsigned outputs) {
    Impl &r = state();
    check_render_args(camera, frame, samples_per_frame);
    constexpr unsigned ALL = OUTPUT_RGB | OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID;
    require(outputs != 0, "render_frame needs at least one output");
    require((outputs & ~ALL) == 0, "unknown output bits in " + std::to_string(outputs));
    return r.backend->render_frame(camera, frame, samples_per_frame, outputs);
}

HostFrame Renderer::render_frame_host(const Camera &camera, const int frame, const int samples_per_frame,
    const unsigned outputs) {
    const Frame f = render_frame(camera, frame, samples_per_frame, outputs);
    const detail::Backend &b = *state().backend;
    const size_t pixels = static_cast<size_t>(f.width) * f.height;
    HostFrame out;
    out.width = f.width;
    out.height = f.height;
    if (f.rgb) { out.rgb.resize(pixels * 3); b.copy_to_host(f.rgb, out.rgb.data(), pixels * 3); }
    if (f.depth) { out.depth.resize(pixels); b.copy_to_host(f.depth, out.depth.data(), pixels * sizeof(float)); }
    if (f.normal) { out.normal.resize(pixels * 3); b.copy_to_host(f.normal, out.normal.data(), pixels * 3 * sizeof(float)); }
    if (f.object_id) { out.object_id.resize(pixels); b.copy_to_host(f.object_id, out.object_id.data(), pixels * sizeof(int32_t)); }
    return out;
}

RayHits Renderer::cast_rays(const float *origins, const float *directions, const int count, const float max_distance,
    const Memory input_memory) {
    Impl &r = state();
    require(count >= 0, "ray count must be 0 or more, got " + std::to_string(count));
    require(count <= INT32_MAX / 3, "too many rays: " + std::to_string(count));
    require(count == 0 || (origins != nullptr && directions != nullptr), "ray origins and directions must not be null");
    // Not NaN, and more than 0: infinity means no limit.
    require(max_distance > 0.0f, "max_distance must be positive, got " + std::to_string(max_distance));
    require(input_memory == Memory::Host || r.backend->device() == Device::Cuda,
        "device pointers need a CUDA renderer; this one runs on the CPU");
    if (input_memory == Memory::Host) {
        // A NaN origin would be traced as if it were a number (the trace is built without NaN checks).
        for (int i = 0; i < 3 * count; ++i)
            require(finite(origins[i]), "ray origin " + std::to_string(i / 3) + " has a NaN or infinite value");
    }
    return r.backend->cast_rays(origins, directions, count, max_distance, input_memory);
}

HostRayHits Renderer::cast_rays_host(const std::vector<float> &origins, const std::vector<float> &directions,
    const float max_distance) {
    require(origins.size() % 3 == 0 && origins.size() == directions.size(),
        "origins and directions must each hold 3 floats per ray, and the same number of rays");
    const int count = static_cast<int>(origins.size() / 3);
    const RayHits hits = cast_rays(origins.data(), directions.data(), count, max_distance, Memory::Host);
    const detail::Backend &b = *state().backend;
    HostRayHits out;
    out.distance.resize(count);
    out.normal.resize(static_cast<size_t>(count) * 3);
    out.object_id.resize(count);
    if (count > 0) {
        b.copy_to_host(hits.distance, out.distance.data(), count * sizeof(float));
        b.copy_to_host(hits.normal, out.normal.data(), static_cast<size_t>(count) * 3 * sizeof(float));
        b.copy_to_host(hits.object_id, out.object_id.data(), count * sizeof(int32_t));
    }
    return out;
}

Memory Renderer::memory() const { return state().backend->device() == Device::Cuda ? Memory::Device : Memory::Host; }

Device Renderer::device() const { return state().backend->device(); }

float Renderer::last_kernel_ms() const { return state().backend->last_kernel_ms(); }

int Renderer::width() const { return state().width; }

int Renderer::height() const { return state().height; }

}  // namespace neraulgraphics
