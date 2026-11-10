// BatchRenderer: checks its input, chooses a backend, and hands the work to it. Like renderer.cpp,
// nothing here touches CUDA.

#include <cmath>
#include <cstdint>
#include <string>
#include <neraulgraphics/batch.h>
#include "backend.h"
#include "mesh_build.h"
#include "validate.h"

namespace neraulgraphics {
namespace {

using detail::check_frame_size;
using detail::check_scene;
using detail::finite;
using detail::require;

constexpr unsigned ALL_OUTPUTS = OUTPUT_RGB | OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID;

bool all_finite(const float *values, const size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (!finite(values[i])) return false;
    return true;
}

bool quaternion_is_usable(const float *q) {
    return finite(q[0]) && finite(q[1]) && finite(q[2]) && finite(q[3]) &&
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] > 0.0f;
}

}  // namespace

struct BatchRenderer::Impl {
    std::unique_ptr<detail::Backend> backend;
    int width = 0;
    int height = 0;
    int worlds = 0;
    int cameras_per_world = 0;
    size_t per_world = 0;
    BodyBinding binding;
    bool bound = false;
};

BatchRenderer::BatchRenderer(const int width, const int height, const int world_count, const int cameras_per_world,
    const std::vector<Sphere> &spheres, const std::vector<Light> &lights, const Plane &floor,
    const std::vector<Mesh> &meshes, const std::vector<Instance> &instances, const std::vector<Texture> &textures,
    const Device device) {
    check_frame_size(width, height);
    require(world_count >= 1, "world_count must be 1 or more, got " + std::to_string(world_count));
    require(cameras_per_world >= 1, "cameras_per_world must be 1 or more, got " + std::to_string(cameras_per_world));
    const long long views = static_cast<long long>(world_count) * cameras_per_world;
    // A work item is a row of a view, counted with an int.
    require(views * height <= INT32_MAX, std::to_string(views) + " views of height " + std::to_string(height) + " are too many");
    check_scene(spheres, lights, floor);
    detail::check_meshes(meshes, instances, textures);

    detail::BuiltMeshes built;
    if (!instances.empty()) built = detail::build_meshes(meshes);
    detail::SceneInput scene{width, height, spheres, lights, floor, textures, instances,
        instances.empty() ? nullptr : &built};
    scene.worlds = world_count;
    scene.cameras_per_world = cameras_per_world;

    auto r = std::make_unique<Impl>();
    r->width = width;
    r->height = height;
    r->worlds = world_count;
    r->cameras_per_world = cameras_per_world;
    r->per_world = instances.size();
    r->backend = detail::resolve_device(device) == Device::Cuda ? detail::make_cuda_backend(scene) : detail::make_cpu_backend(scene);
    impl_ = std::move(r);
}

BatchRenderer::~BatchRenderer() = default;
BatchRenderer::BatchRenderer(BatchRenderer &&) noexcept = default;
BatchRenderer &BatchRenderer::operator=(BatchRenderer &&) noexcept = default;

BatchRenderer::Impl &BatchRenderer::state() {
    if (!impl_) throw std::logic_error("neraulgraphics::BatchRenderer used after being moved from");
    return *impl_;
}

const BatchRenderer::Impl &BatchRenderer::state() const {
    if (!impl_) throw std::logic_error("neraulgraphics::BatchRenderer used after being moved from");
    return *impl_;
}

void BatchRenderer::set_instance_transforms(const float *matrices, const Memory memory) {
    Impl &r = state();
    require(r.backend->device() == Device::Cuda || memory == Memory::Host,
        "device pointers need a CUDA renderer; this one runs on the CPU");
    if (r.per_world == 0) return;
    require(matrices != nullptr, "matrices must not be null");
    if (memory == Memory::Host) {
        const size_t total = r.per_world * static_cast<size_t>(r.worlds);
        for (size_t g = 0; g < total; ++g) {
            Transform t;
            for (int k = 0; k < 12; ++k) t.m[k] = matrices[12 * g + k];
            // The message is only built for the one that fails: building it for every matrix cost more than the checking.
            if (!detail::transform_ok(t))
                detail::check_transform(t, "world " + std::to_string(g / r.per_world) + " instance " +
                    std::to_string(g % r.per_world) + " transform");
        }
    }
    r.backend->set_instance_matrices(matrices, memory);
}

void BatchRenderer::bind_bodies(const BodyBinding &binding) {
    Impl &r = state();
    const size_t n = r.per_world;
    require(binding.bodies_per_world >= 0, "bodies_per_world must be 0 or more");
    require(binding.body_of_instance.size() == n && binding.local_pose.size() == 7 * n && binding.scale.size() == 3 * n,
        "a body binding needs one body index, 7 pose floats and 3 scale floats for each of the " + std::to_string(n) +
        " instances of a world");
    for (size_t i = 0; i < n; ++i) {
        const int b = binding.body_of_instance[i];
        require(b >= -1 && b < binding.bodies_per_world, "instance " + std::to_string(i) + " is on body " + std::to_string(b) +
            ", but there are " + std::to_string(binding.bodies_per_world) + " bodies per world");
        require(all_finite(&binding.local_pose[7 * i], 7) && quaternion_is_usable(&binding.local_pose[7 * i + 3]),
            "instance " + std::to_string(i) + " has a pose with a NaN or infinite value, or a zero quaternion");
        require(all_finite(&binding.scale[3 * i], 3), "instance " + std::to_string(i) + " has a scale with a NaN or infinite value");
    }
    r.backend->bind_bodies(binding);
    r.binding = binding;
    r.bound = true;
}

void BatchRenderer::set_body_poses(const float *body_poses, const Memory memory) {
    Impl &r = state();
    require(r.bound, "set_body_poses needs bind_bodies to be called first");
    require(r.backend->device() == Device::Cuda || memory == Memory::Host,
        "device pointers need a CUDA renderer; this one runs on the CPU");
    const size_t bodies = static_cast<size_t>(r.worlds) * r.binding.bodies_per_world;
    require(bodies == 0 || body_poses != nullptr, "body_poses must not be null");
    if (memory == Memory::Host) {
        for (size_t b = 0; b < bodies; ++b)
            require(all_finite(&body_poses[7 * b], 7) && quaternion_is_usable(&body_poses[7 * b + 3]),
                "body " + std::to_string(b) + " has a NaN or infinite value, or a zero quaternion");
    }
    r.backend->set_body_poses(body_poses, memory);
}

BatchFrame BatchRenderer::render(const float *camera_matrices, const float fov, const int samples_per_pixel,
    const unsigned outputs, const int frame, const Memory camera_memory) {
    Impl &r = state();
    require(outputs != 0, "render needs at least one output");
    require((outputs & ~ALL_OUTPUTS) == 0, "unknown output bits in " + std::to_string(outputs));
    require(samples_per_pixel >= 1, "samples_per_pixel must be 1 or more, got " + std::to_string(samples_per_pixel));
    require(frame >= 0, "frame must be 0 or more, got " + std::to_string(frame));
    // Not NaN, and strictly between 0 and pi.
    require(fov > 0.0f && fov < 3.14159265f, "fov must be between 0 and pi radians, got " + std::to_string(fov));
    require(r.backend->device() == Device::Cuda || camera_memory == Memory::Host,
        "device pointers need a CUDA renderer; this one runs on the CPU");
    const size_t views = static_cast<size_t>(r.worlds) * r.cameras_per_world;
    require(camera_matrices != nullptr, "camera_matrices must not be null");
    if (camera_memory == Memory::Host)
        require(all_finite(camera_matrices, 12 * views), "a camera has a NaN or infinite value");
    return r.backend->render_batch(camera_matrices, camera_memory, fov, samples_per_pixel, outputs, frame);
}

HostBatchFrame BatchRenderer::render_host(const float *camera_matrices, const float fov, const int samples_per_pixel,
    const unsigned outputs, const int frame) {
    const BatchFrame f = render(camera_matrices, fov, samples_per_pixel, outputs, frame, Memory::Host);
    const detail::Backend &b = *state().backend;
    const size_t pixels = static_cast<size_t>(f.views) * f.width * f.height;
    HostBatchFrame out;
    out.views = f.views;
    out.width = f.width;
    out.height = f.height;
    if (f.rgb) { out.rgb.resize(pixels * 3); b.copy_to_host(f.rgb, out.rgb.data(), pixels * 3); }
    if (f.depth) { out.depth.resize(pixels); b.copy_to_host(f.depth, out.depth.data(), pixels * sizeof(float)); }
    if (f.normal) { out.normal.resize(pixels * 3); b.copy_to_host(f.normal, out.normal.data(), pixels * 3 * sizeof(float)); }
    if (f.object_id) { out.object_id.resize(pixels); b.copy_to_host(f.object_id, out.object_id.data(), pixels * sizeof(int32_t)); }
    return out;
}

RayHits BatchRenderer::cast_rays(const float *origins, const float *directions, const int rays_per_world,
    const float max_distance, const Memory input_memory) {
    Impl &r = state();
    require(rays_per_world >= 0, "rays_per_world must be 0 or more, got " + std::to_string(rays_per_world));
    const long long count = static_cast<long long>(rays_per_world) * r.worlds;
    require(count <= INT32_MAX / 3, "too many rays: " + std::to_string(count));
    require(count == 0 || (origins != nullptr && directions != nullptr), "ray origins and directions must not be null");
    require(max_distance > 0.0f, "max_distance must be positive, got " + std::to_string(max_distance));
    require(input_memory == Memory::Host || r.backend->device() == Device::Cuda,
        "device pointers need a CUDA renderer; this one runs on the CPU");
    if (input_memory == Memory::Host) require(all_finite(origins, static_cast<size_t>(count) * 3), "a ray origin has a NaN or infinite value");
    return r.backend->cast_rays_batch(origins, directions, rays_per_world, max_distance, input_memory);
}

HostRayHits BatchRenderer::cast_rays_host(const std::vector<float> &origins, const std::vector<float> &directions,
    const int rays_per_world, const float max_distance) {
    const int worlds = state().worlds;
    const size_t expected = static_cast<size_t>(rays_per_world >= 0 ? rays_per_world : 0) * worlds * 3;
    require(origins.size() == expected && directions.size() == expected,
        "origins and directions must each hold 3 floats for each of the " + std::to_string(rays_per_world) +
        " rays of each of the " + std::to_string(worlds) + " worlds");
    const RayHits hits = cast_rays(origins.data(), directions.data(), rays_per_world, max_distance, Memory::Host);
    const detail::Backend &b = *state().backend;
    HostRayHits out;
    out.distance.resize(hits.count);
    out.normal.resize(static_cast<size_t>(hits.count) * 3);
    out.object_id.resize(hits.count);
    if (hits.count > 0) {
        b.copy_to_host(hits.distance, out.distance.data(), hits.count * sizeof(float));
        b.copy_to_host(hits.normal, out.normal.data(), static_cast<size_t>(hits.count) * 3 * sizeof(float));
        b.copy_to_host(hits.object_id, out.object_id.data(), hits.count * sizeof(int32_t));
    }
    return out;
}

int BatchRenderer::width() const { return state().width; }
int BatchRenderer::height() const { return state().height; }
int BatchRenderer::world_count() const { return state().worlds; }
int BatchRenderer::cameras_per_world() const { return state().cameras_per_world; }
int BatchRenderer::view_count() const { return state().worlds * state().cameras_per_world; }
int BatchRenderer::instances_per_world() const { return static_cast<int>(state().per_world); }
Device BatchRenderer::device() const { return state().backend->device(); }
Memory BatchRenderer::memory() const { return device() == Device::Cuda ? Memory::Device : Memory::Host; }
float BatchRenderer::last_kernel_ms() const { return state().backend->last_kernel_ms(); }
void BatchRenderer::set_ambient(const float ambient) {
    require(finite(ambient) && ambient >= 0.0f, "ambient must be 0 or more, got " + std::to_string(ambient));
    state().backend->set_ambient(ambient);
}

}  // namespace neraulgraphics
