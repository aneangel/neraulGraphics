#ifndef NERAULGRAPHICS_BACKEND_H
#define NERAULGRAPHICS_BACKEND_H

// What Renderer needs from the thing that does the tracing. Renderer validates input and
// chooses a backend; a backend only has to hold the scene and draw it. Two exist: CUDA, in
// cuda_backend.cu, and the CPU, in cpu_backend.cpp. Both run the same rendering code (see
// backend_api.h), so they draw the same image.

#include <cstdint>
#include <memory>
#include <vector>
#include <neraulgraphics/batch.h>
#include <neraulgraphics/renderer.h>
#include "mesh_build.h"

namespace neraulgraphics::detail {

// A validated scene. References are only good for the duration of the constructor call.
struct SceneInput {
    int width;
    int height;
    const std::vector<Sphere> &spheres;
    const std::vector<Light> &lights;
    const Plane &floor;
    const std::vector<Texture> &textures;
    const std::vector<Instance> &instances;
    // The meshes' acceleration structures; null when there are no instances. A backend that
    // keeps them for itself may move out of them.
    BuiltMeshes *built;
    // Batched worlds (see BatchRenderer): instances is what one world holds, and there are this many
    // worlds, each seen by this many cameras. A plain Renderer is one world with one camera.
    int worlds = 1;
    int cameras_per_world = 1;
};

class Backend {
public:
    virtual ~Backend() = default;

    // Traces a frame and returns width * height 0xAARRGGBB pixels, valid until the next call.
    // Arguments are already validated.
    virtual const uint32_t *render(const Camera &camera, int frame, int samples_per_frame) = 0;

    // The images asked for in outputs (an Output mask), left in the backend's own memory. Arguments
    // are already validated. The work has finished when this returns, so the images can be read at once.
    virtual Frame render_frame(const Camera &camera, int frame, int samples_per_frame, unsigned outputs) = 0;

    // Rays are already validated. Origins and directions are 3 floats per ray in input_memory.
    // Results are in the backend's own memory, valid until the next ray query.
    virtual RayHits cast_rays(const float *origins, const float *directions, int count, float max_distance,
        Memory input_memory) = 0;

    // Copies bytes from one of the backend's own arrays into ordinary memory.
    virtual void copy_to_host(const void *src, void *dst, size_t bytes) const = 0;

    // Transforms are one per instance and already checked.
    virtual void set_instance_transforms(const std::vector<Transform> &transforms) = 0;

    // --- Batched worlds. Pointers are in the memory named, and already validated if they are in host memory.

    // 12 floats per instance, over every world, world-major.
    virtual void set_instance_matrices(const float *matrices, Memory memory) = 0;
    virtual void bind_bodies(const BodyBinding &binding) = 0;
    // 7 floats per body, over every world, world-major.
    virtual void set_body_poses(const float *body_poses, Memory memory) = 0;
    // One 12-float camera-to-world matrix per view.
    virtual BatchFrame render_batch(const float *camera_matrices, Memory camera_memory, float fov,
        int samples_per_pixel, unsigned outputs, int frame) = 0;
    virtual RayHits cast_rays_batch(const float *origins, const float *directions, int rays_per_world,
        float max_distance, Memory input_memory) = 0;

    // Replaces the lights (already validated). Takes effect on the next frame.
    virtual void set_lights(const std::vector<Light> &lights) = 0;
    virtual void set_ambient(float ambient) = 0;
    virtual float ambient() const = 0;

    // Already validated.
    virtual void set_render_mode(RenderMode mode, const PathTracing &settings) = 0;
    virtual RenderMode render_mode() const = 0;
    virtual void set_environment(const Environment &environment) = 0;

    // GPU time for CUDA, wall time for the CPU.
    virtual float last_kernel_ms() const = 0;
    virtual Device device() const = 0;
};

// Throws CudaError if this library was built without CUDA, or the device refuses.
std::unique_ptr<Backend> make_cuda_backend(const SceneInput &scene);
std::unique_ptr<Backend> make_cpu_backend(const SceneInput &scene);

// Built with CUDA, and a usable device is present.
bool cuda_usable();

// What Device::Auto means right now (NERAULGRAPHICS_DEVICE if set, else CUDA when usable); other values pass through.
Device resolve_device(Device requested);

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_BACKEND_H
