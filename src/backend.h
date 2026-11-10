#ifndef NERAULGRAPHICS_BACKEND_H
#define NERAULGRAPHICS_BACKEND_H

// What Renderer needs from the thing that does the tracing. Renderer validates input and
// chooses a backend; a backend only has to hold the scene and draw it. Two exist: CUDA, in
// cuda_backend.cu, and the CPU, in cpu_backend.cpp. Both run the same rendering code (see
// backend_api.h), so they draw the same image.

#include <cstdint>
#include <memory>
#include <vector>
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
};

class Backend {
public:
    virtual ~Backend() = default;

    // Traces a frame and returns width * height 0xAARRGGBB pixels, valid until the next call.
    // Arguments are already validated.
    virtual const uint32_t *render(const Camera &camera, int frame, int samples_per_frame) = 0;

    // Transforms are one per instance and already checked.
    virtual void set_instance_transforms(const std::vector<Transform> &transforms) = 0;

    virtual void set_ambient(float ambient) = 0;
    virtual float ambient() const = 0;

    // GPU time for CUDA, wall time for the CPU.
    virtual float last_kernel_ms() const = 0;
    virtual Device device() const = 0;
};

// Throws CudaError if this library was built without CUDA, or the device refuses.
std::unique_ptr<Backend> make_cuda_backend(const SceneInput &scene);
std::unique_ptr<Backend> make_cpu_backend(const SceneInput &scene);

// Built with CUDA, and a usable device is present.
bool cuda_usable();

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_BACKEND_H
