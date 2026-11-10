#ifndef NERAULGRAPHICS_RENDERER_H
#define NERAULGRAPHICS_RENDERER_H

// Host-side interface to the renderer. It runs on an NVIDIA GPU through CUDA when there is
// one, and on the CPU otherwise, drawing the same image either way. Callers never see a
// kernel launch, and this header pulls in no CUDA headers.
//
// Failures are reported as exceptions, never by exiting, so the program using
// the renderer decides what an error means:
//   std::invalid_argument      bad input, rejected before anything reaches the device
//   std::logic_error           a call on a moved-from Renderer
//   neraulgraphics::CudaError  the CUDA runtime refused or failed a call, or CUDA was asked
//                              for in a library built without it
// Every check runs before any state changes, so a rejected call leaves the
// Renderer exactly as it was.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// Through the package prefix, not "": version.h is generated into the build tree,
// so it sits beside this header only once installed.
#include <neraulgraphics/mesh.h>
#include <neraulgraphics/scene.h>
#include <neraulgraphics/version.h>

namespace neraulgraphics {

// A failed CUDA call. code() is the cudaError_t value. A fault inside a kernel is
// sticky: the CUDA context stays broken, so every later CUDA call fails too.
class CudaError : public std::runtime_error {
public:
    CudaError(const int code, const std::string &what) : std::runtime_error(what), code_(code) {}
    int code() const noexcept { return code_; }

private:
    int code_;
};

// Where a Renderer does its tracing.
enum class Device {
    Auto,  // CUDA if the library was built with it and a GPU is present, otherwise the CPU
    Cuda,  // an NVIDIA GPU; throws CudaError if there is none
    Cpu,   // every core of the machine; slower than a GPU, but needs nothing
};

// True if this library was built with CUDA and an NVIDIA GPU is present, so Device::Auto
// would use it. The environment variable NERAULGRAPHICS_DEVICE set to cpu or cuda overrides
// what Auto chooses (for debugging, or to leave the GPU to something else).
bool cuda_available();

// One scene and its frame buffers. Each instance owns its own device memory and
// scene copy, so several can live in one process. The device state sits behind
// Impl, which keeps CUDA handles out of this header.
class Renderer {
public:
    // Copies the scene and allocates the frame buffers. The default floor is no floor.
    //
    // meshes are shared geometry and instances place them: each Instance names a
    // mesh by its index here and carries its own Transform and Material. Any number
    // of instances may use one mesh, and the count of instances is not limited the
    // way spheres and lights are. Each mesh's BVH is built here, once.
    //
    // textures are images instances can name; see Instance::texture.
    //
    // device chooses where to trace; see Device. The CPU uses one thread per core, or
    // NERAULGRAPHICS_THREADS of them if that environment variable is set.
    //
    // Throws std::invalid_argument if width or height is not positive, if there are
    // more than MAX_SPHERES spheres or MAX_LIGHTS lights, if a sphere radius is not
    // positive, if a light radius is negative, if a mesh has no triangles or an index
    // out of range, if a mesh's normals or uvs are not one per vertex, if an instance
    // names a mesh or texture that does not exist, uses a texture on a mesh without
    // uvs, or has a singular transform, if a texture's size does not match its
    // pixels, or if any value is NaN or infinite.
    Renderer(int width, int height, const std::vector<Sphere> &spheres,
        const std::vector<Light> &lights, const Plane &floor = Plane{},
        const std::vector<Mesh> &meshes = {}, const std::vector<Instance> &instances = {},
        const std::vector<Texture> &textures = {}, Device device = Device::Auto);
    ~Renderer();

    // Owns device memory, so it can be moved but not copied.
    Renderer(Renderer &&) noexcept;
    Renderer &operator=(Renderer &&) noexcept;
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // Traces one frame. Returns width*height pixels in pinned host memory, valid
    // until the next render() on this instance. Rows run top to bottom; each pixel
    // is 0xAARRGGBB (SDL_PIXELFORMAT_ARGB8888), gamma-encoded, alpha always 0xFF.
    //
    // frame: accumulation index. Pass 0 to restart (the camera moved), then increment
    //        while it holds still and samples will keep averaging in. Must be >= 0.
    // samples_per_frame: rays per pixel this frame. 1 is fastest; higher values
    //        anti-alias while the camera is moving. Must be >= 1.
    //
    // A NaN reaching the running average would blacken that pixel until the next
    // restart, so a camera with any non-finite value is rejected as well.
    const uint32_t *render(const Camera &camera, int frame, int samples_per_frame);

    // Light that reaches every surface regardless of the lights, as a fraction of its diffuse
    // colour: a stand-in for light bounced from the rest of the scene, which this renderer does
    // not trace. 0 (the default) leaves surfaces facing away from every light black; around
    // 0.1 to 0.3 keeps detail visible there. Takes effect on the next render(), and the
    // accumulated image should be restarted with frame 0, as for any scene change.
    //
    // Throws std::invalid_argument, changing nothing, if ambient is negative, NaN or infinite.
    void set_ambient(float ambient);
    float ambient() const;

    // Moves the instances: transforms[i] replaces instance i's Transform. Only the
    // top-level BVH over instances is rebuilt (on the host, then uploaded); the
    // meshes' own BVHs are untouched, so this is the cheap per-frame update for a
    // simulation. Camera and accumulation are separate: call render() with frame 0
    // afterwards, since the image built up so far shows the old positions.
    //
    // Throws std::invalid_argument, changing nothing, if transforms.size() is not
    // the instance count or a transform is NaN, infinite or singular.
    void set_instance_transforms(const std::vector<Transform> &transforms);

    int instance_count() const;

    // The device this Renderer traces on: Cuda or Cpu, never Auto.
    Device device() const;

    // Time of the last render(): from CUDA events on the GPU, excluding the copy back to the
    // host, and wall-clock time on the CPU.
    float last_kernel_ms() const;

    int width() const;
    int height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // impl_, or std::logic_error if this Renderer has been moved from.
    Impl &state();
    const Impl &state() const;
};

}  // namespace neraulgraphics

#endif  // NERAULGRAPHICS_RENDERER_H
