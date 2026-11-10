#ifndef NERAULGRAPHICS_RENDERER_H
#define NERAULGRAPHICS_RENDERER_H

// Host-side interface to the CUDA renderer. Callers never see a kernel launch,
// and this header pulls in no CUDA headers.
//
// Failures are reported as exceptions, never by exiting, so the program using
// the renderer decides what an error means:
//   std::invalid_argument      bad input, rejected before anything reaches the device
//   std::logic_error           a call on a moved-from Renderer
//   neraulgraphics::CudaError  the CUDA runtime refused or failed a call
// Every check runs before any state changes, so a rejected call leaves the
// Renderer exactly as it was.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// Through the package prefix, not "": version.h is generated into the build tree,
// so it sits beside this header only once installed.
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

// One scene and its frame buffers. Each instance owns its own device memory and
// scene copy, so several can live in one process. The device state sits behind
// Impl, which keeps CUDA handles out of this header.
class Renderer {
public:
    // Copies the scene and allocates the frame buffers. The default floor is no floor.
    //
    // Throws std::invalid_argument if width or height is not positive, if there are
    // more than MAX_SPHERES spheres or MAX_LIGHTS lights, if a sphere radius is not
    // positive, if a light radius is negative, or if any value is NaN or infinite.
    Renderer(int width, int height, const std::vector<Sphere> &spheres,
        const std::vector<Light> &lights, const Plane &floor = Plane{});
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

    // GPU time of the last render(), from CUDA events. Excludes the host copy.
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
