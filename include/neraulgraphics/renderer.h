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

// Where an array lives: ordinary memory, or an NVIDIA GPU's.
enum class Memory { Host, Device };

// Which images render_frame() should produce, combined with |. Asking for fewer is faster: a
// depth-only frame does no shading at all.
enum Output : unsigned {
    OUTPUT_RGB = 1,        // the picture: 3 bytes per pixel, red, green, blue
    OUTPUT_DEPTH = 2,      // distance along the camera's view axis, 1 float per pixel; +infinity where nothing is hit
    OUTPUT_NORMAL = 4,     // the surface normal in world space facing the camera, 3 floats; 0, 0, 0 where nothing is hit
    OUTPUT_OBJECT_ID = 8,  // which object: see OBJECT_SPHERE, OBJECT_FLOOR, OBJECT_INSTANCE; OBJECT_NONE where nothing is hit
};

// The images of one frame, in the renderer's own buffers: valid until the next call that renders.
// Rows run top to bottom. memory says where the pointers point; on a GPU they are device pointers
// and the work that wrote them has finished, so any CUDA code (PyTorch, Warp, JAX) can read them.
// A pointer is null for an image that was not asked for.
struct Frame {
    Memory memory;
    int device_id;  // which GPU, when memory is Device; 0 otherwise
    int width;
    int height;
    const uint8_t *rgb;
    const float *depth;
    const float *normal;
    const int32_t *object_id;
};

// The same images copied into ordinary vectors, which you own.
struct HostFrame {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
    std::vector<float> depth;
    std::vector<float> normal;
    std::vector<int32_t> object_id;
};

// What each ray of a ray query found. Arrays have one entry per ray (normal has three), in the
// renderer's own buffers, valid until the next ray query. A ray that hit nothing has distance
// +infinity, normal 0, 0, 0 and OBJECT_NONE.
struct RayHits {
    Memory memory;
    int device_id;  // which GPU, when memory is Device; 0 otherwise
    int count;
    const float *distance;
    const float *normal;
    const int32_t *object_id;
};

struct HostRayHits {
    std::vector<float> distance;
    std::vector<float> normal;
    std::vector<int32_t> object_id;
};

// How the picture is made.
enum class RenderMode {
    Classic,     // direct light from each light with hard or soft shadows, mirror reflections: fast, and what a Renderer always did
    PathTraced,  // light bounced between surfaces, soft shadows, sky light, physically based materials: realistic, and noisy until denoised
};

// The settings of RenderMode::PathTraced.
struct PathTracing {
    int bounces = 4;               // how many times light may bounce before a path ends, 0 to 32; more is more realistic and slower
    float exposure = 1.0f;         // scales the light before it is shown
    float max_radiance = 20.0f;    // a sample brighter than this is scaled down to it (0 for no limit): keeps rare very bright paths from speckling the picture
    bool tonemap = true;           // compress bright light smoothly to what a screen can show; false clips it instead
    // Denoising makes a picture from only a few samples per pixel look clean, by reusing the previous frame and
    // neighbouring pixels where the surface is the same. It follows a moving camera, but a moving object leaves a short
    // trail, since only the camera is tracked.
    bool denoise = true;
    int denoise_iterations = 5;    // passes of the spatial filter, 0 to 8; each reaches twice as far as the one before
    float temporal_blend = 0.05f;  // the smallest weight a new frame gets once there is history: lower is smoother and slower to follow change
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

    // Renders the images asked for in `outputs` (see Output). RGB is the picture render() draws,
    // with the same frame and samples_per_frame meaning; the other images are one ray through the
    // middle of each pixel, exact and never averaged, so an object's edge is not blurred into a
    // false depth. The images stay where the renderer keeps them, which on a GPU is the GPU:
    // nothing is copied, so a policy network can read them directly. See Frame.
    //
    // Throws std::invalid_argument for no outputs, an unknown output, or arguments render() rejects.
    Frame render_frame(const Camera &camera, int frame, int samples_per_frame, unsigned outputs);

    // The same, copied into vectors.
    HostFrame render_frame_host(const Camera &camera, int frame, int samples_per_frame, unsigned outputs);

    // Casts rays into the scene and reports what each hits first: a LiDAR, a depth sensor, or a
    // visibility query. origins and directions hold 3 floats per ray, and live where
    // `input_memory` says. Directions need not be unit length (a zero or non-finite one is a miss).
    // A hit farther than max_distance counts as a miss. The results are in the renderer's own memory.
    //
    // Device pointers need a CUDA renderer. Throws std::invalid_argument for a negative count, a
    // null pointer, a max_distance that is not positive, or (for host inputs) a non-finite origin.
    RayHits cast_rays(const float *origins, const float *directions, int count, float max_distance,
        Memory input_memory = Memory::Host);

    // The same from vectors of 3 floats per ray, with the results copied back into vectors.
    HostRayHits cast_rays_host(const std::vector<float> &origins, const std::vector<float> &directions,
        float max_distance);

    // Where this renderer keeps its images: Device for CUDA, Host for the CPU.
    Memory memory() const;

    // Chooses how render() and render_frame() draw the picture. RenderMode::PathTraced follows light as it
    // bounces, with physically based materials (Material's roughness, metallic and emission; the classic
    // fields are translated), the lights' soft shadows and the environment as the sky's light, and denoises the
    // result; see PathTracing. The meaning of render()'s frame changes with it: 0 starts again, and every
    // later frame adds to what the previous ones saw, following the camera if it moved, so pass an increasing
    // frame whether or not the camera moves (with denoising off, the camera must hold still). Ambient light is not
    // used in this mode, since the sky does that work. The other images (depth, normals, IDs) are unchanged.
    //
    // Throws std::invalid_argument, changing nothing, for a setting out of range or not finite.
    void set_render_mode(RenderMode mode, const PathTracing &settings = {});
    RenderMode render_mode() const;

    // The sky: what rays that leave the scene see, and, in a path-traced picture, the light it casts. A plain
    // gradient by default; see Environment. Takes effect on the next render.
    //
    // Throws std::invalid_argument, changing nothing, for a map of the wrong size or with a negative or non-finite value.
    void set_environment(const Environment &environment);

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
