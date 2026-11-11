// The CUDA backend: the scene in device memory and a kernel that runs render_pixel, one thread
// per pixel. The rendering itself is in render_core.h, shared with the CPU backend.

#include <algorithm>
#include <cstdint>
#include <string>
#include "backend.h"
#include "denoise_core.h"
#include "environment.h"
#include "instances.h"
#include "path_core.h"
#include "render_core.h"

// Throws neraulgraphics::CudaError naming the failed call and where it was made.
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) throw_cuda_error(err_, #call, __FILE__, __LINE__);             \
    } while (0)

namespace {

[[noreturn]] void throw_cuda_error(const cudaError_t err, const char *call, const char *file,
    const int line) {
    throw neraulgraphics::CudaError(static_cast<int>(err), std::string(file) + ":" +
        std::to_string(line) + ": " + call + " failed: " + cudaGetErrorString(err) + " (" +
        cudaGetErrorName(err) + ")");
}

}  // namespace

#include "cuda_tlas.cuh"

namespace neraulgraphics::device {
namespace {

// 8x8 puts neighbouring pixels in the same warp, which keeps ray paths coherent.
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;

// One thread per pixel. __grid_constant__ lets the device functions take s by reference;
// without it the compiler would first copy the whole scene from the parameter bank into local memory.
__global__ void render_kernel(const __grid_constant__ SceneParams s, float4 *accum, uint32_t *out,
    const int width, const int height, const Camera cam, const int frame, const int spp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;  // the grid rounds up to whole blocks
    render_pixel(s, 0, accum, out, x, y, width, height, cam, frame, spp);
}

// Depth, normal and object ID for each pixel: see aov_pixel.
__global__ void aov_kernel(const __grid_constant__ SceneParams s, const AovBuffers out, const int width,
    const int height, const Camera cam) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    aov_pixel(s, 0, out, x, y, width, height, cam);
}

// 0xAARRGGBB pixels to red, green, blue bytes.
__global__ void rgb_kernel(const uint32_t *argb, uint8_t *rgb, const int pixels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    const uint32_t p = argb[i];
    rgb[3 * i] = static_cast<uint8_t>(p >> 16);
    rgb[3 * i + 1] = static_cast<uint8_t>(p >> 8);
    rgb[3 * i + 2] = static_cast<uint8_t>(p);
}

// One thread per ray.
__global__ void ray_kernel(const __grid_constant__ SceneParams s, const float *origins, const float *directions,
    const float max_distance, const int count, float *distance, float *normal, int32_t *object_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    cast_ray(s, 0, origins, directions, max_distance, i, distance, normal, object_id);
}

// Every view of a batch: the picture and the other images for each pixel of each camera. The grid is one
// long row of tiles: the tiles of the first view, then the second's, and so on, which sidesteps the limit on a
// grid's other dimensions. The camera of a view is built here from its matrix.
__global__ void batch_kernel(const __grid_constant__ SceneParams s, const float *cameras, const float fov,
    const int width, const int height, const int cameras_per_world, const int frame, const int spp, uint8_t *rgb,
    const AovBuffers aov, const int tiles_x, const int tiles_y) {
    const int tiles = tiles_x * tiles_y;
    const int view = blockIdx.x / tiles, tile = blockIdx.x % tiles;
    const int x = (tile % tiles_x) * blockDim.x + threadIdx.x;
    const int y = (tile / tiles_x) * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Camera cam = camera_from_matrix(cameras + 12 * static_cast<size_t>(view), fov, width, height);
    const int world = view / cameras_per_world;
    const size_t offset = static_cast<size_t>(view) * width * height;
    if (rgb) render_pixel_rgb8(s, world, rgb + 3 * offset, x, y, width, height, cam, frame, spp);
    if (aov.depth || aov.normal || aov.object_id) {
        const AovBuffers mine{aov.depth ? aov.depth + offset : nullptr, aov.normal ? aov.normal + 3 * offset : nullptr,
            aov.object_id ? aov.object_id + offset : nullptr};
        aov_pixel(s, world, mine, x, y, width, height, cam);
    }
}

// Ray queries over the worlds of a batch: ray i belongs to world i / rays_per_world.
__global__ void ray_batch_kernel(const __grid_constant__ SceneParams s, const float *origins, const float *directions,
    const float max_distance, const int rays_per_world, const int count, float *distance, float *normal, int32_t *object_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    cast_ray(s, i / rays_per_world, origins, directions, max_distance, i, distance, normal, object_id);
}

// Path tracing without the denoiser: a running average of the light, tone mapped.
__global__ void path_simple_kernel(const __grid_constant__ SceneParams s, float4 *accum, uint32_t *out, const int width,
    const int height, const Camera cam, const int frame, const int spp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    FirstHit first;
    float3 irradiance;
    const float3 radiance = path_pixel(s, 0, x, y, width, height, cam, frame, spp, first, irradiance);
    const int idx = y * width + x;
    const float4 acc = frame == 0 ? to_float4(radiance, 1.0f) : accum[idx] + to_float4(radiance, 1.0f);
    accum[idx] = acc;
    out[idx] = resolve_pixel(s.path, xyz(acc) * (1.0f / acc.w), s.path.tonemap != 0);
}

// Denoised path tracing, pass 1: the noisy light with its colour divided out, and what the filter needs to know about the surface.
__global__ void path_trace_kernel(const __grid_constant__ SceneParams s, float4 *noisy, float4 *gbuf, float4 *albedo,
    const int width, const int height, const Camera cam, const int frame, const int spp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    FirstHit first;
    float3 irradiance;
    path_pixel(s, 0, x, y, width, height, cam, frame, spp, first, irradiance);
    const int idx = y * width + x;
    noisy[idx] = to_float4(irradiance, 1.0f);
    gbuf[idx] = make_float4(first.normal.x, first.normal.y, first.normal.z, first.depth);
    albedo[idx] = to_float4(first.albedo, 1.0f);
}

__global__ void temporal_kernel(const int width, const int height, const Camera cam, const Camera prev, const int valid_history,
    const float4 *noisy, const float4 *gbuf, const float4 *prev_gbuf, const float4 *hist_in, const float4 *mom_in,
    float4 *hist_out, float4 *mom_out, float4 *filtered, const float blend_min) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    temporal_accumulate_pixel(x, y, width, height, cam, prev, valid_history != 0, noisy, gbuf, prev_gbuf, hist_in, mom_in,
        hist_out, mom_out, filtered, blend_min);
}

__global__ void atrous_kernel(const int width, const int height, const float4 *in, float4 *out, const float4 *gbuf, const int step) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    atrous_pixel(x, y, width, height, in, out, gbuf, step);
}

__global__ void resolve_kernel(const PathParams path, const int width, const int height, const float4 *filtered, const float4 *albedo,
    uint32_t *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int idx = y * width + x;
    out[idx] = resolve_pixel(path, xyz(filtered[idx]) * xyz(albedo[idx]), path.tonemap != 0);
}

// Rotation matrix of a quaternion x, y, z, w (normalized first), row-major.
__device__ __forceinline__ void quat_matrix(const float *q, float *r) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    const float n = sqrtf(x * x + y * y + z * z + w * w);
    if (n > 0.0f) { x /= n; y /= n; z /= n; w /= n; } else { x = y = z = 0.0f; w = 1.0f; }
    r[0] = 1 - 2 * (y * y + z * z); r[1] = 2 * (x * y - z * w);     r[2] = 2 * (x * z + y * w);
    r[3] = 2 * (x * y + z * w);     r[4] = 1 - 2 * (x * x + z * z); r[5] = 2 * (y * z - x * w);
    r[6] = 2 * (x * z - y * w);     r[7] = 2 * (y * z + x * w);     r[8] = 1 - 2 * (x * x + y * y);
}

// One thread per instance of every world: its matrix from its body's pose, its own offset and its scale. The same
// arithmetic as body_poses_to_matrices on the host.
__global__ void bodies_to_matrices(const float *body_poses, const int *body_of_instance, const float *local_pose,
    const float *scale, const int per_world, const int bodies_per_world, const int total, float *matrices) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const int w = g / per_world, i = g % per_world;
    const float *lp = local_pose + 7 * i, *sc = scale + 3 * i;
    float rl[9], r[9], p[3] = {lp[0], lp[1], lp[2]};
    quat_matrix(lp + 3, rl);
    const int body = body_of_instance[i];
    if (body < 0) {
        for (int k = 0; k < 9; ++k) r[k] = rl[k];
    } else {
        const float *bp = body_poses + 7 * (static_cast<size_t>(w) * bodies_per_world + body);
        float rb[9];
        quat_matrix(bp + 3, rb);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col)
                r[3 * row + col] = rb[3 * row] * rl[col] + rb[3 * row + 1] * rl[3 + col] + rb[3 * row + 2] * rl[6 + col];
            p[row] = bp[row] + rb[3 * row] * lp[0] + rb[3 * row + 1] * lp[1] + rb[3 * row + 2] * lp[2];
        }
    }
    float *m = matrices + 12 * static_cast<size_t>(g);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) m[4 * row + col] = r[3 * row + col] * sc[col];
        m[4 * row + 3] = p[row];
    }
}

}  // namespace
}  // namespace neraulgraphics::device

namespace neraulgraphics::detail {
namespace {

// Everything allocated on the device, owned by a base class of its own. A destructor only runs for
// a fully constructed object, so if a CUDA call throws partway through CudaBackend's constructor,
// it is this base's destructor, not CudaBackend's, that frees what was already allocated.
struct CudaResources {
    float4 *d_accum_ = nullptr;
    uint32_t *d_pixels_ = nullptr;
    uint32_t *h_pixels_ = nullptr;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_stop_ = nullptr;

    // Meshes. The bottom-level arrays are built once and never change; the top-level
    // ones are rewritten by every set_instance_transforms().
    bvh::Node *d_blas_ = nullptr;
    float4 *d_triangles_ = nullptr;
    device::Instance *d_instances_ = nullptr;
    float4 *d_attributes_ = nullptr;

    // Batched worlds: the matrices of every instance (the tree builder reads them), the bodies that carry
    // the instances, the cameras of the last batch, and the images of a batch.
    float *d_matrices_ = nullptr;
    int *d_body_of_instance_ = nullptr;
    float *d_local_pose_ = nullptr;
    float *d_scale_ = nullptr;
    float *d_body_poses_ = nullptr;
    float *d_cameras_ = nullptr;
    int camera_capacity_ = 0;
    uint8_t *d_batch_rgb_ = nullptr;
    float *d_batch_depth_ = nullptr;
    float *d_batch_normal_ = nullptr;
    int32_t *d_batch_id_ = nullptr;

    // What render_frame() hands out and what ray queries read and write, allocated when first needed
    // and kept (a ray query grows them to the largest batch seen).
    uint8_t *d_rgb_ = nullptr;
    float *d_depth_ = nullptr;
    float *d_normal_ = nullptr;
    int32_t *d_id_ = nullptr;
    float *d_ray_origin_ = nullptr;
    float *d_ray_direction_ = nullptr;
    float *d_hit_distance_ = nullptr;
    float *d_hit_normal_ = nullptr;
    int32_t *d_hit_id_ = nullptr;
    int ray_capacity_ = 0;
    int ray_input_capacity_ = 0;

    // The sky's map and the tables that choose directions on it, and the denoiser's buffers (allocated when path tracing is first used).
    float4 *d_env_map_ = nullptr;
    float *d_env_marginal_ = nullptr;
    float *d_env_conditional_ = nullptr;
    float *d_env_density_ = nullptr;
    float4 *d_noisy_ = nullptr;
    float4 *d_albedo_ = nullptr;
    float4 *d_gbuf_[2] = {nullptr, nullptr};
    float4 *d_hist_[2] = {nullptr, nullptr};
    float4 *d_mom_[2] = {nullptr, nullptr};
    float4 *d_filt_[2] = {nullptr, nullptr};

    // d_textures_ is the array of handles the kernel indexes.
    std::vector<cudaArray_t> texture_arrays_;
    std::vector<cudaTextureObject_t> texture_objects_;
    cudaTextureObject_t *d_textures_ = nullptr;

    CudaResources() = default;
    CudaResources(const CudaResources &) = delete;
    CudaResources &operator=(const CudaResources &) = delete;

    // Destructors must not throw, so CUDA errors here are ignored. After a sticky
    // fault these calls fail anyway, and the process is about to report it.
    ~CudaResources() {
        cudaFree(d_accum_);
        cudaFree(d_pixels_);
        cudaFreeHost(h_pixels_);
        cudaFree(d_blas_);
        cudaFree(d_triangles_);
        cudaFree(d_env_map_);
        cudaFree(d_env_marginal_);
        cudaFree(d_env_conditional_);
        cudaFree(d_env_density_);
        cudaFree(d_noisy_);
        cudaFree(d_albedo_);
        for (int k = 0; k < 2; ++k) {
            cudaFree(d_gbuf_[k]);
            cudaFree(d_hist_[k]);
            cudaFree(d_mom_[k]);
            cudaFree(d_filt_[k]);
        }
        cudaFree(d_matrices_);
        cudaFree(d_body_of_instance_);
        cudaFree(d_local_pose_);
        cudaFree(d_scale_);
        cudaFree(d_body_poses_);
        cudaFree(d_cameras_);
        cudaFree(d_batch_rgb_);
        cudaFree(d_batch_depth_);
        cudaFree(d_batch_normal_);
        cudaFree(d_batch_id_);
        cudaFree(d_instances_);
        cudaFree(d_attributes_);
        cudaFree(d_rgb_);
        cudaFree(d_depth_);
        cudaFree(d_normal_);
        cudaFree(d_id_);
        cudaFree(d_ray_origin_);
        cudaFree(d_ray_direction_);
        cudaFree(d_hit_distance_);
        cudaFree(d_hit_normal_);
        cudaFree(d_hit_id_);
        cudaFree(d_textures_);
        for (const cudaTextureObject_t t : texture_objects_) cudaDestroyTextureObject(t);
        for (const cudaArray_t a : texture_arrays_) cudaFreeArray(a);
        if (ev_start_) cudaEventDestroy(ev_start_);
        if (ev_stop_) cudaEventDestroy(ev_stop_);
    }
};

class CudaBackend final : public Backend, private CudaResources {
public:
    explicit CudaBackend(const SceneInput &scene)
        : width_(scene.width), height_(scene.height), worlds_(scene.worlds), cameras_per_world_(scene.cameras_per_world) {
        // A grid is at most 65535 blocks tall.
        if (height_ > 65535 * device::BLOCK_Y)
            throw std::invalid_argument("frame height " + std::to_string(height_) + " is too large");

        for (size_t i = 0; i < scene.spheres.size(); ++i) params_.spheres[i] = device::to_device(scene.spheres[i]);
        for (size_t i = 0; i < scene.lights.size(); ++i) params_.lights[i] = device::to_device(scene.lights[i]);
        params_.floor = device::to_device(scene.floor);
        params_.sphere_count = static_cast<int>(scene.spheres.size());
        params_.light_count = static_cast<int>(scene.lights.size());

        CUDA_CHECK(cudaGetDevice(&device_id_));  // the GPU this renderer's memory lives on

        const size_t pixel_count = static_cast<size_t>(width_) * height_;
        CUDA_CHECK(cudaMalloc(&d_accum_, pixel_count * sizeof(float4)));
        // cudaMalloc does not zero, and a first render with frame > 0 adds to this.
        CUDA_CHECK(cudaMemset(d_accum_, 0, pixel_count * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_pixels_, pixel_count * sizeof(uint32_t)));
        // Pinned host memory: the DMA engine reads it directly, roughly doubling
        // device-to-host transfer speed.
        CUDA_CHECK(cudaHostAlloc(&h_pixels_, pixel_count * sizeof(uint32_t), cudaHostAllocDefault));

        CUDA_CHECK(cudaEventCreate(&ev_start_));
        CUDA_CHECK(cudaEventCreate(&ev_stop_));
        set_environment(Environment{});
        set_render_mode(RenderMode::Classic, PathTracing{});

        for (const Texture &tex : scene.textures) upload_texture(tex);
        if (!scene.textures.empty()) {
            CUDA_CHECK(cudaMalloc(&d_textures_, scene.textures.size() * sizeof(cudaTextureObject_t)));
            CUDA_CHECK(cudaMemcpy(d_textures_, texture_objects_.data(),
                scene.textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        }

        if (scene.built != nullptr) upload_meshes(scene);
    }

    const uint32_t *render(const Camera &camera, const int frame, const int samples_per_frame) override {
        CUDA_CHECK(cudaEventRecord(ev_start_));
        launch_beauty(camera, frame, samples_per_frame);
        CUDA_CHECK(cudaEventRecord(ev_stop_));

        CUDA_CHECK(cudaMemcpy(h_pixels_, d_pixels_, static_cast<size_t>(width_) * height_ * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_, ev_start_, ev_stop_));
        return h_pixels_;
    }

    Frame render_frame(const Camera &camera, const int frame, const int samples_per_frame,
        const unsigned outputs) override {
        const size_t pixels = static_cast<size_t>(width_) * height_;
        Frame f{Memory::Device, device_id_, width_, height_, nullptr, nullptr, nullptr, nullptr};
        if (outputs & OUTPUT_RGB) f.rgb = ensure(d_rgb_, pixels * 3);
        if (outputs & OUTPUT_DEPTH) f.depth = ensure(d_depth_, pixels);
        if (outputs & OUTPUT_NORMAL) f.normal = ensure(d_normal_, pixels * 3);
        if (outputs & OUTPUT_OBJECT_ID) f.object_id = ensure(d_id_, pixels);

        CUDA_CHECK(cudaEventRecord(ev_start_));
        if (outputs & OUTPUT_RGB) {
            launch_beauty(camera, frame, samples_per_frame);
            constexpr int THREADS = 256;
            device::rgb_kernel<<<(static_cast<int>(pixels) + THREADS - 1) / THREADS, THREADS>>>(d_pixels_, d_rgb_,
                static_cast<int>(pixels));
            CUDA_CHECK(cudaGetLastError());
        }
        if (outputs & (OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID)) {
            const device::AovBuffers aov{d_depth_ && (outputs & OUTPUT_DEPTH) ? d_depth_ : nullptr,
                d_normal_ && (outputs & OUTPUT_NORMAL) ? d_normal_ : nullptr,
                d_id_ && (outputs & OUTPUT_OBJECT_ID) ? d_id_ : nullptr};
            const dim3 block(device::BLOCK_X, device::BLOCK_Y);
            const dim3 grid((width_ + device::BLOCK_X - 1) / device::BLOCK_X, (height_ + device::BLOCK_Y - 1) / device::BLOCK_Y);
            device::aov_kernel<<<grid, block>>>(params_, aov, width_, height_, device::to_device(camera));
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ev_stop_));
        // The caller reads these from another library's stream, so the work must be done, not just queued.
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_, ev_start_, ev_stop_));
        return f;
    }

    RayHits cast_rays(const float *origins, const float *directions, const int count, const float max_distance,
        const Memory input_memory) override {
        if (count == 0) return RayHits{Memory::Device, device_id_, 0, nullptr, nullptr, nullptr};
        const float *o = origins, *d = directions;
        if (input_memory == Memory::Host) {
            // Grown to the largest batch seen, so a steady stream of queries allocates nothing.
            if (count > ray_input_capacity_) {
                cudaFree(d_ray_origin_);
                cudaFree(d_ray_direction_);
                d_ray_origin_ = d_ray_direction_ = nullptr;
                ray_input_capacity_ = 0;
                CUDA_CHECK(cudaMalloc(&d_ray_origin_, static_cast<size_t>(count) * 3 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ray_direction_, static_cast<size_t>(count) * 3 * sizeof(float)));
                ray_input_capacity_ = count;
            }
            CUDA_CHECK(cudaMemcpy(d_ray_origin_, origins, static_cast<size_t>(count) * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ray_direction_, directions, static_cast<size_t>(count) * 3 * sizeof(float), cudaMemcpyHostToDevice));
            o = d_ray_origin_;
            d = d_ray_direction_;
        }
        if (count > ray_capacity_) {
            cudaFree(d_hit_distance_);
            cudaFree(d_hit_normal_);
            cudaFree(d_hit_id_);
            d_hit_distance_ = d_hit_normal_ = nullptr;
            d_hit_id_ = nullptr;
            ray_capacity_ = 0;
            CUDA_CHECK(cudaMalloc(&d_hit_distance_, static_cast<size_t>(count) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hit_normal_, static_cast<size_t>(count) * 3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hit_id_, static_cast<size_t>(count) * sizeof(int32_t)));
            ray_capacity_ = count;
        }
        constexpr int THREADS = 128;
        device::ray_kernel<<<(count + THREADS - 1) / THREADS, THREADS>>>(params_, o, d, max_distance, count,
            d_hit_distance_, d_hit_normal_, d_hit_id_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return RayHits{Memory::Device, device_id_, count, d_hit_distance_, d_hit_normal_, d_hit_id_};
    }

    void copy_to_host(const void *src, void *dst, const size_t bytes) const override {
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }

    void set_instance_transforms(const std::vector<Transform> &transforms) override {
        set_instance_matrices(transforms.empty() ? nullptr : transforms.front().m, Memory::Host);
    }

    // --- Batched worlds.

    // The trees over the instances are rebuilt on the GPU from these matrices: from the caller's own array if it is
    // on the device, else from a copy of the host array.
    void set_instance_matrices(const float *matrices, const Memory memory) override {
        if (!tlas_builder_) return;
        const float *source = matrices;
        if (memory == Memory::Host) {
            CUDA_CHECK(cudaMemcpy(d_matrices_, matrices, static_cast<size_t>(tlas_builder_->total()) * 12 * sizeof(float),
                cudaMemcpyHostToDevice));
            source = d_matrices_;
        }
        tlas_builder_->build(source, d_instances_);
    }

    void bind_bodies(const BodyBinding &binding) override {
        if (!tlas_builder_) return;
        const size_t n = binding.body_of_instance.size();
        bodies_per_world_ = binding.bodies_per_world;
        // Replaces any earlier binding.
        cudaFree(d_body_of_instance_);
        cudaFree(d_local_pose_);
        cudaFree(d_scale_);
        cudaFree(d_body_poses_);
        d_body_of_instance_ = nullptr;
        d_local_pose_ = d_scale_ = d_body_poses_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_body_of_instance_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_local_pose_, 7 * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_, 3 * n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_body_of_instance_, binding.body_of_instance.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_pose_, binding.local_pose.data(), 7 * n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale_, binding.scale.data(), 3 * n * sizeof(float), cudaMemcpyHostToDevice));
        const size_t body_floats = 7 * static_cast<size_t>(worlds_) * bodies_per_world_;
        if (body_floats > 0) CUDA_CHECK(cudaMalloc(&d_body_poses_, body_floats * sizeof(float)));
    }

    void set_body_poses(const float *body_poses, const Memory memory) override {
        if (!tlas_builder_) return;
        const float *source = body_poses;
        const size_t body_floats = 7 * static_cast<size_t>(worlds_) * bodies_per_world_;
        if (memory == Memory::Host && body_floats > 0) {
            CUDA_CHECK(cudaMemcpy(d_body_poses_, body_poses, body_floats * sizeof(float), cudaMemcpyHostToDevice));
            source = d_body_poses_;
        }
        constexpr int THREADS = 128;
        const int total = tlas_builder_->total();
        device::bodies_to_matrices<<<(total + THREADS - 1) / THREADS, THREADS>>>(source, d_body_of_instance_, d_local_pose_,
            d_scale_, per_world_, bodies_per_world_, total, d_matrices_);
        CUDA_CHECK(cudaGetLastError());
        tlas_builder_->build(d_matrices_, d_instances_);
    }

    BatchFrame render_batch(const float *camera_matrices, const Memory camera_memory, const float fov,
        const int samples_per_pixel, const unsigned outputs, const int frame) override {
        const int views = worlds_ * cameras_per_world_;
        const size_t pixels = static_cast<size_t>(views) * width_ * height_;
        BatchFrame f{Memory::Device, device_id_, views, width_, height_, nullptr, nullptr, nullptr, nullptr};
        if (outputs & OUTPUT_RGB) f.rgb = ensure(d_batch_rgb_, pixels * 3);
        if (outputs & OUTPUT_DEPTH) f.depth = ensure(d_batch_depth_, pixels);
        if (outputs & OUTPUT_NORMAL) f.normal = ensure(d_batch_normal_, pixels * 3);
        if (outputs & OUTPUT_OBJECT_ID) f.object_id = ensure(d_batch_id_, pixels);

        const float *cameras = camera_matrices;
        if (camera_memory == Memory::Host) {
            if (views > camera_capacity_) {
                cudaFree(d_cameras_);
                d_cameras_ = nullptr;
                camera_capacity_ = 0;
                CUDA_CHECK(cudaMalloc(&d_cameras_, static_cast<size_t>(views) * 12 * sizeof(float)));
                camera_capacity_ = views;
            }
            CUDA_CHECK(cudaMemcpy(d_cameras_, camera_matrices, static_cast<size_t>(views) * 12 * sizeof(float), cudaMemcpyHostToDevice));
            cameras = d_cameras_;
        }

        const int tiles_x = (width_ + device::BLOCK_X - 1) / device::BLOCK_X, tiles_y = (height_ + device::BLOCK_Y - 1) / device::BLOCK_Y;
        const long long blocks = static_cast<long long>(views) * tiles_x * tiles_y;
        if (blocks > INT32_MAX) throw std::invalid_argument("too many views for one batch: " + std::to_string(views));
        const device::AovBuffers aov{f.depth ? d_batch_depth_ : nullptr, f.normal ? d_batch_normal_ : nullptr,
            f.object_id ? d_batch_id_ : nullptr};
        CUDA_CHECK(cudaEventRecord(ev_start_));
        device::batch_kernel<<<static_cast<unsigned>(blocks), dim3(device::BLOCK_X, device::BLOCK_Y)>>>(params_, cameras, fov, width_,
            height_, cameras_per_world_, frame, samples_per_pixel, f.rgb ? d_batch_rgb_ : nullptr, aov, tiles_x, tiles_y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop_));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_, ev_start_, ev_stop_));
        return f;
    }

    RayHits cast_rays_batch(const float *origins, const float *directions, const int rays_per_world,
        const float max_distance, const Memory input_memory) override {
        const int count = worlds_ * rays_per_world;
        if (count == 0) return RayHits{Memory::Device, device_id_, 0, nullptr, nullptr, nullptr};
        const float *o = origins, *d = directions;
        if (input_memory == Memory::Host) {
            if (count > ray_input_capacity_) {
                cudaFree(d_ray_origin_);
                cudaFree(d_ray_direction_);
                d_ray_origin_ = d_ray_direction_ = nullptr;
                ray_input_capacity_ = 0;
                CUDA_CHECK(cudaMalloc(&d_ray_origin_, static_cast<size_t>(count) * 3 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ray_direction_, static_cast<size_t>(count) * 3 * sizeof(float)));
                ray_input_capacity_ = count;
            }
            CUDA_CHECK(cudaMemcpy(d_ray_origin_, origins, static_cast<size_t>(count) * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ray_direction_, directions, static_cast<size_t>(count) * 3 * sizeof(float), cudaMemcpyHostToDevice));
            o = d_ray_origin_;
            d = d_ray_direction_;
        }
        if (count > ray_capacity_) {
            cudaFree(d_hit_distance_);
            cudaFree(d_hit_normal_);
            cudaFree(d_hit_id_);
            d_hit_distance_ = d_hit_normal_ = nullptr;
            d_hit_id_ = nullptr;
            ray_capacity_ = 0;
            CUDA_CHECK(cudaMalloc(&d_hit_distance_, static_cast<size_t>(count) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hit_normal_, static_cast<size_t>(count) * 3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hit_id_, static_cast<size_t>(count) * sizeof(int32_t)));
            ray_capacity_ = count;
        }
        constexpr int THREADS = 128;
        device::ray_batch_kernel<<<(count + THREADS - 1) / THREADS, THREADS>>>(params_, o, d, max_distance, rays_per_world, count,
            d_hit_distance_, d_hit_normal_, d_hit_id_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return RayHits{Memory::Device, device_id_, count, d_hit_distance_, d_hit_normal_, d_hit_id_};
    }

    void set_render_mode(const RenderMode mode, const PathTracing &settings) override {
        mode_ = mode;
        pt_ = settings;
        params_.path = device::PathParams{settings.bounces, settings.max_radiance, settings.exposure, settings.tonemap ? 1 : 0};
        have_prev_ = false;  // whatever was accumulated was made differently
    }
    RenderMode render_mode() const override { return mode_; }

    void set_environment(const Environment &environment) override {
        const EnvironmentTables t = build_environment_tables(environment);
        // Replaces the old map and tables, if any.
        cudaFree(d_env_map_);
        cudaFree(d_env_marginal_);
        cudaFree(d_env_conditional_);
        cudaFree(d_env_density_);
        d_env_map_ = nullptr;
        d_env_marginal_ = d_env_conditional_ = d_env_density_ = nullptr;
        if (!t.rgba.empty()) {
            CUDA_CHECK(cudaMalloc(&d_env_map_, t.rgba.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_env_map_, t.rgba.data(), t.rgba.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_env_marginal_, t.marginal.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_env_marginal_, t.marginal.data(), t.marginal.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_env_conditional_, t.conditional.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_env_conditional_, t.conditional.data(), t.conditional.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_env_density_, t.density.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_env_density_, t.density.data(), t.density.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        device::EnvParams &e = params_.env;
        e.right = make_float3(t.right[0], t.right[1], t.right[2]);
        e.up = make_float3(t.up[0], t.up[1], t.up[2]);
        e.forward = make_float3(t.forward[0], t.forward[1], t.forward[2]);
        e.color_up = device::to_device(environment.color_up);
        e.color_down = device::to_device(environment.color_down);
        e.intensity = environment.intensity;
        e.rotation = environment.rotation;
        e.map = d_env_map_;
        e.map_width = environment.map_width;
        e.map_height = environment.map_height;
        e.marginal = d_env_marginal_;
        e.conditional = d_env_conditional_;
        e.density = d_env_density_;
        have_prev_ = false;
    }

    void set_ambient(const float ambient) override { params_.ambient = ambient; }
    float ambient() const override { return params_.ambient; }
    float last_kernel_ms() const override { return kernel_ms_; }
    Device device() const override { return Device::Cuda; }

private:
    // The picture, into d_pixels_ (0xAARRGGBB). Queued, not waited for.
    void launch_beauty(const Camera &camera, const int frame, const int samples_per_frame) {
        using device::BLOCK_X;
        using device::BLOCK_Y;
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid((width_ + BLOCK_X - 1) / BLOCK_X, (height_ + BLOCK_Y - 1) / BLOCK_Y);
        if (mode_ == RenderMode::PathTraced) {
            launch_path_traced(camera, frame, samples_per_frame, block, grid);
            return;
        }
        device::render_kernel<<<grid, block>>>(params_, d_accum_, d_pixels_, width_, height_,
            device::to_device(camera), frame, samples_per_frame);
        // Catches a launch that never started. A fault while running surfaces at the next sync.
        CUDA_CHECK(cudaGetLastError());
    }

    // Path tracing, with or without the denoiser; see denoise_core.h for the steps.
    void launch_path_traced(const Camera &camera, const int frame, const int spp, const dim3 block, const dim3 grid) {
        const device::Camera cam = device::to_device(camera);
        if (!pt_.denoise) {
            device::path_simple_kernel<<<grid, block>>>(params_, d_accum_, d_pixels_, width_, height_, cam, frame, spp);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const size_t n = static_cast<size_t>(width_) * height_;
        if (d_noisy_ == nullptr) {
            ensure(d_noisy_, n);
            ensure(d_albedo_, n);
            for (int k = 0; k < 2; ++k) {
                ensure(d_gbuf_[k], n);
                ensure(d_hist_[k], n);
                ensure(d_mom_[k], n);
                ensure(d_filt_[k], n);
            }
            have_prev_ = false;
        }
        const int cur = flip_, old = 1 - flip_;
        device::path_trace_kernel<<<grid, block>>>(params_, d_noisy_, d_gbuf_[cur], d_albedo_, width_, height_, cam, frame, spp);
        const int valid_history = frame > 0 && have_prev_ ? 1 : 0;
        device::temporal_kernel<<<grid, block>>>(width_, height_, cam, prev_cam_, valid_history, d_noisy_, d_gbuf_[cur], d_gbuf_[old],
            d_hist_[old], d_mom_[old], d_hist_[cur], d_mom_[cur], d_filt_[0], pt_.temporal_blend);
        const float4 *result = d_filt_[0];
        for (int i = 0, from = 0; i < pt_.denoise_iterations; ++i, from ^= 1) {
            device::atrous_kernel<<<grid, block>>>(width_, height_, d_filt_[from], d_filt_[from ^ 1], d_gbuf_[cur], 1 << i);
            result = d_filt_[from ^ 1];
        }
        device::resolve_kernel<<<grid, block>>>(params_.path, width_, height_, result, d_albedo_, d_pixels_);
        CUDA_CHECK(cudaGetLastError());
        prev_cam_ = cam;
        have_prev_ = true;
        flip_ ^= 1;
    }

    // Allocates a buffer the first time it is wanted; its size never changes after that.
    template <class T>
    T *ensure(T *&buffer, const size_t count) {
        if (buffer == nullptr) CUDA_CHECK(cudaMalloc(&buffer, count * sizeof(T)));
        return buffer;
    }

    // A CUDA array behind a texture object, which does the filtering and the sRGB decode in hardware.
    void upload_texture(const Texture &tex) {
        cudaArray_t array = nullptr;
        const cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&array, &format, tex.width, tex.height));
        texture_arrays_.push_back(array);  // owned from here, so freed if a later call throws
        const size_t pitch = static_cast<size_t>(tex.width) * 4;
        CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, tex.rgba.data(), pitch, pitch, tex.height,
            cudaMemcpyHostToDevice));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;
        cudaTextureDesc sampler{};
        sampler.addressMode[0] = cudaAddressModeWrap;  // uvs outside 0..1 repeat
        sampler.addressMode[1] = cudaAddressModeWrap;
        sampler.filterMode = cudaFilterModeLinear;
        sampler.readMode = cudaReadModeNormalizedFloat;
        sampler.sRGB = 1;  // decoded to linear light before filtering, which is what blending needs
        sampler.normalizedCoords = 1;
        cudaTextureObject_t object = 0;
        CUDA_CHECK(cudaCreateTextureObject(&object, &resource, &sampler, nullptr));
        texture_objects_.push_back(object);
    }

    void upload_meshes(const SceneInput &scene) {
        const BuiltMeshes &built = *scene.built;
        // The records' fixed parts (mesh root, texture, material) and a first set of inverses come from the host
        // set; the trees are never built on the host: the GPU builds them, from the matrices, every update.
        InstanceSet initial(built, scene.instances, scene.worlds, false);
        const size_t n = initial.size();
        per_world_ = static_cast<int>(initial.per_world());
        CUDA_CHECK(cudaMalloc(&d_blas_, built.nodes.size() * sizeof(bvh::Node)));
        CUDA_CHECK(cudaMalloc(&d_triangles_, built.triangles.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_instances_, n * sizeof(device::Instance)));
        CUDA_CHECK(cudaMalloc(&d_matrices_, n * 12 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_blas_, built.nodes.data(), built.nodes.size() * sizeof(bvh::Node), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_triangles_, built.triangles.data(), built.triangles.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (!built.attributes.empty()) {
            CUDA_CHECK(cudaMalloc(&d_attributes_, built.attributes.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_attributes_, built.attributes.data(), built.attributes.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        std::vector<device::Instance> records(n);
        for (size_t i = 0; i < n; ++i) {
            const InstanceRecord &r = initial.records[i];
            for (int row = 0; row < 3; ++row)
                records[i].inv[row] = make_float4(r.inv[4 * row], r.inv[4 * row + 1], r.inv[4 * row + 2], r.inv[4 * row + 3]);
            records[i].blas_root = r.blas_root;
            records[i].texture = r.texture;
            records[i].material = device::to_device(r.material);
        }
        CUDA_CHECK(cudaMemcpy(d_instances_, records.data(), n * sizeof(device::Instance), cudaMemcpyHostToDevice));

        // Each instance's mesh box in the mesh's own space, which the builder transforms into the world.
        std::vector<float> local_bounds;
        for (const Instance &inst : scene.instances) {
            const bvh::Aabb &b = built.bounds[inst.mesh];
            local_bounds.insert(local_bounds.end(), {b.lo[0], b.lo[1], b.lo[2], b.hi[0], b.hi[1], b.hi[2]});
        }
        tlas_builder_ = std::make_unique<device::tlas::Builder>(scene.worlds, per_world_, local_bounds);

        // Every world starts with the transforms the instances were given.
        std::vector<float> matrices;
        matrices.reserve(n * 12);
        for (int w = 0; w < scene.worlds; ++w)
            for (const Instance &inst : scene.instances) matrices.insert(matrices.end(), inst.transform.m, inst.transform.m + 12);
        set_instance_matrices(matrices.data(), Memory::Host);

        // Pointers are stable from here: every update rewrites these buffers in place.
        params_.mesh = device::MeshScene{tlas_builder_->nodes, tlas_builder_->roots, d_instances_, d_blas_, d_triangles_,
            d_attributes_, d_textures_, static_cast<int>(n), per_world_};
    }

    int width_;
    int height_;
    int worlds_;
    int cameras_per_world_;
    int per_world_ = 0;
    int bodies_per_world_ = 0;
    int device_id_ = 0;
    device::SceneParams params_{};
    RenderMode mode_ = RenderMode::Classic;
    PathTracing pt_;
    device::Camera prev_cam_{};
    bool have_prev_ = false;
    int flip_ = 0;
    std::unique_ptr<device::tlas::Builder> tlas_builder_;

    float kernel_ms_ = 0.0f;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const SceneInput &scene) { return std::make_unique<CudaBackend>(scene); }

bool cuda_usable() {
    int count = 0;
    const bool ok = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    cudaGetLastError();  // a failed probe is not an error to report later
    return ok;
}

}  // namespace neraulgraphics::detail
