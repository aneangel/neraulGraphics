// The CUDA backend: the scene in device memory and a kernel that runs render_pixel, one thread
// per pixel. The rendering itself is in render_core.h, shared with the CPU backend.

#include <algorithm>
#include <cstdint>
#include <string>
#include "backend.h"
#include "instances.h"
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
    render_pixel(s, accum, out, x, y, width, height, cam, frame, spp);
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
    bvh::Node *d_tlas_ = nullptr;
    int *d_tlas_instance_ = nullptr;
    device::Instance *d_instances_ = nullptr;
    float4 *d_attributes_ = nullptr;

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
        cudaFree(d_tlas_);
        cudaFree(d_tlas_instance_);
        cudaFree(d_instances_);
        cudaFree(d_attributes_);
        cudaFree(d_textures_);
        for (const cudaTextureObject_t t : texture_objects_) cudaDestroyTextureObject(t);
        for (const cudaArray_t a : texture_arrays_) cudaFreeArray(a);
        if (ev_start_) cudaEventDestroy(ev_start_);
        if (ev_stop_) cudaEventDestroy(ev_stop_);
    }
};

class CudaBackend final : public Backend, private CudaResources {
public:
    explicit CudaBackend(const SceneInput &scene) : width_(scene.width), height_(scene.height) {
        // A grid is at most 65535 blocks tall.
        if (height_ > 65535 * device::BLOCK_Y)
            throw std::invalid_argument("frame height " + std::to_string(height_) + " is too large");

        for (size_t i = 0; i < scene.spheres.size(); ++i) params_.spheres[i] = device::to_device(scene.spheres[i]);
        for (size_t i = 0; i < scene.lights.size(); ++i) params_.lights[i] = device::to_device(scene.lights[i]);
        params_.floor = device::to_device(scene.floor);
        params_.sphere_count = static_cast<int>(scene.spheres.size());
        params_.light_count = static_cast<int>(scene.lights.size());

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

        for (const Texture &tex : scene.textures) upload_texture(tex);
        if (!scene.textures.empty()) {
            CUDA_CHECK(cudaMalloc(&d_textures_, scene.textures.size() * sizeof(cudaTextureObject_t)));
            CUDA_CHECK(cudaMemcpy(d_textures_, texture_objects_.data(),
                scene.textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        }

        if (scene.built != nullptr) upload_meshes(scene);
    }

    const uint32_t *render(const Camera &camera, const int frame, const int samples_per_frame) override {
        using device::BLOCK_X;
        using device::BLOCK_Y;
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid((width_ + BLOCK_X - 1) / BLOCK_X, (height_ + BLOCK_Y - 1) / BLOCK_Y);

        CUDA_CHECK(cudaEventRecord(ev_start_));
        device::render_kernel<<<grid, block>>>(params_, d_accum_, d_pixels_, width_, height_,
            device::to_device(camera), frame, samples_per_frame);
        // Catches a launch that never started. A fault while running surfaces at the copy.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop_));

        CUDA_CHECK(cudaMemcpy(h_pixels_, d_pixels_, static_cast<size_t>(width_) * height_ * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_, ev_start_, ev_stop_));
        return h_pixels_;
    }

    void set_instance_transforms(const std::vector<Transform> &transforms) override {
        const bool rebuilt = instances_->update(transforms);
        upload_instances(rebuilt);
    }

    void set_ambient(const float ambient) override { params_.ambient = ambient; }
    float ambient() const override { return params_.ambient; }
    float last_kernel_ms() const override { return kernel_ms_; }
    Device device() const override { return Device::Cuda; }

private:
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
        instances_ = std::make_unique<InstanceSet>(built, scene.instances);
        // 2n-1 nodes at most for n instances.
        const size_t n = scene.instances.size();
        CUDA_CHECK(cudaMalloc(&d_blas_, built.nodes.size() * sizeof(bvh::Node)));
        CUDA_CHECK(cudaMalloc(&d_triangles_, built.triangles.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tlas_, 2 * n * sizeof(bvh::Node)));
        CUDA_CHECK(cudaMalloc(&d_tlas_instance_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_instances_, n * sizeof(device::Instance)));
        CUDA_CHECK(cudaMemcpy(d_blas_, built.nodes.data(), built.nodes.size() * sizeof(bvh::Node),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_triangles_, built.triangles.data(), built.triangles.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        if (!built.attributes.empty()) {
            CUDA_CHECK(cudaMalloc(&d_attributes_, built.attributes.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_attributes_, built.attributes.data(), built.attributes.size() * sizeof(float),
                cudaMemcpyHostToDevice));
        }
        upload_instances(true);

        // Pointers are stable from here: set_instance_transforms() rewrites these buffers in place.
        params_.mesh = device::MeshScene{d_tlas_, d_tlas_instance_, d_instances_, d_blas_, d_triangles_,
            d_attributes_, d_textures_, static_cast<int>(n)};
    }

    // Uploads the instance records and the top-level BVH. A refit keeps the leaf order, so the
    // instance indirection only needs uploading after a rebuild.
    void upload_instances(const bool rebuilt) {
        const size_t n = instances_->size();
        std::vector<device::Instance> records(n);
        for (size_t i = 0; i < n; ++i) {
            const InstanceRecord &r = instances_->records[i];
            for (int row = 0; row < 3; ++row)
                records[i].inv[row] = make_float4(r.inv[4 * row], r.inv[4 * row + 1], r.inv[4 * row + 2], r.inv[4 * row + 3]);
            records[i].blas_root = r.blas_root;
            records[i].texture = r.texture;
            records[i].material = device::to_device(r.material);
        }
        const bvh::Bvh &tlas = instances_->tlas;
        CUDA_CHECK(cudaMemcpy(d_tlas_, tlas.nodes.data(), tlas.nodes.size() * sizeof(bvh::Node),
            cudaMemcpyHostToDevice));
        if (rebuilt) {
            const std::vector<int> leaf_instance(tlas.order.begin(), tlas.order.end());
            CUDA_CHECK(cudaMemcpy(d_tlas_instance_, leaf_instance.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_instances_, records.data(), n * sizeof(device::Instance), cudaMemcpyHostToDevice));
    }

    int width_;
    int height_;
    device::SceneParams params_{};

    float kernel_ms_ = 0.0f;
    std::unique_ptr<InstanceSet> instances_;
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
