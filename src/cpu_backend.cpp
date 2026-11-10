// The CPU backend: the same rendering code as the CUDA kernel (render_core.h), compiled as plain
// C++ and run for every pixel across all the machine's cores. It needs no GPU, no driver and no
// CUDA toolkit, which is what lets the library work on any machine.

#define NG_BACKEND_CPU  // before any header: see backend_api.h

// Everything the headers below need, included first and outside any namespace.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <neraulgraphics/renderer.h>
#include "backend.h"
#include "bvh.h"
#include "instances.h"
#include "render_core.h"

namespace neraulgraphics::host {

namespace {

// 8-bit sRGB to linear light, exactly, so the CPU decodes what the GPU's texture unit does.
const float *srgb_table() {
    static const std::vector<float> table = [] {
        std::vector<float> t(256);
        for (int i = 0; i < 256; ++i) {
            const double c = i / 255.0;
            t[i] = static_cast<float>(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
        }
        return t;
    }();
    return table.data();
}

}  // namespace

// Repeating, bilinear, sRGB-decoded before filtering, texel centres at half-integers: the same
// as the GPU's texture sampler with normalized coordinates and linear filtering.
float4 sample_texture(const TextureHandle texture, const float u, const float v) {
    const float *decode = srgb_table();
    const int w = texture->width, h = texture->height;
    // Wrap into [0, size) before turning into an index, so a huge coordinate cannot overflow an int.
    const float x = u * static_cast<float>(w) - 0.5f, y = v * static_cast<float>(h) - 0.5f;
    const float xr = x - static_cast<float>(w) * std::floor(x / static_cast<float>(w));
    const float yr = y - static_cast<float>(h) * std::floor(y / static_cast<float>(h));
    const int x0 = std::min(static_cast<int>(xr), w - 1), y0 = std::min(static_cast<int>(yr), h - 1);
    const float tx = xr - static_cast<float>(x0), ty = yr - static_cast<float>(y0);
    const int x1 = (x0 + 1) % w, y1 = (y0 + 1) % h;

    auto texel = [&](const int px, const int py, float out[4]) {
        const uint8_t *p = texture->rgba + (static_cast<size_t>(py) * w + px) * 4;
        out[0] = decode[p[0]];
        out[1] = decode[p[1]];
        out[2] = decode[p[2]];
        out[3] = static_cast<float>(p[3]) * (1.0f / 255.0f);  // alpha is linear
    };
    float a[4], b[4], c[4], d[4];
    texel(x0, y0, a);
    texel(x1, y0, b);
    texel(x0, y1, c);
    texel(x1, y1, d);
    float out[4];
    for (int i = 0; i < 4; ++i) {
        const float top = a[i] + (b[i] - a[i]) * tx, bottom = c[i] + (d[i] - c[i]) * tx;
        out[i] = top + (bottom - top) * ty;
    }
    return make_float4(out[0], out[1], out[2], out[3]);
}

}  // namespace neraulgraphics::host

namespace neraulgraphics::detail {
namespace {

int thread_count(const int work_items) {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char *env = std::getenv("NERAULGRAPHICS_THREADS")) {
        const int requested = std::atoi(env);
        if (requested > 0) n = requested;
    }
    return std::max(1, std::min(n, work_items));
}

// Runs fn(begin, end) over [0, total) in chunks, spread across the threads. Chunks are handed out as
// threads finish, so a core that is done early takes more. The calling thread works too.
template <class Fn>
void parallel_chunks(const int total, const int chunk, Fn &&fn) {
    std::atomic<int> next{0};
    auto work = [&] {
        for (;;) {
            const int begin = next.fetch_add(chunk);
            if (begin >= total) return;
            fn(begin, std::min(begin + chunk, total));
        }
    };
    const int threads = thread_count((total + chunk - 1) / chunk);
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(work);
    work();
    for (std::thread &t : pool) t.join();
}

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(SceneInput scene) : width_(scene.width), height_(scene.height) {
        for (size_t i = 0; i < scene.spheres.size(); ++i) params_.spheres[i] = host::to_device(scene.spheres[i]);
        for (size_t i = 0; i < scene.lights.size(); ++i) params_.lights[i] = host::to_device(scene.lights[i]);
        params_.floor = host::to_device(scene.floor);
        params_.sphere_count = static_cast<int>(scene.spheres.size());
        params_.light_count = static_cast<int>(scene.lights.size());

        const size_t pixels = static_cast<size_t>(width_) * height_;
        accum_.assign(pixels, host::float4{0.0f, 0.0f, 0.0f, 0.0f});
        pixels_.assign(pixels, 0u);

        // Keep the textures' bytes, and a table of handles the shading code indexes.
        for (const Texture &t : scene.textures) texture_bytes_.push_back(t.rgba);
        for (size_t i = 0; i < scene.textures.size(); ++i)
            texture_data_.push_back(host::CpuTexture{scene.textures[i].width, scene.textures[i].height,
                texture_bytes_[i].data()});
        for (const host::CpuTexture &t : texture_data_) texture_handles_.push_back(&t);

        if (scene.built != nullptr) {
            BuiltMeshes &built = *scene.built;
            instances_ = std::make_unique<InstanceSet>(built, scene.instances);
            blas_ = std::move(built.nodes);
            // Stored as float4 so the alignment the shading code assumes is guaranteed.
            triangles_ = to_float4(built.triangles);
            attributes_ = to_float4(built.attributes);
            tlas_instance_.resize(scene.instances.size());
            records_.resize(scene.instances.size());
            refresh_instances(true);
            params_.mesh = host::MeshScene{instances_->tlas.nodes.data(), tlas_instance_.data(), records_.data(),
                blas_.data(), triangles_.data(), attributes_.empty() ? nullptr : attributes_.data(),
                texture_handles_.empty() ? nullptr : texture_handles_.data(), static_cast<int>(scene.instances.size())};
        }
    }

    const uint32_t *render(const Camera &camera, const int frame, const int samples_per_frame) override {
        const auto start = std::chrono::steady_clock::now();
        draw(camera, frame, samples_per_frame);
        ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        return pixels_.data();
    }

    Frame render_frame(const Camera &camera, const int frame, const int samples_per_frame,
        const unsigned outputs) override {
        const auto start = std::chrono::steady_clock::now();
        const size_t pixels = static_cast<size_t>(width_) * height_;
        Frame f{Memory::Host, 0, width_, height_, nullptr, nullptr, nullptr, nullptr};

        if (outputs & OUTPUT_RGB) {
            draw(camera, frame, samples_per_frame);
            rgb_.resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                rgb_[3 * i] = static_cast<uint8_t>(pixels_[i] >> 16);
                rgb_[3 * i + 1] = static_cast<uint8_t>(pixels_[i] >> 8);
                rgb_[3 * i + 2] = static_cast<uint8_t>(pixels_[i]);
            }
            f.rgb = rgb_.data();
        }
        host::AovBuffers aov{nullptr, nullptr, nullptr};
        if (outputs & OUTPUT_DEPTH) { depth_.resize(pixels); aov.depth = depth_.data(); f.depth = depth_.data(); }
        if (outputs & OUTPUT_NORMAL) { normal_.resize(pixels * 3); aov.normal = normal_.data(); f.normal = normal_.data(); }
        if (outputs & OUTPUT_OBJECT_ID) { id_.resize(pixels); aov.object_id = id_.data(); f.object_id = id_.data(); }
        if (aov.depth || aov.normal || aov.object_id) {
            const host::Camera cam = host::to_device(camera);
            parallel_chunks(height_, 4, [&](const int y0, const int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < width_; ++x) host::aov_pixel(params_, aov, x, y, width_, height_, cam);
            });
        }
        ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        return f;
    }

    RayHits cast_rays(const float *origins, const float *directions, const int count, const float max_distance,
        const Memory) override {
        ray_distance_.resize(count);
        ray_normal_.resize(static_cast<size_t>(count) * 3);
        ray_id_.resize(count);
        parallel_chunks(count, 256, [&](const int begin, const int end) {
            for (int i = begin; i < end; ++i)
                host::cast_ray(params_, origins, directions, max_distance, i, ray_distance_.data(), ray_normal_.data(),
                    ray_id_.data());
        });
        return RayHits{Memory::Host, 0, count, ray_distance_.data(), ray_normal_.data(), ray_id_.data()};
    }

    void copy_to_host(const void *src, void *dst, const size_t bytes) const override { std::memcpy(dst, src, bytes); }

    void set_instance_transforms(const std::vector<Transform> &transforms) override {
        refresh_instances(instances_->update(transforms));
    }

    void set_ambient(const float ambient) override { params_.ambient = ambient; }
    float ambient() const override { return params_.ambient; }
    float last_kernel_ms() const override { return ms_; }
    Device device() const override { return Device::Cpu; }

private:
    // The picture, into pixels_ (0xAARRGGBB).
    void draw(const Camera &camera, const int frame, const int samples_per_frame) {
        const host::Camera cam = host::to_device(camera);
        parallel_chunks(height_, 4, [&](const int y0, const int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < width_; ++x)
                    host::render_pixel(params_, accum_.data(), pixels_.data(), x, y, width_, height_, cam, frame,
                        samples_per_frame);
        });
    }

    static std::vector<host::float4> to_float4(const std::vector<float> &flat) {
        std::vector<host::float4> out(flat.size() / 4);
        for (size_t i = 0; i < out.size(); ++i) out[i] = host::float4{flat[4 * i], flat[4 * i + 1], flat[4 * i + 2], flat[4 * i + 3]};
        return out;
    }

    // Converts the shared instance records to this backend's layout. The top-level BVH's nodes
    // are read in place from instances_, which refits them, so only the records need rewriting.
    void refresh_instances(const bool rebuilt) {
        for (size_t i = 0; i < records_.size(); ++i) {
            const InstanceRecord &r = instances_->records[i];
            for (int row = 0; row < 3; ++row)
                records_[i].inv[row] = host::make_float4(r.inv[4 * row], r.inv[4 * row + 1], r.inv[4 * row + 2], r.inv[4 * row + 3]);
            records_[i].blas_root = r.blas_root;
            records_[i].texture = r.texture;
            records_[i].material = host::to_device(r.material);
        }
        if (rebuilt) std::copy(instances_->tlas.order.begin(), instances_->tlas.order.end(), tlas_instance_.begin());
        // A rebuild replaces the node array, so the pointer the trace reads must follow it.
        params_.mesh.tlas = instances_->tlas.nodes.data();
    }

    int width_;
    int height_;
    host::SceneParams params_{};
    std::vector<host::float4> accum_;
    std::vector<uint32_t> pixels_;
    float ms_ = 0.0f;

    // The images render_frame() hands out, and the results of the last ray query. Allocated when first asked for.
    std::vector<uint8_t> rgb_;
    std::vector<float> depth_;
    std::vector<float> normal_;
    std::vector<int32_t> id_;
    std::vector<float> ray_distance_;
    std::vector<float> ray_normal_;
    std::vector<int32_t> ray_id_;

    std::unique_ptr<InstanceSet> instances_;
    std::vector<bvh::Node> blas_;
    std::vector<host::float4> triangles_;
    std::vector<host::float4> attributes_;
    std::vector<int> tlas_instance_;
    std::vector<host::Instance> records_;

    std::vector<std::vector<uint8_t>> texture_bytes_;
    std::vector<host::CpuTexture> texture_data_;
    std::vector<host::TextureHandle> texture_handles_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const SceneInput &scene) {
    return std::make_unique<CpuBackend>(scene);
}

}  // namespace neraulgraphics::detail
