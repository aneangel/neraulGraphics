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
#include "denoise_core.h"
#include "environment.h"
#include "instances.h"
#include "path_core.h"
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

// Bytes to 0..1, for data textures.
const float *linear_table() {
    static const std::vector<float> table = [] {
        std::vector<float> t(256);
        for (int i = 0; i < 256; ++i) t[i] = static_cast<float>(i) / 255.0f;
        return t;
    }();
    return table.data();
}

}  // namespace

// Repeating, bilinear, sRGB-decoded before filtering, texel centres at half-integers: the same
// as the GPU's texture sampler with normalized coordinates and linear filtering.
float4 sample_texture(const TextureHandle texture, const float u, const float v) {
    const float *decode = texture->linear ? linear_table() : srgb_table();
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
    explicit CpuBackend(SceneInput scene)
        : width_(scene.width), height_(scene.height), worlds_(scene.worlds), cameras_per_world_(scene.cameras_per_world) {
        for (size_t i = 0; i < scene.spheres.size(); ++i) params_.spheres[i] = host::to_device(scene.spheres[i]);
        for (size_t i = 0; i < scene.lights.size(); ++i) params_.lights[i] = host::to_device(scene.lights[i]);
        params_.floor = host::to_device(scene.floor);
        params_.sphere_count = static_cast<int>(scene.spheres.size());
        params_.light_count = static_cast<int>(scene.lights.size());

        const size_t pixels = static_cast<size_t>(width_) * height_;
        accum_.assign(pixels, host::float4{0.0f, 0.0f, 0.0f, 0.0f});
        pixels_.assign(pixels, 0u);
        set_environment(Environment{});
        set_render_mode(RenderMode::Classic, PathTracing{});

        // Keep the textures' bytes, and a table of handles the shading code indexes.
        for (const Texture &t : scene.textures) texture_bytes_.push_back(t.rgba);
        for (size_t i = 0; i < scene.textures.size(); ++i)
            texture_data_.push_back(host::CpuTexture{scene.textures[i].width, scene.textures[i].height,
                texture_bytes_[i].data(), scene.textures[i].linear});
        for (const host::CpuTexture &t : texture_data_) texture_handles_.push_back(&t);

        if (scene.built != nullptr) {
            BuiltMeshes &built = *scene.built;
            instances_ = std::make_unique<InstanceSet>(built, scene.instances, scene.worlds);
            blas_ = std::move(built.nodes);
            // Stored as float4 so the alignment the shading code assumes is guaranteed.
            triangles_ = to_float4(built.triangles);
            attributes_ = to_float4(built.attributes);
            records_.resize(instances_->size());
            params_.mesh = host::MeshScene{nullptr, nullptr, records_.data(), blas_.data(), triangles_.data(),
                attributes_.empty() ? nullptr : attributes_.data(),
                texture_handles_.empty() ? nullptr : texture_handles_.data(), static_cast<int>(instances_->size()),
                static_cast<int>(instances_->per_world())};
            refresh_instances();
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
                    for (int x = 0; x < width_; ++x) host::aov_pixel(params_, 0, aov, x, y, width_, height_, cam);
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
                host::cast_ray(params_, 0, origins, directions, max_distance, i, ray_distance_.data(), ray_normal_.data(),
                    ray_id_.data());
        });
        return RayHits{Memory::Host, 0, count, ray_distance_.data(), ray_normal_.data(), ray_id_.data()};
    }

    void copy_to_host(const void *src, void *dst, const size_t bytes) const override { std::memcpy(dst, src, bytes); }

    void set_instance_transforms(const std::vector<Transform> &transforms) override {
        instances_->update(transforms);
        refresh_instances();
    }

    // --- Batched worlds.

    void set_instance_matrices(const float *matrices, const Memory memory) override {
        require_host(memory);
        if (!instances_) return;
        instances_->update(matrices);
        refresh_instances();
    }

    void bind_bodies(const BodyBinding &binding) override { binding_ = binding; }

    void set_body_poses(const float *body_poses, const Memory memory) override {
        require_host(memory);
        if (!instances_) return;
        std::vector<float> matrices(instances_->size() * 12);
        body_poses_to_matrices(binding_, worlds_, body_poses, matrices.data());
        instances_->update(matrices.data());
        refresh_instances();
    }

    BatchFrame render_batch(const float *camera_matrices, const Memory camera_memory, const float fov,
        const int samples_per_pixel, const unsigned outputs, const int frame) override {
        require_host(camera_memory);
        const auto start = std::chrono::steady_clock::now();
        const int views = worlds_ * cameras_per_world_;
        const size_t pixels = static_cast<size_t>(views) * width_ * height_;
        BatchFrame f{Memory::Host, 0, views, width_, height_, nullptr, nullptr, nullptr, nullptr};
        if (outputs & OUTPUT_RGB) { batch_rgb_.resize(pixels * 3); f.rgb = batch_rgb_.data(); }
        if (outputs & OUTPUT_DEPTH) { batch_depth_.resize(pixels); f.depth = batch_depth_.data(); }
        if (outputs & OUTPUT_NORMAL) { batch_normal_.resize(pixels * 3); f.normal = batch_normal_.data(); }
        if (outputs & OUTPUT_OBJECT_ID) { batch_id_.resize(pixels); f.object_id = batch_id_.data(); }

        std::vector<host::Camera> cameras(views);
        for (int v = 0; v < views; ++v) cameras[v] = host::camera_from_matrix(camera_matrices + 12 * v, fov, width_, height_);

        // A work item is one row of one view.
        parallel_chunks(views * height_, 4, [&](const int begin, const int end) {
            for (int item = begin; item < end; ++item) {
                const int view = item / height_, y = item % height_, world = view / cameras_per_world_;
                const size_t offset = static_cast<size_t>(view) * width_ * height_;
                const host::AovBuffers aov{f.depth ? batch_depth_.data() + offset : nullptr,
                    f.normal ? batch_normal_.data() + 3 * offset : nullptr, f.object_id ? batch_id_.data() + offset : nullptr};
                for (int x = 0; x < width_; ++x) {
                    if (f.rgb)
                        host::render_pixel_rgb8(params_, world, batch_rgb_.data() + 3 * offset, x, y, width_, height_,
                            cameras[view], frame, samples_per_pixel);
                    if (aov.depth || aov.normal || aov.object_id)
                        host::aov_pixel(params_, world, aov, x, y, width_, height_, cameras[view]);
                }
            }
        });
        ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        return f;
    }

    RayHits cast_rays_batch(const float *origins, const float *directions, const int rays_per_world,
        const float max_distance, const Memory input_memory) override {
        require_host(input_memory);
        const int count = worlds_ * rays_per_world;
        ray_distance_.resize(count);
        ray_normal_.resize(static_cast<size_t>(count) * 3);
        ray_id_.resize(count);
        parallel_chunks(count, 256, [&](const int begin, const int end) {
            for (int i = begin; i < end; ++i)
                host::cast_ray(params_, i / rays_per_world, origins, directions, max_distance, i, ray_distance_.data(),
                    ray_normal_.data(), ray_id_.data());
        });
        return RayHits{Memory::Host, 0, count, ray_distance_.data(), ray_normal_.data(), ray_id_.data()};
    }

    void set_lights(const std::vector<Light> &lights) override {
        for (size_t i = 0; i < lights.size(); ++i) params_.lights[i] = host::to_device(lights[i]);
        params_.light_count = static_cast<int>(lights.size());
    }
    void set_ambient(const float ambient) override { params_.ambient = ambient; }
    float ambient() const override { return params_.ambient; }

    void set_render_mode(const RenderMode mode, const PathTracing &settings) override {
        mode_ = mode;
        pt_ = settings;
        params_.path = host::PathParams{settings.bounces, settings.max_radiance, settings.exposure, settings.tonemap ? 1 : 0};
        have_prev_ = false;  // whatever was accumulated was made differently
    }
    RenderMode render_mode() const override { return mode_; }

    void set_environment(const Environment &environment) override {
        env_tables_ = build_environment_tables(environment);
        host::EnvParams &e = params_.env;
        e.right = host::make_float3(env_tables_.right[0], env_tables_.right[1], env_tables_.right[2]);
        e.up = host::make_float3(env_tables_.up[0], env_tables_.up[1], env_tables_.up[2]);
        e.forward = host::make_float3(env_tables_.forward[0], env_tables_.forward[1], env_tables_.forward[2]);
        e.color_up = host::to_device(environment.color_up);
        e.color_down = host::to_device(environment.color_down);
        e.intensity = environment.intensity;
        e.rotation = environment.rotation;
        env_map_ = to_float4(env_tables_.rgba);
        e.map = env_map_.empty() ? nullptr : env_map_.data();
        e.map_width = environment.map_width;
        e.map_height = environment.map_height;
        e.marginal = env_tables_.marginal.empty() ? nullptr : env_tables_.marginal.data();
        e.conditional = env_tables_.conditional.empty() ? nullptr : env_tables_.conditional.data();
        e.density = env_tables_.density.empty() ? nullptr : env_tables_.density.data();
        have_prev_ = false;
    }
    float last_kernel_ms() const override { return ms_; }
    Device device() const override { return Device::Cpu; }

private:
    static void require_host(const Memory memory) {
        if (memory != Memory::Host)
            throw std::invalid_argument("device pointers need a CUDA renderer; this one runs on the CPU");
    }

    // The picture, into pixels_ (0xAARRGGBB).
    void draw(const Camera &camera, const int frame, const int samples_per_frame) {
        if (mode_ == RenderMode::PathTraced) {
            draw_path_traced(camera, frame, samples_per_frame);
            return;
        }
        const host::Camera cam = host::to_device(camera);
        parallel_chunks(height_, 4, [&](const int y0, const int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < width_; ++x)
                    host::render_pixel(params_, 0, accum_.data(), pixels_.data(), x, y, width_, height_, cam, frame,
                        samples_per_frame);
        });
    }

    // Path tracing, with or without the denoiser; see denoise_core.h for the steps.
    void draw_path_traced(const Camera &camera, const int frame, const int spp) {
        const host::Camera cam = host::to_device(camera);
        const int w = width_, h = height_;
        if (!pt_.denoise) {
            // A running average of the light, which converges as long as the camera holds still.
            parallel_chunks(h, 4, [&](const int y0, const int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) {
                        host::FirstHit first;
                        host::float3 irradiance;
                        const host::float3 radiance = host::path_pixel(params_, 0, x, y, w, h, cam, frame, spp, first, irradiance);
                        const int idx = y * w + x;
                        const host::float4 acc = frame == 0 ? host::to_float4(radiance, 1.0f) : accum_[idx] + host::to_float4(radiance, 1.0f);
                        accum_[idx] = acc;
                        pixels_[idx] = host::resolve_pixel(params_.path, host::xyz(acc) * (1.0f / acc.w), pt_.tonemap);
                    }
            });
            return;
        }

        const size_t n = static_cast<size_t>(w) * h;
        if (noisy_.size() != n) {
            const host::float4 zero{0.0f, 0.0f, 0.0f, 0.0f};
            noisy_.assign(n, zero);
            albedo_.assign(n, zero);
            for (int k = 0; k < 2; ++k) {
                gbuf_[k].assign(n, zero);
                hist_[k].assign(n, zero);
                mom_[k].assign(n, zero);
                filt_[k].assign(n, zero);
            }
            have_prev_ = false;
        }
        const int cur = flip_, old = 1 - flip_;

        // 1. Trace.
        parallel_chunks(h, 4, [&](const int y0, const int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    host::FirstHit first;
                    host::float3 irradiance;
                    host::path_pixel(params_, 0, x, y, w, h, cam, frame, spp, first, irradiance);
                    const int idx = y * w + x;
                    noisy_[idx] = host::to_float4(irradiance, 1.0f);
                    gbuf_[cur][idx] = host::make_float4(first.normal.x, first.normal.y, first.normal.z, first.depth);
                    albedo_[idx] = host::to_float4(first.albedo, 1.0f);
                }
        });

        // 2. Blend with last frame.
        const bool valid_history = frame > 0 && have_prev_;
        const float blend = host::effective_blend(cam, prev_cam_, pt_.temporal_blend, pt_.moving_blend);
        parallel_chunks(h, 4, [&](const int y0, const int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x)
                    host::temporal_accumulate_pixel(x, y, w, h, cam, prev_cam_, valid_history, noisy_.data(), gbuf_[cur].data(),
                        gbuf_[old].data(), hist_[old].data(), mom_[old].data(), hist_[cur].data(), mom_[cur].data(), filt_[0].data(),
                        blend);
        });

        // 3. Filter in space, each pass reaching twice as far as the one before.
        const host::float4 *result = filt_[0].data();
        for (int i = 0, from = 0; i < pt_.denoise_iterations; ++i, from ^= 1) {
            host::float4 *to = filt_[from ^ 1].data();
            const host::float4 *source = filt_[from].data();
            parallel_chunks(h, 4, [&](const int y0, const int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w; ++x) host::atrous_pixel(x, y, w, h, source, to, gbuf_[cur].data(), 1 << i);
            });
            result = to;
        }

        // 4. Put the colour back and show it.
        parallel_chunks(h, 4, [&](const int y0, const int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    const int idx = y * w + x;
                    const host::float3 light = host::xyz(result[idx]) * host::xyz(albedo_[idx]);
                    pixels_[idx] = host::resolve_pixel(params_.path, light, pt_.tonemap);
                }
        });

        prev_cam_ = cam;
        have_prev_ = true;
        flip_ ^= 1;
    }

    static std::vector<host::float4> to_float4(const std::vector<float> &flat) {
        std::vector<host::float4> out(flat.size() / 4);
        for (size_t i = 0; i < out.size(); ++i) out[i] = host::float4{flat[4 * i], flat[4 * i + 1], flat[4 * i + 2], flat[4 * i + 3]};
        return out;
    }

    // Converts the shared instance records to this backend's layout. The instance trees are read in
    // place from instances_, which updates them, so only the records need rewriting.
    void refresh_instances() {
        for (size_t i = 0; i < records_.size(); ++i) {
            const InstanceRecord &r = instances_->records[i];
            for (int row = 0; row < 3; ++row)
                records_[i].inv[row] = host::make_float4(r.inv[4 * row], r.inv[4 * row + 1], r.inv[4 * row + 2], r.inv[4 * row + 3]);
            records_[i].blas_root = r.blas_root;
            records_[i].texture = r.texture;
            records_[i].normal_texture = r.normal_texture;
            records_[i].orm_texture = r.orm_texture;
            records_[i].material = host::to_device(r.material);
        }
        // An update can reallocate the node array, so the pointers the trace reads must follow it.
        params_.mesh.tlas = instances_->nodes.data();
        params_.mesh.tlas_root = instances_->roots.data();
    }

    int width_;
    int height_;
    int worlds_;
    int cameras_per_world_;
    BodyBinding binding_;
    host::SceneParams params_{};
    std::vector<host::float4> accum_;
    std::vector<uint32_t> pixels_;
    float ms_ = 0.0f;

    // Path tracing: the mode, the sky's tables, and the buffers the denoiser keeps between frames.
    RenderMode mode_ = RenderMode::Classic;
    PathTracing pt_;
    EnvironmentTables env_tables_;
    std::vector<host::float4> env_map_;
    std::vector<host::float4> noisy_, albedo_, gbuf_[2], hist_[2], mom_[2], filt_[2];
    host::Camera prev_cam_{};
    bool have_prev_ = false;
    int flip_ = 0;

    // The images render_frame() hands out, and the results of the last ray query. Allocated when first asked for.
    std::vector<uint8_t> rgb_;
    std::vector<float> depth_;
    std::vector<float> normal_;
    std::vector<int32_t> id_;
    std::vector<uint8_t> batch_rgb_;
    std::vector<float> batch_depth_;
    std::vector<float> batch_normal_;
    std::vector<int32_t> batch_id_;
    std::vector<float> ray_distance_;
    std::vector<float> ray_normal_;
    std::vector<int32_t> ray_id_;

    std::unique_ptr<InstanceSet> instances_;
    std::vector<bvh::Node> blas_;
    std::vector<host::float4> triangles_;
    std::vector<host::float4> attributes_;
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
