#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <raytracer/render_gpu.h>

// Throws gpu::CudaError naming the failed call and where it was made.
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) throw_cuda_error(err_, #call, __FILE__, __LINE__);             \
    } while (0)

namespace {

constexpr int MAX_DEPTH = 4;
constexpr float MIN_THROUGHPUT = 1e-3f;
constexpr float SURFACE_EPS = 1e-3f;

// 8x8 puts neighbouring pixels in the same warp, which keeps ray paths coherent.
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;

// Everything the kernel reads besides the frame buffers. It is passed by value,
// and kernel parameters live in constant memory, so every thread in a warp
// reading the same sphere still gets it in one broadcast. Each Renderer
// launches with its own copy, which is what lets several coexist.
struct SceneParams {
    Sphere spheres[MAX_SPHERES];
    Light lights[MAX_LIGHTS];
    Plane floor;
    int sphere_count;
    int light_count;
};

[[noreturn]] void throw_cuda_error(const cudaError_t err, const char *call, const char *file,
    const int line) {
    throw gpu::CudaError(static_cast<int>(err), std::string(file) + ":" + std::to_string(line) +
        ": " + call + " failed: " + cudaGetErrorString(err) + " (" + cudaGetErrorName(err) + ")");
}

struct Hit {
    float3 point;
    float3 normal;
    Material material;
    float t;
};

// Analytic ray-sphere test. t is the distance along dir, which is normalized.
__device__ bool sphere_hit(const Sphere &sphere, const float3 &origin, const float3 &dir, float &t) {
    const float3 L = sphere.center - origin;
    const float tca = dot(L, dir);
    const float r2 = sphere.radius * sphere.radius;
    const float d2 = dot(L, L) - tca * tca;
    if (d2 > r2) return false;
    const float thc = sqrtf(r2 - d2);
    t = tca - thc;                // front of the sphere
    if (t < 0.0f) t = tca + thc;  // back, only when the origin is inside
    return t >= 0.0f;
}

// Closest-hit query. Records an index and resolves the surface once at the end.
__device__ bool scene_intersect(const SceneParams &s, const float3 &origin, const float3 &dir, Hit &hit) {
    float best = 1e30f;
    int best_sphere = -1;

    for (int i = 0; i < s.sphere_count; ++i) {
        float t;
        if (sphere_hit(s.spheres[i], origin, dir, t) && t < best) {
            best = t;
            best_sphere = i;
        }
    }

    bool hit_floor = false;
    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (s.floor.y - origin.y) / dir.y; d > 0.0f && d < best) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < s.floor.half_extent_x && pt.z > s.floor.z_min && pt.z < s.floor.z_max) {
                best = d;
                hit_floor = true;
            }
        }
    }

    if (best_sphere < 0 && !hit_floor) return false;

    hit.t = best;
    hit.point = origin + dir * best;

    if (hit_floor) {
        hit.normal = make_float3(0.0f, 1.0f, 0.0f);
        hit.material = s.floor.material;
        const int cx = static_cast<int>(floorf(hit.point.x * s.floor.checker_scale));
        const int cz = static_cast<int>(floorf(hit.point.z * s.floor.checker_scale));
        const float3 tint = ((cx + cz) & 1) ? make_float3(1.0f, 1.0f, 1.0f) : make_float3(1.0f, 0.7f, 0.3f);
        hit.material.diffuse_color = tint * 0.3f;
    } else {
        hit.normal = normalize(hit.point - s.spheres[best_sphere].center);
        hit.material = s.spheres[best_sphere].material;
    }
    return true;
}

// Any-hit query for shadow rays: returns on the first occluder found.
__device__ bool scene_occluded(const SceneParams &s, const float3 &origin, const float3 &dir, const float max_dist) {
    for (int i = 0; i < s.sphere_count; ++i) {
        float t;
        if (sphere_hit(s.spheres[i], origin, dir, t) && t < max_dist) return true;
    }

    if (fabsf(dir.y) > 1e-3f) {
        if (const float d = (s.floor.y - origin.y) / dir.y; d > 0.0f && d < max_dist) {
            const float3 pt = origin + dir * d;
            if (fabsf(pt.x) < s.floor.half_extent_x && pt.z > s.floor.z_min && pt.z < s.floor.z_max)
                return true;
        }
    }
    return false;
}

// Vertical gradient background.
__device__ float3 sky(const float3 &dir) {
    const float t = saturate(dir.y * 0.5f + 0.5f);
    return lerp(make_float3(0.35f, 0.52f, 0.70f), make_float3(0.10f, 0.26f, 0.60f), t);
}

// Nudges a new ray off the surface it started on to avoid self-intersection.
__device__ float3 offset_origin(const float3 &point, const float3 &normal, const float3 &dir) {
    return point + normal * (dot(dir, normal) < 0.0f ? -SURFACE_EPS : SURFACE_EPS);
}

// Iterative path loop. `throughput` is the running product of reflectivities, so
// the loop can stop once further bounces cannot affect the result.
__device__ float3 trace(const SceneParams &s, float3 origin, float3 dir, RNG &rng) {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float throughput = 1.0f;

    for (int depth = 0;; ++depth) {
        Hit hit;
        if (depth > MAX_DEPTH || !scene_intersect(s, origin, dir, hit)) {
            radiance += sky(dir) * throughput;
            break;
        }

        float diffuse = 0.0f;
        float specular = 0.0f;
        for (int i = 0; i < s.light_count; ++i) {
            // Sampling a random point on the light is what softens the shadow.
            const float3 lp = sample_sphere_light(s.lights[i].position, s.lights[i].radius, rng);
            const float3 to_light = lp - hit.point;
            const float light_distance = length(to_light);
            const float3 light_dir = to_light * (1.0f / light_distance);

            if (scene_occluded(s, offset_origin(hit.point, hit.normal, light_dir), light_dir,
                light_distance))
                continue;

            diffuse += s.lights[i].intensity * fmaxf(0.0f, dot(light_dir, hit.normal));
            specular += fast_pow(fmaxf(0.0f, dot(reflect(light_dir, hit.normal), dir)),
                            hit.material.specular_exponent) * s.lights[i].intensity;
        }

        const float3 direct = hit.material.diffuse_color * (diffuse * hit.material.diffuse_albedo) +
            make_float3(1.0f, 1.0f, 1.0f) * (specular * hit.material.specular_albedo);
        radiance += direct * throughput;

        throughput *= hit.material.reflectivity;
        if (throughput < MIN_THROUGHPUT) break;

        dir = reflect(dir, hit.normal);
        origin = offset_origin(hit.point, hit.normal, dir);
    }
    return radiance;
}

// One thread per pixel: jitter, trace, accumulate, convert, pack.
// __grid_constant__ lets the device functions take s by reference; without it the
// compiler would first copy the whole scene from the parameter bank into local memory.
__global__ void render_kernel(const __grid_constant__ SceneParams s, float4 *accum, uint32_t *out,
    const int width, const int height, const Camera cam, const int frame, const int spp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;  // the grid rounds up to whole blocks

    const int idx = y * width + x;
    RNG rng = make_rng(idx, frame);

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < spp; ++sample) {
        const float jx = rand_float(rng);
        const float jy = rand_float(rng);
        sum += trace(s, cam.position,
            camera_ray(cam, static_cast<float>(x) + jx, static_cast<float>(y) + jy, width, height), rng);
    }
    sum = sum * (1.0f / static_cast<float>(spp));

    // Running average across frames. .w holds the sample count.
    const float4 acc = frame == 0 ? to_float4(sum, 1.0f) : accum[idx] + to_float4(sum, 1.0f);
    accum[idx] = acc;

    // Gamma only. This scene has little range above 1.0, so a tone map would
    // just flatten contrast; reinstate one if the lighting starts to clip.
    const float3 average = xyz(acc) * (1.0f / acc.w);
    out[idx] = pack_argb8888(linear_to_srgb(average));
}

}  // namespace


namespace {

// Input checks. They live here rather than in a header because main.cpp is built
// with -ffast-math, which lets the compiler assume NaN never occurs and delete
// std::isfinite. nvcc compiles this file's host code without that flag.

void require(const bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument(message);
}

bool finite(const float v) { return std::isfinite(v); }

bool finite(const float3 &v) { return finite(v.x) && finite(v.y) && finite(v.z); }

bool finite(const Material &m) {
    return finite(m.diffuse_color) && finite(m.diffuse_albedo) && finite(m.specular_albedo) &&
        finite(m.reflectivity) && finite(m.specular_exponent);
}

void check_frame_size(const int width, const int height) {
    require(width > 0 && height > 0, "frame size must be positive, got " + std::to_string(width) +
        "x" + std::to_string(height));
    // The kernel indexes pixels with an int, and a grid is at most 65535 blocks tall.
    require(static_cast<long long>(width) * height <= INT32_MAX,
        "frame of " + std::to_string(width) + "x" + std::to_string(height) + " pixels is too large");
    require(height <= 65535 * BLOCK_Y, "frame height " + std::to_string(height) + " is too large");
}

void check_scene(const std::vector<Sphere> &spheres, const std::vector<Light> &lights,
    const Plane &floor) {
    require(spheres.size() <= MAX_SPHERES, "scene has " + std::to_string(spheres.size()) +
        " spheres, at most " + std::to_string(MAX_SPHERES) + " fit");
    require(lights.size() <= MAX_LIGHTS, "scene has " + std::to_string(lights.size()) +
        " lights, at most " + std::to_string(MAX_LIGHTS) + " fit");

    for (size_t i = 0; i < spheres.size(); ++i) {
        const Sphere &sp = spheres[i];
        const std::string name = "sphere " + std::to_string(i);
        require(finite(sp.center) && finite(sp.radius) && finite(sp.material),
            name + " has a NaN or infinite value");
        // A zero radius makes the hit normal 0/0.
        require(sp.radius > 0.0f, name + " radius must be positive, got " +
            std::to_string(sp.radius));
    }
    for (size_t i = 0; i < lights.size(); ++i) {
        const Light &l = lights[i];
        const std::string name = "light " + std::to_string(i);
        require(finite(l.position) && finite(l.intensity) && finite(l.radius),
            name + " has a NaN or infinite value");
        require(l.radius >= 0.0f, name + " radius must be 0 or more, got " +
            std::to_string(l.radius));
    }
    require(finite(floor.y) && finite(floor.half_extent_x) && finite(floor.z_min) &&
        finite(floor.z_max) && finite(floor.checker_scale) && finite(floor.material),
        "floor has a NaN or infinite value");
}

void check_render_args(const Camera &c, const int frame, const int samples_per_frame) {
    require(frame >= 0, "frame must be 0 or more, got " + std::to_string(frame));
    require(samples_per_frame >= 1, "samples_per_frame must be 1 or more, got " +
        std::to_string(samples_per_frame));
    require(finite(c.position) && finite(c.right) && finite(c.up) && finite(c.forward) &&
        finite(c.fov_scale) && finite(c.aspect),
        "camera has a NaN or infinite value (is its target on the view axis, or up parallel to it?)");
}

}  // namespace

namespace gpu {

// Device memory and the scene copy for one Renderer.
struct Renderer::Impl {
    float4 *d_accum = nullptr;
    uint32_t *d_pixels = nullptr;
    uint32_t *h_pixels = nullptr;
    int width = 0;
    int height = 0;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    float kernel_ms = 0.0f;
    SceneParams params{};

    // Destructors must not throw, so CUDA errors here are ignored. After a sticky
    // fault these calls fail anyway, and the process is about to report it.
    ~Impl() {
        cudaFree(d_accum);
        cudaFree(d_pixels);
        cudaFreeHost(h_pixels);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
    }
};

Renderer::Renderer(const int width, const int height, const std::vector<Sphere> &spheres,
    const std::vector<Light> &lights, const Plane &floor) {
    check_frame_size(width, height);
    check_scene(spheres, lights, floor);

    // Built in a local first: if any CUDA call throws, its destructor frees
    // whatever was already allocated.
    auto r = std::make_unique<Impl>();
    r->width = width;
    r->height = height;

    std::copy(spheres.begin(), spheres.end(), r->params.spheres);
    std::copy(lights.begin(), lights.end(), r->params.lights);
    r->params.floor = floor;
    r->params.sphere_count = static_cast<int>(spheres.size());
    r->params.light_count = static_cast<int>(lights.size());

    const size_t pixel_count = static_cast<size_t>(width) * height;
    CUDA_CHECK(cudaMalloc(&r->d_accum, pixel_count * sizeof(float4)));
    // cudaMalloc does not zero, and a first render with frame > 0 adds to this.
    CUDA_CHECK(cudaMemset(r->d_accum, 0, pixel_count * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&r->d_pixels, pixel_count * sizeof(uint32_t)));
    // Pinned host memory: the DMA engine reads it directly, roughly doubling
    // device-to-host transfer speed.
    CUDA_CHECK(cudaHostAlloc(&r->h_pixels, pixel_count * sizeof(uint32_t), cudaHostAllocDefault));

    CUDA_CHECK(cudaEventCreate(&r->ev_start));
    CUDA_CHECK(cudaEventCreate(&r->ev_stop));

    impl_ = std::move(r);
}

// Defined here, where Impl is complete, because unique_ptr must see it to delete it.
Renderer::~Renderer() = default;
Renderer::Renderer(Renderer &&) noexcept = default;
Renderer &Renderer::operator=(Renderer &&) noexcept = default;

Renderer::Impl &Renderer::state() {
    if (!impl_) throw std::logic_error("gpu::Renderer used after being moved from");
    return *impl_;
}

const Renderer::Impl &Renderer::state() const {
    if (!impl_) throw std::logic_error("gpu::Renderer used after being moved from");
    return *impl_;
}

const uint32_t *Renderer::render(const Camera &camera, const int frame, const int samples_per_frame) {
    Impl &r = state();
    check_render_args(camera, frame, samples_per_frame);

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((r.width + BLOCK_X - 1) / BLOCK_X, (r.height + BLOCK_Y - 1) / BLOCK_Y);

    CUDA_CHECK(cudaEventRecord(r.ev_start));
    render_kernel<<<grid, block>>>(r.params, r.d_accum, r.d_pixels, r.width, r.height, camera,
        frame, samples_per_frame);
    // Catches a launch that never started. A fault while running surfaces at the copy.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(r.ev_stop));

    CUDA_CHECK(cudaMemcpy(r.h_pixels, r.d_pixels,
        static_cast<size_t>(r.width) * r.height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&r.kernel_ms, r.ev_start, r.ev_stop));

    return r.h_pixels;
}

float Renderer::last_kernel_ms() const { return state().kernel_ms; }

int Renderer::width() const { return state().width; }

int Renderer::height() const { return state().height; }

}  // namespace gpu
