// Tests for the path tracer and denoiser. Light transport has closed-form answers in simple scenes, so these
// check numbers, not looks: a white furnace, a surface under a point light, a surface under a sun in a
// map, noise falling with samples, the denoiser beating plain accumulation. Every test runs on the CPU and on a
// GPU if there is one, since the two must agree.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <neraulgraphics/renderer.h>

using namespace neraulgraphics;

namespace {

int failures = 0;

void check(const bool ok, const std::string &what) {
    std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

template <class E>
void expect_throw(const std::string &what, const std::function<void()> &f) {
    try {
        f();
        check(false, what + " (nothing thrown)");
    } catch (const E &e) {
        check(true, what + "  ->  " + e.what());
    } catch (const std::exception &e) {
        check(false, what + " (wrong exception: " + e.what() + ")");
    }
}

constexpr int W = 64, H = 36;
constexpr float FOV = 1.5708f;

Camera look_forward() { return make_camera({0, 0, 0}, {0, 0, -5}, {0, 1, 0}, FOV, W, H); }

// One light level as the pixel byte the renderer writes with no tone mapping: gamma 2.2.
float linear_of(const uint32_t px, const int shift) { return std::pow(static_cast<float>((px >> shift) & 0xFF) / 255.0f, 2.2f); }

PathTracing plain(const int bounces) {
    PathTracing p;
    p.bounces = bounces;
    p.tonemap = false;   // so a pixel is the light itself, and can be read back as such
    p.denoise = false;   // so the picture is the plain average of the paths
    p.max_radiance = 0;  // no clamp: nothing may bias the average
    return p;
}

// The average over `frames` frames of `spp` paths each, as linear light per pixel (RGB).
std::vector<float> accumulate(Renderer &r, const Camera &cam, const int frames, const int spp) {
    const uint32_t *px = nullptr;
    for (int f = 0; f < frames; ++f) px = r.render(cam, f, spp);
    std::vector<float> out(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i)
        for (int k = 0; k < 3; ++k) out[3 * i + k] = linear_of(px[i], 16 - 8 * k);
    return out;
}

double mean_over(const std::vector<float> &img, const int x0, const int y0, const int x1, const int y1, const int channel = 1) {
    double sum = 0;
    int n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x, ++n) sum += img[3 * (static_cast<size_t>(y) * W + x) + channel];
    return sum / n;
}

double rmse(const std::vector<float> &a, const std::vector<float> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s / static_cast<double>(a.size()));
}

Environment uniform_sky(const float level) {
    Environment e;
    e.color_up = e.color_down = {level, level, level};
    return e;
}

// A map of the whole sky as black except for one bright patch, and the irradiance that patch gives a surface facing straight up.
struct SunMap {
    Environment env;
    double irradiance_up;  // what a surface facing up receives from the patch
    Vec3 direction;        // where the patch is
};
SunMap make_sun_map(const float radiance, const bool sun_only) {
    SunMap s;
    s.env.map_width = 128;
    s.env.map_height = 64;
    s.env.map.assign(128 * 64 * 3, sun_only ? 0.0f : 0.05f);
    s.irradiance_up = 0;
    const double pi = 3.14159265358979;
    double dx = 0, dy = 0, dz = 0, total = 0;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 128; ++x) {
            // The patch: a small disc about 45 degrees from straight up, towards -z.
            const double theta = pi * (y + 0.5) / 64, phi = 2 * pi * (x + 0.5) / 128;
            const double ux = std::sin(theta) * std::cos(phi), uy = std::cos(theta), uz = std::sin(theta) * std::sin(phi);
            const double c = ux * 0.0 + uy * std::cos(pi / 4) + uz * -std::sin(pi / 4);  // cosine of the angle from the sun's centre
            if (c > std::cos(0.08)) {
                for (int k = 0; k < 3; ++k) s.env.map[3 * (static_cast<size_t>(y) * 128 + x) + k] = radiance;
                const double solid_angle = (2 * pi / 128) * (pi / 64) * std::sin(theta);
                s.irradiance_up += radiance * solid_angle * uy;  // times the cosine against the surface normal (up)
                dx += ux * solid_angle; dy += uy * solid_angle; dz += uz * solid_angle; total += solid_angle;
            }
        }
    s.direction = {static_cast<float>(dx / total), static_cast<float>(dy / total), static_cast<float>(dz / total)};
    return s;
}

// A square of side 2 * half in the plane y = height, facing up (counter-clockwise seen from above).
Mesh horizontal_quad(const float half, const float height) {
    Mesh m;
    m.vertices = {{-half, height, -half}, {half, height, -half}, {half, height, half}, {-half, height, half}};
    m.indices = {0, 2, 1, 0, 3, 2};
    return m;
}

// A square facing the camera (towards +z) at z = -5.
Mesh wall(const float half) {
    Mesh m;
    m.vertices = {{-half, -half, -5}, {half, -half, -5}, {half, half, -5}, {-half, half, -5}};
    m.indices = {0, 1, 2, 0, 2, 3};
    return m;
}

std::vector<Device> devices() {
    std::vector<Device> d = {Device::Cpu};
    if (const char *only = std::getenv("NERAULGRAPHICS_DEVICE"); only != nullptr && std::strcmp(only, "cpu") == 0) return d;
    try { Renderer probe(8, 8, {}, {}, Plane{}, {}, {}, {}, Device::Cuda); d.push_back(Device::Cuda); } catch (const CudaError &) {}
    return d;
}

const char *name(const Device d) { return d == Device::Cuda ? "GPU" : "CPU"; }

const Material WHITE_DIFFUSE = make_material({1, 1, 1}, 1.0f, 0.0f, 0.0f, 50.0f);

}  // namespace

int main() {
    volatile float zero = 0.0f;
    const float nan = zero / zero;

    for (const Device dev : devices()) {
        std::printf("== %s ==\n", name(dev));
        const int bigger = dev == Device::Cuda ? 4 : 1;  // the GPU can afford more samples

        std::puts("white furnace: inside uniform light a perfectly reflective object is invisible");
        {
            const Camera cam = look_forward();
            struct Case { const char *what; Material m; double lo, hi; };
            Material metal_smooth = make_material({1, 1, 1}, 1, 0, 0, 50);
            metal_smooth.roughness = 0.05f;
            metal_smooth.metallic = 1.0f;
            Material metal_rough = metal_smooth;
            metal_rough.roughness = 0.3f;
            Material dielectric = make_material({1, 1, 1}, 1, 0, 0, 50);
            dielectric.roughness = 0.4f;
            dielectric.metallic = 0.0f;
            const Case cases[] = {
                {"a white diffuse sphere", WHITE_DIFFUSE, 0.985, 1.015},
                {"a smooth white metal sphere", metal_smooth, 0.97, 1.01},
                {"a rougher white metal sphere (loses a little to multiple scattering)", metal_rough, 0.92, 1.01},
                {"a white plastic sphere (diffuse under a clear coat)", dielectric, 0.95, 1.01},
            };
            for (const Case &c : cases) {
                Renderer r(W, H, {{{0, 0, -6}, 2.0f, c.m}}, {}, Plane{}, {}, {}, {}, dev);
                r.set_environment(uniform_sky(1.0f));
                r.set_render_mode(RenderMode::PathTraced, plain(16));
                const std::vector<float> img = accumulate(r, cam, 40 * bigger, 4);
                // The middle of the picture, all sphere.
                const double m = mean_over(img, W / 2 - 8, H / 2 - 4, W / 2 + 8, H / 2 + 4);
                std::printf("  %s: radiance %.4f (the sky is 1)\n", c.what, m);
                check(m > c.lo && m < c.hi, std::string(c.what) + " is as bright as the sky it sits in");
            }
        }

        std::puts("direct light has the closed-form answer");
        {
            const Camera cam = look_forward();
            // A wall of albedo 0.5, facing the camera, lit by a light at the camera of intensity 2: radiance is albedo * intensity * cos.
            Material m = make_material({0.5f, 0.5f, 0.5f}, 1.0f, 0.0f, 0.0f, 50.0f);
            for (const float radius : {0.0f, 0.8f}) {
                Renderer r(W, H, {}, {{{0, 0, 0}, 2.0f, radius}}, Plane{}, {wall(8.0f)}, {{0, identity_transform(), m}}, {}, dev);
                Environment dark = uniform_sky(0.0f);
                r.set_environment(dark);
                r.set_render_mode(RenderMode::PathTraced, plain(0));  // direct light only
                const std::vector<float> img = accumulate(r, cam, 12 * bigger, 4);
                const double centre = mean_over(img, W / 2 - 1, H / 2 - 1, W / 2 + 1, H / 2 + 1);
                // A point on the wall 2 units to the side: the light is at the origin, the wall 5 away, so cos = 5 / sqrt(25 + 4).
                const double ppu = (W / 2.0) / (5.0 * (static_cast<double>(W) / H));
                const int sx = W / 2 + static_cast<int>(std::lround(2.0 * ppu));
                const double side = mean_over(img, sx - 1, H / 2 - 1, sx + 1, H / 2 + 1);
                const double expected_side = 1.0 * 5.0 / std::sqrt(29.0);
                std::printf("  light radius %.1f: centre %.4f (expect 1.0000), 2 to the side %.4f (expect %.4f)\n", radius, centre, side, expected_side);
                check(std::fabs(centre - 1.0) < 0.02 && std::fabs(side - expected_side) < 0.03,
                    radius == 0.0f ? "a point light gives albedo * intensity * cos" : "a soft (spherical) light gives the same irradiance, not more or less");
            }
        }

        std::puts("a sun in a sky map: importance sampling and multiple importance sampling");
        {
            const SunMap sun = make_sun_map(136.0f, true);
            // A horizontal floor of albedo 0.8 seen from above.
            const Material m = make_material({0.8f, 0.8f, 0.8f}, 1.0f, 0.0f, 0.0f, 50.0f);
            Renderer r(W, H, {}, {}, Plane{}, {horizontal_quad(20.0f, -3.0f)}, {{0, identity_transform(), m}}, {}, dev);
            r.set_environment(sun.env);
            r.set_render_mode(RenderMode::PathTraced, plain(0));
            const Camera down = make_camera({0, 2, 0}, {0, -3, -0.01f}, {0, 0, -1}, FOV, W, H);
            const std::vector<float> img = accumulate(r, down, 6 * bigger, 16);
            const double expected = 0.8 * sun.irradiance_up / 3.14159265358979;
            const double got = mean_over(img, W / 2 - 6, H / 2 - 3, W / 2 + 6, H / 2 + 3);
            std::printf("  floor radiance %.4f, expected albedo * E / pi = %.4f (the sun alone gives E = %.2f)\n", got, expected, sun.irradiance_up);
            check(std::fabs(got / expected - 1.0) < 0.06, "a small bright sun lights a floor by the irradiance it should, with few samples");

            // The same sky as a uniform level: the importance-sampled path must agree with the plain one.
            Environment flat;
            flat.map_width = 64;
            flat.map_height = 32;
            flat.map.assign(64 * 32 * 3, 1.0f);
            Renderer furnace(W, H, {{{0, 0, -6}, 2.0f, WHITE_DIFFUSE}}, {}, Plane{}, {}, {}, {}, dev);
            furnace.set_environment(flat);
            furnace.set_render_mode(RenderMode::PathTraced, plain(16));
            const std::vector<float> fimg = accumulate(furnace, look_forward(), 40 * bigger, 4);
            const double fm = mean_over(fimg, W / 2 - 8, H / 2 - 4, W / 2 + 8, H / 2 + 4);
            std::printf("  white furnace under a uniform MAP (importance sampled + MIS): %.4f\n", fm);
            check(std::fabs(fm - 1.0) < 0.02, "combining light sampling with material sampling neither adds nor loses light");
        }

        std::puts("emission, and the classic renderer's lights mean the same thing in both");
        {
            Material glow = make_material({0, 0, 0}, 0.0f, 0.0f, 0.0f, 50.0f);
            glow.emission = {0.5f, 0.4f, 0.2f};
            Renderer r(W, H, {{{0, 0, -6}, 2.0f, glow}}, {}, Plane{}, {}, {}, {}, dev);
            r.set_environment(uniform_sky(0.0f));
            r.set_render_mode(RenderMode::PathTraced, plain(2));
            const std::vector<float> img = accumulate(r, look_forward(), 8, 4);
            std::printf("  an emissive sphere: %.3f %.3f %.3f (expect 0.5 0.4 0.2)\n", mean_over(img, W / 2 - 4, H / 2 - 2, W / 2 + 4, H / 2 + 2, 0),
                mean_over(img, W / 2 - 4, H / 2 - 2, W / 2 + 4, H / 2 + 2, 1), mean_over(img, W / 2 - 4, H / 2 - 2, W / 2 + 4, H / 2 + 2, 2));
            check(std::fabs(mean_over(img, W / 2 - 4, H / 2 - 2, W / 2 + 4, H / 2 + 2, 0) - 0.5) < 0.02 &&
                      std::fabs(mean_over(img, W / 2 - 4, H / 2 - 2, W / 2 + 4, H / 2 + 2, 2) - 0.2) < 0.02,
                "a surface that emits shows exactly the light it emits");

            // A diffuse sphere under one light, no sky: the classic renderer and the path tracer must agree.
            const Material diffuse = make_material({0.7f, 0.5f, 0.3f}, 0.8f, 0.0f, 0.0f, 50.0f);
            const std::vector<Light> light = {{{-6, 8, 4}, 1.0f, 0.0f}};
            Environment black = uniform_sky(0.0f);
            Renderer classic(W, H, {{{0, 0, -6}, 2.0f, diffuse}}, light, Plane{}, {}, {}, {}, dev);
            classic.set_environment(black);
            Renderer traced(W, H, {{{0, 0, -6}, 2.0f, diffuse}}, light, Plane{}, {}, {}, {}, dev);
            traced.set_environment(black);
            traced.set_render_mode(RenderMode::PathTraced, plain(0));
            const std::vector<float> a = accumulate(classic, look_forward(), 8, 4), b = accumulate(traced, look_forward(), 8, 4);
            double worst = 0;
            for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
            std::printf("  a lit diffuse sphere, classic vs path traced: largest difference %.4f\n", worst);
            check(rmse(a, b) < 0.015, "a light of intensity I lights a diffuse surface the same in both renderers");
        }

        std::puts("noise falls as samples grow, and the denoiser beats plain accumulation");
        {
            // A sphere and a floor under a sky and a light: a scene with soft shadows, bounced light and edges.
            Plane floor{};
            floor.y = -2;
            floor.half_extent_x = 30;
            floor.z_min = -60;
            floor.z_max = 5;
            floor.checker_scale = 1;
            floor.material = make_material({1, 1, 1}, 1, 0, 0, 1);
            const std::vector<Sphere> spheres = {{{0, -0.5f, -6}, 1.5f, make_material({0.8f, 0.3f, 0.2f}, 0.9f, 0.3f, 0.0f, 80.0f)},
                {{2.5f, -1.0f, -7}, 1.0f, make_material({0.2f, 0.6f, 0.9f}, 0.9f, 0.3f, 0.0f, 80.0f)}};
            const std::vector<Light> lights = {{{-6, 9, 3}, 1.2f, 1.5f}};
            auto make = [&] {
                Renderer r(W, H, spheres, lights, floor, {}, {}, {}, dev);
                r.set_render_mode(RenderMode::PathTraced, plain(4));
                return r;
            };
            const Camera cam = make_camera({0, 0.5f, 2}, {0, -0.5f, -6}, {0, 1, 0}, FOV, W, H);
            Renderer ref = make();
            const std::vector<float> truth = accumulate(ref, cam, 100 * bigger, 8);

            // Noise falls with the number of paths: four times the paths, half the error (here, less than 0.7).
            double err[3];
            int idx = 0;
            for (const int spp : {1, 4, 16}) {
                Renderer r = make();
                r.set_render_mode(RenderMode::PathTraced, plain(4));
                err[idx++] = rmse(accumulate(r, cam, 1, spp), truth);
            }
            std::printf("  error against a long render at 1, 4, 16 paths per pixel: %.4f %.4f %.4f\n", err[0], err[1], err[2]);
            check(err[1] < 0.7 * err[0] && err[2] < 0.7 * err[1], "four times the samples gives clearly less noise");

            // Eight frames of one path per pixel: plain accumulation against the denoiser.
            Renderer raw = make();
            const double raw_err = rmse(accumulate(raw, cam, 8, 1), truth);
            Renderer clean = make();
            PathTracing p = plain(4);
            p.denoise = true;
            clean.set_render_mode(RenderMode::PathTraced, p);
            const double denoised_err = rmse(accumulate(clean, cam, 8, 1), truth);
            std::printf("  8 frames of 1 path per pixel: error %.4f accumulated, %.4f denoised\n", raw_err, denoised_err);
            // With that much history there is little left for the filter to do, so it must at least do no harm.
            check(denoised_err < 1.1 * raw_err, "with history to lean on, the denoiser is no worse than the same samples simply averaged");
            check(denoised_err < 0.1, "and it is close in absolute terms");

            // Where it earns its keep: one path per pixel and no history at all.
            Renderer one_raw = make();
            Renderer one_clean = make();
            one_clean.set_render_mode(RenderMode::PathTraced, p);
            const double one_raw_err = rmse(accumulate(one_raw, cam, 1, 1), truth);
            const double one_clean_err = rmse(accumulate(one_clean, cam, 1, 1), truth);
            std::printf("  a single path per pixel: error %.4f raw, %.4f denoised\n", one_raw_err, one_clean_err);
            check(one_clean_err < 0.8 * one_raw_err, "on a single noisy frame the denoiser is clearly closer to the truth");

            // Following a camera that moves: history is carried across, and it must not smear.
            Renderer follow = make();
            follow.set_render_mode(RenderMode::PathTraced, p);
            Renderer fresh = make();
            fresh.set_render_mode(RenderMode::PathTraced, p);
            std::vector<float> moved_truth;
            const uint32_t *a = nullptr, *b = nullptr;
            Camera last = cam;
            for (int f = 0; f < 10; ++f) {
                last = make_camera({0.04f * static_cast<float>(f), 0.5f, 2}, {0, -0.5f, -6}, {0, 1, 0}, FOV, W, H);
                a = follow.render(last, f, 1);   // history carried along as the camera slides
                b = fresh.render(last, 0, 1);    // no history: a one-sample picture each time
            }
            std::vector<float> fa(static_cast<size_t>(W) * H * 3), fb(fa.size());
            for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i)
                for (int k = 0; k < 3; ++k) { fa[3 * i + k] = linear_of(a[i], 16 - 8 * k); fb[3 * i + k] = linear_of(b[i], 16 - 8 * k); }
            Renderer ref2 = make();
            const std::vector<float> truth2 = accumulate(ref2, last, 100 * bigger, 8);
            const double follow_err = rmse(fa, truth2), fresh_err = rmse(fb, truth2);
            std::printf("  after the camera has moved 10 times: error %.4f with history carried over, %.4f without\n", follow_err, fresh_err);
            check(follow_err < 0.85 * fresh_err, "history follows a moving camera and improves the picture, with no smearing");
        }

        std::puts("the new settings are checked");
        {
            Renderer r(W, H, {{{0, 0, -6}, 2.0f, WHITE_DIFFUSE}}, {}, Plane{}, {}, {}, {}, dev);
            auto with = [&](const std::function<void(PathTracing &)> &edit) { PathTracing p; edit(p); r.set_render_mode(RenderMode::PathTraced, p); };
            expect_throw<std::invalid_argument>("negative bounces", [&] { with([](PathTracing &p) { p.bounces = -1; }); });
            expect_throw<std::invalid_argument>("33 bounces", [&] { with([](PathTracing &p) { p.bounces = 33; }); });
            expect_throw<std::invalid_argument>("zero exposure", [&] { with([](PathTracing &p) { p.exposure = 0; }); });
            expect_throw<std::invalid_argument>("NaN exposure", [&] { with([&](PathTracing &p) { p.exposure = nan; }); });
            expect_throw<std::invalid_argument>("negative max_radiance", [&] { with([](PathTracing &p) { p.max_radiance = -1; }); });
            expect_throw<std::invalid_argument>("9 denoise iterations", [&] { with([](PathTracing &p) { p.denoise_iterations = 9; }); });
            expect_throw<std::invalid_argument>("temporal_blend 0", [&] { with([](PathTracing &p) { p.temporal_blend = 0; }); });
            expect_throw<std::invalid_argument>("temporal_blend 2", [&] { with([](PathTracing &p) { p.temporal_blend = 2; }); });
            check(r.render_mode() == RenderMode::Classic, "a rejected setting leaves the mode as it was");

            Environment bad;
            bad.map_width = 4;
            bad.map_height = 2;
            bad.map.assign(10, 1.0f);
            expect_throw<std::invalid_argument>("an environment map with too few values", [&] { r.set_environment(bad); });
            bad.map.assign(24, 1.0f);
            bad.map[5] = -1.0f;
            expect_throw<std::invalid_argument>("a negative value in an environment map", [&] { r.set_environment(bad); });
            bad.map[5] = nan;
            expect_throw<std::invalid_argument>("a NaN in an environment map", [&] { r.set_environment(bad); });
            Environment up_zero;
            up_zero.up = {0, 0, 0};
            expect_throw<std::invalid_argument>("a zero up axis", [&] { r.set_environment(up_zero); });
            Environment negative;
            negative.intensity = -1;
            expect_throw<std::invalid_argument>("a negative intensity", [&] { r.set_environment(negative); });

            Material rough = WHITE_DIFFUSE;
            rough.roughness = 2.0f;
            expect_throw<std::invalid_argument>("roughness above 1", [&] { Renderer x(W, H, {{{0, 0, -6}, 2.0f, rough}}, {}, Plane{}, {}, {}, {}, dev); });
            Material negative_glow = WHITE_DIFFUSE;
            negative_glow.emission = {-1, 0, 0};
            expect_throw<std::invalid_argument>("negative emission", [&] { Renderer x(W, H, {{{0, 0, -6}, 2.0f, negative_glow}}, {}, Plane{}, {}, {}, {}, dev); });
        }

        std::puts("going back to the classic renderer gives the classic picture");
        {
            Renderer r(W, H, {{{0, 0, -6}, 2.0f, WHITE_DIFFUSE}}, {{{-6, 8, 4}, 1.0f, 0.0f}}, Plane{}, {}, {}, {}, dev);
            const uint32_t *p = r.render(look_forward(), 0, 1);
            const std::vector<uint32_t> before(p, p + static_cast<size_t>(W) * H);
            r.set_render_mode(RenderMode::PathTraced);
            r.render(look_forward(), 0, 2);
            r.set_render_mode(RenderMode::Classic);
            p = r.render(look_forward(), 0, 1);
            check(std::vector<uint32_t>(p, p + static_cast<size_t>(W) * H) == before, "a round trip through path tracing changes nothing");
        }
    }

    // The two devices run the same code, so a converged picture is the same on both.
    if (devices().size() > 1) {
        std::puts("== the CPU and the GPU agree ==");
        Plane floor{};
        floor.y = -2;
        floor.half_extent_x = 30;
        floor.z_min = -60;
        floor.z_max = 5;
        floor.checker_scale = 1;
        floor.material = make_material({1, 1, 1}, 1, 0, 0, 1);
        const std::vector<Sphere> spheres = {{{0, -0.5f, -6}, 1.5f, make_material({0.8f, 0.3f, 0.2f}, 0.9f, 0.3f, 0.0f, 80.0f)}};
        const std::vector<Light> lights = {{{-6, 9, 3}, 1.2f, 1.5f}};
        const Camera cam = make_camera({0, 0.5f, 2}, {0, -0.5f, -6}, {0, 1, 0}, FOV, W, H);
        std::vector<float> img[2];
        int k = 0;
        for (const Device d : {Device::Cpu, Device::Cuda}) {
            Renderer r(W, H, spheres, lights, floor, {}, {}, {}, d);
            r.set_render_mode(RenderMode::PathTraced, plain(4));
            img[k++] = accumulate(r, cam, 150, 8);
        }
        const double e = rmse(img[0], img[1]);
        std::printf("  a path-traced scene averaged over 1200 paths per pixel: CPU and GPU differ by %.4f (rms)\n", e);
        check(e < 0.02, "the CPU and the GPU converge to the same picture");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures;
}
