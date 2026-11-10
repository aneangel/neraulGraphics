// Tests for neraulgraphics::Renderer. Needs a CUDA GPU. Built with -ffast-math on
// purpose, like the viewer, to show the input checks survive it. Exit status is
// the failure count, so ctest reports a failure.

#include <cstdio>
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

constexpr int W = 320, H = 180;

struct Scene {
    std::vector<Sphere> spheres;
    std::vector<Light> lights;
    Plane floor{};
};

Scene make_scene() {
    const Material ivory = make_material({0.4f, 0.4f, 0.3f}, 0.6f, 0.3f, 0.1f, 50.0f);
    const Material mirror = make_material({1.0f, 1.0f, 1.0f}, 0.0f, 10.0f, 0.8f, 1425.0f);
    Scene s;
    s.spheres = {{{-3, 0, -16}, 2, ivory}, {{-1, -1.5f, -12}, 2, mirror}, {{2, 1, -20}, 3, ivory}};
    s.lights = {{{-20, 20, 20}, 1.5f, 0.0f}, {{30, 50, 30}, 1.7f, 2.0f}};
    s.floor.y = -4;
    s.floor.half_extent_x = 10;
    s.floor.z_min = -30;
    s.floor.z_max = -10;
    s.floor.checker_scale = 0.5f;
    s.floor.material = make_material({1, 1, 1}, 1, 0, 0, 1);
    return s;
}

Camera camera() { return make_camera({0, 0, 0}, {0, 0, -16}, {0, 1, 0}, 1.5708f, W, H); }

// Frames 0..frames-1 accumulated, returning the final image.
std::vector<uint32_t> accumulate(Renderer &r, const int frames) {
    const uint32_t *p = nullptr;
    for (int f = 0; f < frames; ++f) p = r.render(camera(), f, 1);
    return {p, p + static_cast<size_t>(W) * H};
}

}  // namespace

int main() {
    const Scene s = make_scene();
    // Built at runtime so -ffast-math cannot fold them away as impossible.
    volatile float zero = 0.0f;
    const float nan = zero / zero, inf = 1.0f / zero;

    std::puts("version");
    check(std::string(NERAULGRAPHICS_VERSION) ==
            std::to_string(NERAULGRAPHICS_VERSION_MAJOR) + "." +
            std::to_string(NERAULGRAPHICS_VERSION_MINOR) + "." +
            std::to_string(NERAULGRAPHICS_VERSION_PATCH),
        std::string("version.h agrees with itself: ") + NERAULGRAPHICS_VERSION);

    std::puts("rendering");
    {
        Renderer a(W, H, s.spheres, s.lights, s.floor);
        Renderer b(W, H, s.spheres, s.lights, s.floor);
        check(accumulate(a, 4) == accumulate(b, 4), "same scene and frames give identical images");

        Renderer no_floor(W, H, s.spheres, s.lights);
        check(accumulate(no_floor, 4) != accumulate(a, 4), "the default floor is no floor");
    }

    std::puts("independent renderers");
    {
        const std::vector<Sphere> other = {{{0, 0, -10}, 3, s.spheres[0].material}};
        Renderer solo_a(W, H, s.spheres, s.lights, s.floor);
        Renderer solo_b(W, H, other, s.lights, s.floor);
        const auto ref_a = accumulate(solo_a, 4), ref_b = accumulate(solo_b, 4);

        Renderer a(W, H, s.spheres, s.lights, s.floor);
        Renderer b(W, H, other, s.lights, s.floor);
        std::vector<uint32_t> out_a, out_b;
        for (int f = 0; f < 4; ++f) {
            const uint32_t *pa = a.render(camera(), f, 1);
            out_a.assign(pa, pa + static_cast<size_t>(W) * H);
            const uint32_t *pb = b.render(camera(), f, 1);
            out_b.assign(pb, pb + static_cast<size_t>(W) * H);
        }
        check(ref_a != ref_b, "the two scenes render differently");
        check(out_a == ref_a && out_b == ref_b, "interleaved renders match each renderer alone");

        Renderer moved = std::move(b);
        check(accumulate(moved, 4) == ref_b, "a moved-to renderer keeps its scene");
        expect_throw<std::logic_error>("render on moved-from", [&] { b.render(camera(), 0, 1); });
        expect_throw<std::logic_error>("width() on moved-from", [&] { (void)b.width(); });
    }

    std::puts("constructor rejects bad input");
    auto build = [&](const int w, const int h, const std::vector<Sphere> &sp,
                     const std::vector<Light> &l, const Plane &f) { Renderer r(w, h, sp, l, f); };
    expect_throw<std::invalid_argument>("zero width", [&] { build(0, H, s.spheres, s.lights, s.floor); });
    expect_throw<std::invalid_argument>("negative height", [&] { build(W, -1, s.spheres, s.lights, s.floor); });
    expect_throw<std::invalid_argument>("pixel count overflows int",
        [&] { build(70000, 70000, s.spheres, s.lights, s.floor); });
    expect_throw<std::invalid_argument>("too many spheres",
        [&] { build(W, H, std::vector<Sphere>(MAX_SPHERES + 1, s.spheres[0]), s.lights, s.floor); });
    expect_throw<std::invalid_argument>("too many lights",
        [&] { build(W, H, s.spheres, std::vector<Light>(MAX_LIGHTS + 1, s.lights[0]), s.floor); });
    {
        auto sp = s.spheres;
        sp[2].radius = 0.0f;
        expect_throw<std::invalid_argument>("zero sphere radius", [&] { build(W, H, sp, s.lights, s.floor); });
    }
    {
        auto sp = s.spheres;
        sp[1].center.y = nan;
        expect_throw<std::invalid_argument>("NaN sphere center", [&] { build(W, H, sp, s.lights, s.floor); });
    }
    {
        auto sp = s.spheres;
        sp[0].material.reflectivity = inf;
        expect_throw<std::invalid_argument>("infinite material value", [&] { build(W, H, sp, s.lights, s.floor); });
    }
    {
        auto l = s.lights;
        l[0].radius = -1.0f;
        expect_throw<std::invalid_argument>("negative light radius", [&] { build(W, H, s.spheres, l, s.floor); });
    }
    {
        auto f = s.floor;
        f.z_max = nan;
        expect_throw<std::invalid_argument>("NaN floor", [&] { build(W, H, s.spheres, s.lights, f); });
    }
    {
        bool ok = true;
        try {
            build(W, H, {}, {}, Plane{});
        } catch (...) {
            ok = false;
        }
        check(ok, "an empty scene is accepted");
    }

    std::puts("render rejects bad input and is unchanged afterwards");
    {
        Renderer r(W, H, s.spheres, s.lights, s.floor);
        for (int f = 0; f < 3; ++f) r.render(camera(), f, 1);
        const float ms = r.last_kernel_ms();

        expect_throw<std::invalid_argument>("samples_per_frame 0", [&] { r.render(camera(), 3, 0); });
        expect_throw<std::invalid_argument>("negative frame", [&] { r.render(camera(), -1, 1); });
        // Looking straight up with world-up (0,1,0): the right vector is 0/0.
        const Camera degenerate = make_camera({0, 0, 0}, {0, 5, 0}, {0, 1, 0}, 1.5f, W, H);
        expect_throw<std::invalid_argument>("degenerate camera", [&] { r.render(degenerate, 3, 1); });
        check(r.last_kernel_ms() == ms, "last_kernel_ms untouched by rejected calls");

        Renderer clean(W, H, s.spheres, s.lights, s.floor);
        const auto expect = accumulate(clean, 4);
        const uint32_t *p = r.render(camera(), 3, 1);
        check(std::vector<uint32_t>(p, p + static_cast<size_t>(W) * H) == expect,
            "accumulation continues exactly as if nothing was rejected");
    }

    std::puts("accumulation buffer starts zeroed");
    {
        // A first render at frame > 0 adds to whatever the buffer holds. Fill one
        // renderer's buffer with sums, free it so the next allocation is likely
        // to reuse that memory, then compare against a render on clean memory.
        // If the allocator hands back different memory, this passes trivially.
        std::vector<uint32_t> clean_image;
        {
            Renderer clean(W, H, s.spheres, s.lights, s.floor);
            const uint32_t *p = clean.render(camera(), 5, 1);
            clean_image.assign(p, p + static_cast<size_t>(W) * H);
        }
        {
            Renderer dirty(W, H, s.spheres, s.lights, s.floor);
            accumulate(dirty, 50);
        }
        Renderer reused(W, H, s.spheres, s.lights, s.floor);
        const uint32_t *p = reused.render(camera(), 5, 1);
        check(std::vector<uint32_t>(p, p + static_cast<size_t>(W) * H) == clean_image,
            "first render at frame 5 ignores what freed memory held");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
        failures == 1 ? "" : "s");
    return failures;
}
