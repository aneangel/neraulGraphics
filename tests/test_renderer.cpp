// Tests for neraulgraphics::Renderer. Needs a CUDA GPU. Built with -ffast-math on
// purpose, like the viewer, to show the input checks survive it. Exit status is
// the failure count, so ctest reports a failure.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <neraulgraphics/renderer.h>
#include "bvh.h"  // library-private: the builder is tested directly, on the CPU

using namespace neraulgraphics;

namespace {

int failures = 0;

// These tests are built with -ffast-math, under which the compiler may assume there are no infinities
// or NaNs, and fold std::isinf and std::isfinite to constants. Read the float's exponent bits instead.
bool is_inf(const float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7fffffffu) == 0x7f800000u;
}
bool is_finite_value(const float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

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
Camera camera_at_origin() { return make_camera({0, 0, 0}, {0, 0, -5}, {0, 1, 0}, 1.5708f, W, H); }

// Frames 0..frames-1 accumulated, returning the final image.
std::vector<uint32_t> accumulate(Renderer &r, const int frames) {
    const uint32_t *p = nullptr;
    for (int f = 0; f < frames; ++f) p = r.render(camera(), f, 1);
    return {p, p + static_cast<size_t>(W) * H};
}


// ------------------------------------------------------------------ meshes --

using Image = std::vector<uint32_t>;

// How different two images are: mean absolute error per 8-bit channel, and the
// fraction of pixels where any channel is off by more than 24 (visible, not rounding).
struct Diff {
    double mean;
    double bad_fraction;
};

Diff compare(const Image &a, const Image &b, const int x0 = 0, const int y0 = 0, const int x1 = W, const int y1 = H) {
    double total = 0.0;
    size_t bad = 0, count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const int x = static_cast<int>(i) % W, y = static_cast<int>(i) / W;
        if (x < x0 || x >= x1 || y < y0 || y >= y1) continue;
        ++count;
        int worst = 0;
        for (int shift = 0; shift <= 16; shift += 8) {
            const int d = std::abs(static_cast<int>((a[i] >> shift) & 0xFF) - static_cast<int>((b[i] >> shift) & 0xFF));
            total += d;
            worst = std::max(worst, d);
        }
        if (worst > 24) ++bad;
    }
    return {total / (3.0 * static_cast<double>(count)), static_cast<double>(bad) / static_cast<double>(count)};
}

Image frame(Renderer &r, const int index = 0) {
    const uint32_t *p = r.render(camera(), index, 1);
    return {p, p + static_cast<size_t>(W) * H};
}

Vec3 apply(const Transform &t, const Vec3 &v) {
    return {t.m[0] * v.x + t.m[1] * v.y + t.m[2] * v.z + t.m[3], t.m[4] * v.x + t.m[5] * v.y + t.m[6] * v.z + t.m[7],
        t.m[8] * v.x + t.m[9] * v.y + t.m[10] * v.z + t.m[11]};
}

// The mesh with a transform applied to its vertices: what an instance should look like.
Mesh bake(const Mesh &mesh, const Transform &t) {
    Mesh out = mesh;
    for (Vec3 &v : out.vertices) v = apply(t, v);
    return out;
}

Mesh merge(const Mesh &a, const Mesh &b) {
    Mesh out = a;
    const uint32_t base = static_cast<uint32_t>(a.vertices.size());
    out.vertices.insert(out.vertices.end(), b.vertices.begin(), b.vertices.end());
    for (const uint32_t i : b.indices) out.indices.push_back(base + i);
    return out;
}

// Small deterministic generator, so a failure reproduces.
struct Lcg {
    uint32_t state = 12345;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1 << 24);
    }
};

bvh::Aabb random_box(Lcg &rng) {
    const float c[3] = {rng.next() * 100, rng.next() * 100, rng.next() * 100};
    const float h[3] = {rng.next() * 3, rng.next() * 3, rng.next() * 3};
    return {{c[0] - h[0], c[1] - h[1], c[2] - h[2]}, {c[0] + h[0], c[1] + h[1], c[2] + h[2]}};
}

// Checks what traversal depends on: every primitive in exactly one leaf, every
// child box inside its parent's, leaves no bigger than asked, depth within the limit.
bool bvh_is_valid(const bvh::Bvh &tree, const std::vector<bvh::Aabb> &bounds, const int max_leaf, std::string &why) {
    std::vector<int> seen(bounds.size(), 0);
    std::vector<std::pair<int, int>> stack = {{0, 0}};  // node, depth
    while (!stack.empty()) {
        const auto [index, depth] = stack.back();
        stack.pop_back();
        if (depth > bvh::MAX_DEPTH) { why = "deeper than MAX_DEPTH"; return false; }
        const bvh::Node &n = tree.nodes[index];
        if (n.count > 0) {
            if (n.count > max_leaf) { why = "leaf with " + std::to_string(n.count) + " primitives"; return false; }
            for (int i = n.a; i < n.a + n.count; ++i) {
                ++seen[tree.order[i]];
                const bvh::Aabb &b = bounds[tree.order[i]];
                for (int k = 0; k < 3; ++k)
                    if (b.lo[k] < n.lo[k] || b.hi[k] > n.hi[k]) { why = "primitive outside its leaf"; return false; }
            }
        } else {
            for (const int child : {n.a, n.a + 1}) {
                const bvh::Node &c = tree.nodes[child];
                for (int k = 0; k < 3; ++k)
                    if (c.lo[k] < n.lo[k] || c.hi[k] > n.hi[k]) { why = "child box outside its parent"; return false; }
                stack.push_back({child, depth + 1});
            }
        }
    }
    for (const int count : seen)
        if (count != 1) { why = "a primitive is in " + std::to_string(count) + " leaves"; return false; }
    return true;
}


// ---------------------------------------------------- normals and textures --

// A square facing the camera at z = -5, from -half to +half, with uvs 0..uv_max (origin
// at the image's top left, so the top-left corner of the quad is uv (0, 0)).
Mesh make_quad(const float half, const float uv_max) {
    Mesh m;
    m.vertices = {{-half, -half, -5}, {half, -half, -5}, {half, half, -5}, {-half, half, -5}};
    m.uvs = {{0, uv_max}, {uv_max, uv_max}, {uv_max, 0}, {0, 0}};
    m.indices = {0, 1, 2, 0, 2, 3};
    return m;
}

Texture solid_2x2(const std::array<std::array<uint8_t, 3>, 4> &colors) {  // row 0: [0] [1], row 1: [2] [3]
    Texture t;
    t.width = t.height = 2;
    for (const auto &c : colors) t.rgba.insert(t.rgba.end(), {c[0], c[1], c[2], 255});
    return t;
}

uint32_t pixel_at(const uint32_t *p, const int x, const int y) { return p[y * W + x]; }
// A quad at distance 5 seen by a 90 degree camera: how many pixels one world unit covers.
// Probes are placed in units, so they stay right if the test image size changes.
constexpr double PX_PER_UNIT = (W / 2.0) / (5.0 * (static_cast<double>(W) / H));
int px(const double units) { return static_cast<int>(std::lround(units * PX_PER_UNIT)); }
int channel(const uint32_t px, const int shift) { return static_cast<int>((px >> shift) & 0xFF); }
// Which of red, green, blue clearly dominates, or 'w' when none does.
char dominant(const uint32_t px) {
    const int r = channel(px, 16), g = channel(px, 8), b = channel(px, 0);
    if (r > g + 40 && r > b + 40) return 'r';
    if (g > r + 40 && g > b + 40) return 'g';
    if (b > r + 40 && b > g + 40) return 'b';
    return 'w';
}

// A lit scene for texture tests: one light by the camera, no floor, no spheres.
Renderer textured_scene(const Mesh &mesh, const Texture &tex, const float albedo = 0.5f) {
    const Material white = make_material({1, 1, 1}, albedo, 0.0f, 0.0f, 1.0f);
    return Renderer(W, H, {}, {{{0, 0, 2}, 1.0f, 0.0f}}, Plane{}, {mesh}, {{0, identity_transform(), white, 0}}, {tex});
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


    std::puts("BVH builder (CPU)");
    {
        Lcg rng;
        for (const int n : {1, 2, 3, 100, 5000}) {
            std::vector<bvh::Aabb> boxes;
            for (int i = 0; i < n; ++i) boxes.push_back(random_box(rng));
            std::string why;
            const bool ok = bvh_is_valid(bvh::build(boxes, 4), boxes, 4, why);
            check(ok, std::to_string(n) + " random boxes: valid tree" + (ok ? "" : " (" + why + ")"));
        }

        // No plane separates identical centroids. A naive builder makes one giant leaf.
        std::vector<bvh::Aabb> same(1000, bvh::Aabb{{1, 1, 1}, {2, 2, 2}});
        std::string why;
        const bool ok = bvh_is_valid(bvh::build(same, 4), same, 4, why);
        check(ok, "1000 identical boxes: still split into small leaves" + (ok ? "" : " (" + why + ")"));

        check(bvh::build({}, 4).nodes.empty(), "no primitives gives an empty tree");

        // Refit keeps the tree's shape and must still bound every primitive after they move.
        std::vector<bvh::Aabb> boxes;
        for (int i = 0; i < 3000; ++i) boxes.push_back(random_box(rng));
        bvh::Bvh tree = bvh::build(boxes, 1);
        const float fresh_cost = bvh::sah_cost(tree);
        for (auto &b : boxes) {
            const float dx = rng.next() * 40 - 20;
            for (int k = 0; k < 3; ++k) { b.lo[k] += dx; b.hi[k] += dx; }
        }
        bvh::refit(tree, boxes);
        std::string refit_why;
        const bool refit_ok = bvh_is_valid(tree, boxes, 1, refit_why);
        check(refit_ok, "refit after moving every box: still a valid tree" + (refit_ok ? "" : " (" + refit_why + ")"));
        check(bvh::sah_cost(tree) > fresh_cost * 1.05f, "refit after a big scatter measurably loosens the tree");
        check(bvh::sah_cost(bvh::build(boxes, 1)) < bvh::sah_cost(tree), "a rebuild beats the refitted tree after a big scatter");
    }

    std::puts("meshes match known answers");
    {
        // A scene with something extra added either as an analytic sphere or as a mesh.
        const Material blue = make_material({0.1f, 0.2f, 0.6f}, 0.8f, 0.4f, 0.2f, 80.0f);
        const Vec3 where = {4.0f, 0.5f, -12.0f};

        auto with_sphere = s.spheres;
        with_sphere.push_back({where, 2.0f, blue});
        Renderer analytic(W, H, with_sphere, s.lights, s.floor);
        const Image want = frame(analytic);

        // Unit icosphere, scaled by 2 and moved by the instance's transform.
        const std::vector<Mesh> ball = {make_icosphere(1.0f, 4)};
        Renderer meshed(W, H, s.spheres, s.lights, s.floor, ball, {{0, make_transform(where, {0, 0, 0, 1}, 2.0f), blue}});
        const Image got = frame(meshed);

        const Diff d = compare(got, want);
        std::printf("  icosphere vs analytic sphere: mean error %.3f, %.3f%% of pixels visibly off\n", d.mean,
            d.bad_fraction * 100.0);
        check(d.mean < 1.0 && d.bad_fraction < 0.005,
            "a scaled, moved icosphere matches the analytic sphere (hits, shadows, reflections)");

        Renderer without(W, H, s.spheres, s.lights, s.floor);
        check(compare(got, frame(without)).mean > 0.2, "the mesh is actually visible");
    }

    {
        // Shear plus non-uniform scale plus rotation, which only a correct
        // inverse-transpose normal survives. Baking it into the vertices is the reference.
        const Transform skew = {{1.3f, 0.4f, 0.0f, 3.0f,  0.2f, 0.7f, 0.5f, 0.0f,  -0.3f, 0.1f, 1.1f, -13.0f}};
        const Material mat = make_material({0.7f, 0.3f, 0.2f}, 0.9f, 0.3f, 0.1f, 60.0f);
        const Mesh box = make_box({1, 1, 1});

        Renderer baked(W, H, s.spheres, s.lights, s.floor, {bake(box, skew)}, {{0, identity_transform(), mat}});
        Renderer instanced(W, H, s.spheres, s.lights, s.floor, {box}, {{0, skew, mat}});
        const Diff d = compare(frame(instanced), frame(baked));
        std::printf("  sheared instance vs baked vertices: mean error %.4f, %.3f%% visibly off\n", d.mean,
            d.bad_fraction * 100.0);
        check(d.mean < 0.25 && d.bad_fraction < 0.001, "a sheared, scaled, rotated instance matches its baked geometry");

        // A mirroring transform flips winding. Triangles are two-sided, so nothing should change.
        const Transform mirrored = {{-1, 0, 0, 3.0f,  0, 1, 0, 0.5f,  0, 0, 1, -13.0f}};
        Renderer mirror_baked(W, H, s.spheres, s.lights, s.floor, {bake(box, mirrored)}, {{0, identity_transform(), mat}});
        Renderer mirror_inst(W, H, s.spheres, s.lights, s.floor, {box}, {{0, mirrored, mat}});
        check(compare(frame(mirror_inst), frame(mirror_baked)).mean < 0.25, "a mirroring transform renders correctly");
    }

    std::puts("two-level BVH over many instances");
    {
        // 400 randomly placed and rotated boxes, once as 400 instances (top level
        // does the work) and once merged into a single mesh (bottom level does it all).
        Lcg rng;
        const Mesh box = make_box({0.5f, 0.5f, 0.5f});
        const Material mat = make_material({0.6f, 0.5f, 0.2f}, 0.9f, 0.2f, 0.0f, 40.0f);
        std::vector<Instance> instances;
        Mesh merged;
        for (int i = 0; i < 400; ++i) {
            const Vec3 pos = {-9.0f + 18.0f * rng.next(), -3.0f + 6.0f * rng.next(), -28.0f + 14.0f * rng.next()};
            const Quat q = {rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f, rng.next() + 0.1f};
            const Transform t = make_transform(pos, q, 0.4f + rng.next());
            instances.push_back({0, t, mat});
            merged = merge(merged, bake(box, t));
        }
        Renderer top(W, H, {}, s.lights, s.floor, {box}, instances);
        Renderer flat(W, H, {}, s.lights, s.floor, {merged}, {{0, identity_transform(), mat}});
        const Image a = frame(top), b = frame(flat);
        const Diff d = compare(a, b);
        std::printf("  400 instances vs one merged mesh: mean error %.4f, %.3f%% visibly off, kernel %.2f ms vs %.2f ms\n",
            d.mean, d.bad_fraction * 100.0, top.last_kernel_ms(), flat.last_kernel_ms());
        check(d.mean < 0.3 && d.bad_fraction < 0.002, "instances through the top-level BVH match the merged mesh");
        check(top.instance_count() == 400, "instance_count reports what was given");

        Renderer none(W, H, {}, s.lights, s.floor);
        check(compare(a, frame(none)).mean > 1.0, "the instances are visible");
    }

    std::puts("moving instances");
    {
        const Mesh ball = make_icosphere(1.0f, 3);
        const Material mat = make_material({0.2f, 0.6f, 0.3f}, 0.9f, 0.3f, 0.0f, 50.0f);
        auto place = [&](const float x) { return make_transform({x, 0.0f, -13.0f}); };

        Renderer r(W, H, s.spheres, s.lights, s.floor, {ball}, {{0, place(-3.0f), mat}, {0, place(3.0f), mat}});
        const Image before = frame(r);

        r.set_instance_transforms({place(-1.0f), place(5.0f)});
        const Image moved = frame(r);
        check(compare(before, moved).mean > 0.3, "set_instance_transforms changes the image");

        // A renderer built at the new positions is the reference: moving must equal rebuilding.
        Renderer fresh(W, H, s.spheres, s.lights, s.floor, {ball}, {{0, place(-1.0f), mat}, {0, place(5.0f), mat}});
        check(moved == frame(fresh), "moving instances renders exactly what building them there does");

        r.set_instance_transforms({place(-3.0f), place(3.0f)});
        check(frame(r) == before, "moving them back restores the original image exactly");

        expect_throw<std::invalid_argument>("wrong number of transforms",
            [&] { r.set_instance_transforms({place(0.0f)}); });
        expect_throw<std::invalid_argument>("singular transform", [&] {
            r.set_instance_transforms({place(0.0f), Transform{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}}});
        });
        Transform bad = place(0.0f);
        bad.m[3] = nan;
        expect_throw<std::invalid_argument>("NaN transform", [&] { r.set_instance_transforms({place(0.0f), bad}); });
        check(frame(r) == before, "rejected updates leave the scene exactly as it was");
    }

    std::puts("a long random walk (refit and rebuild paths)");
    {
        // 300 bodies drifting and sometimes teleporting. Every step, the updated renderer must
        // draw exactly what one built from scratch at those positions draws. Small steps are
        // refitted; teleports loosen the tree enough to force a rebuild. Either may be wrong
        // if a stale box hides an object.
        Lcg rng;
        const Mesh box = make_box({0.5f, 0.5f, 0.5f});
        const Material mat = make_material({0.5f, 0.4f, 0.7f}, 0.9f, 0.2f, 0.0f, 40.0f);
        std::vector<Vec3> pos(300);
        for (Vec3 &p : pos) p = {-9.0f + 18.0f * rng.next(), -3.0f + 6.0f * rng.next(), -28.0f + 14.0f * rng.next()};
        auto transforms = [&] {
            std::vector<Transform> t;
            for (const Vec3 &p : pos) t.push_back(make_transform(p, {0.1f, 0.2f, 0.0f, 1.0f}, 0.8f));
            return t;
        };
        auto instances = [&] {
            std::vector<Instance> v;
            for (const Transform &t : transforms()) v.push_back({0, t, mat});
            return v;
        };

        Renderer live(W, H, {}, s.lights, s.floor, {box}, instances());
        int mismatches = 0;
        double worst = 0.0;
        for (int step = 0; step < 14; ++step) {
            const bool teleport = step % 5 == 4;
            for (Vec3 &p : pos) {
                if (teleport) p = {-9.0f + 18.0f * rng.next(), -3.0f + 6.0f * rng.next(), -28.0f + 14.0f * rng.next()};
                else p = {p.x + 0.4f * (rng.next() - 0.5f), p.y + 0.4f * (rng.next() - 0.5f), p.z + 0.4f * (rng.next() - 0.5f)};
            }
            live.set_instance_transforms(transforms());
            Renderer fresh(W, H, {}, s.lights, s.floor, {box}, instances());
            const Diff d = compare(frame(live), frame(fresh));
            worst = std::max(worst, d.mean);
            if (d.mean > 0.05 || d.bad_fraction > 0.0005) ++mismatches;
        }
        std::printf("  14 steps (drifts and teleports), worst mean error vs rebuilt-from-scratch: %.4f\n", worst);
        check(mismatches == 0, "every step matches a renderer built from scratch at those positions");
    }


    std::puts("smooth shading");
    {
        // A faceted sphere shaded with its vertex normals should look rounder than shaded flat,
        // measured against the analytic sphere it approximates.
        const Material mat = make_material({0.7f, 0.3f, 0.2f}, 0.9f, 0.3f, 0.0f, 60.0f);
        const Vec3 where = {4.0f, 0.5f, -12.0f};
        auto with_sphere = s.spheres;
        with_sphere.push_back({where, 2.0f, mat});
        Renderer analytic(W, H, with_sphere, s.lights, s.floor);
        const Image want = frame(analytic);

        // 1280 triangles. A coarser sphere has a visibly polygonal outline that no normal can fix,
        // which hides the shading effect (80 triangles: error 8.5 flat vs 5.7 smooth; 1280: 1.7 vs 0.4).
        Mesh flat = make_icosphere(1.0f, 3);
        Mesh smooth = flat;
        for (const Vec3 &v : flat.vertices) smooth.normals.push_back(v);  // a unit sphere's normal is its position
        const Transform t = make_transform(where, {0, 0, 0, 1}, 2.0f);
        Renderer r_flat(W, H, s.spheres, s.lights, s.floor, {flat}, {{0, t, mat}});
        Renderer r_smooth(W, H, s.spheres, s.lights, s.floor, {smooth}, {{0, t, mat}});
        // Measured over the sphere (centred near pixel (190, 86), about 15 px across), where the
        // difference is, not diluted by the rest of the image.
        const double e_flat = compare(frame(r_flat), want, 170, 66, 210, 106).mean;
        const double e_smooth = compare(frame(r_smooth), want, 170, 66, 210, 106).mean;
        std::printf("  1280-triangle sphere vs analytic, around the sphere: flat error %.2f, smooth error %.2f\n", e_flat, e_smooth);
        check(e_smooth < 0.4 * e_flat, "vertex normals make a faceted sphere shade like a smooth one");
    }

    std::puts("textures");
    {
        const std::array<uint8_t, 3> red = {255, 0, 0}, green = {0, 255, 0}, blue = {0, 0, 255}, white = {255, 255, 255};
        Renderer r = textured_scene(make_quad(3.0f, 1.0f), solid_2x2({red, green, blue, white}));
        const uint32_t *p = r.render(camera_at_origin(), 0, 1);
        // The quad spans +-3, so each quadrant's centre is 1.5 units from the middle.
        const int cx = W / 2, cy = H / 2, d = px(1.5);
        const std::string got = {dominant(pixel_at(p, cx - d, cy - d)), dominant(pixel_at(p, cx + d, cy - d)),
            dominant(pixel_at(p, cx - d, cy + d)), dominant(pixel_at(p, cx + d, cy + d))};
        check(got == "rgbw", "a 2x2 texture lands the right way up: top-left red, top-right green, bottom-left blue, bottom-right white (got " + got + ")");
        check(dominant(pixel_at(p, 3, 3)) == 'b', "outside the quad is the sky");
    }
    {
        // 50% grey (128) is about 21.6% linear light. Rendered and gamma-encoded again it should
        // come out near half as bright as white; with no sRGB decode it would be ~73%.
        const std::array<uint8_t, 3> grey = {128, 128, 128}, white = {255, 255, 255};
        Renderer r = textured_scene(make_quad(3.0f, 1.0f), solid_2x2({grey, grey, white, white}), 0.4f);
        const uint32_t *p = r.render(camera_at_origin(), 0, 1);
        const int d = px(1.5);
        const double ratio = static_cast<double>(channel(pixel_at(p, W / 2 - d, H / 2 - d), 8)) /
            channel(pixel_at(p, W / 2 - d, H / 2 + d), 8);
        std::printf("  grey / white brightness ratio: %.3f (0.50 if sRGB-decoded, 0.73 if not)\n", ratio);
        check(ratio > 0.42 && ratio < 0.58, "textures are decoded from sRGB to linear light");
    }
    {
        // uvs 0..2 tile the 2x2 texture twice each way, so cells two apart match.
        const std::array<uint8_t, 3> red = {255, 0, 0}, green = {0, 255, 0}, blue = {0, 0, 255}, white = {255, 255, 255};
        Renderer r = textured_scene(make_quad(3.0f, 2.0f), solid_2x2({red, green, blue, white}));
        const uint32_t *p = r.render(camera_at_origin(), 0, 1);
        const int cx = W / 2, cy = H / 2, cell_px = px(1.5);  // a 4x4 grid of cells over the quad, 1.5 units each
        auto cell = [&](const int col, const int row) {
            return dominant(pixel_at(p, cx + (col - 2) * cell_px + cell_px / 2, cy + (row - 2) * cell_px + cell_px / 2));
        };
        check(cell(0, 0) == 'r' && cell(1, 0) == 'g' && cell(2, 0) == 'r' && cell(3, 0) == 'g' &&
                  cell(0, 1) == 'b' && cell(1, 1) == 'w' && cell(2, 1) == 'b' && cell(3, 3) == 'w',
            "uvs past 1 repeat the texture");
    }
    {
        // A texture on one instance must not leak onto another that has none.
        const std::array<uint8_t, 3> red = {255, 0, 0};
        const Material white = make_material({1, 1, 1}, 0.5f, 0.0f, 0.0f, 1.0f);
        Mesh quad = make_quad(1.0f, 1.0f);
        Renderer r(W, H, {}, {{{0, 0, 2}, 1.0f, 0.0f}}, Plane{}, {quad},
            {{0, make_transform({-2, 0, 0}), white, 0}, {0, make_transform({2, 0, 0}), white}}, {solid_2x2({red, red, red, red})});
        const uint32_t *p = r.render(camera_at_origin(), 0, 1);
        check(dominant(pixel_at(p, W / 2 - px(2.0), H / 2)) == 'r' && dominant(pixel_at(p, W / 2 + px(2.0), H / 2)) == 'w',
            "only the instance that names the texture is textured");
    }

    std::puts("normals and textures reject bad input");
    {
        const Material mat = s.spheres[0].material;
        const Texture ok = solid_2x2({{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {1, 1, 1}}});
        auto make = [&](const Mesh &m, const std::vector<Instance> &i, const std::vector<Texture> &t) {
            Renderer r(W, H, {}, {}, Plane{}, {m}, i, t);
        };
        Mesh quad = make_quad(1.0f, 1.0f);
        Mesh bad_normals = quad;
        bad_normals.normals = {{0, 0, 1}};
        expect_throw<std::invalid_argument>("normals not one per vertex", [&] { make(bad_normals, {{0, identity_transform(), mat}}, {}); });
        Mesh bad_uvs = quad;
        bad_uvs.uvs.pop_back();
        expect_throw<std::invalid_argument>("uvs not one per vertex", [&] { make(bad_uvs, {{0, identity_transform(), mat}}, {}); });
        Mesh nan_uv = quad;
        nan_uv.uvs[1].u = nan;
        expect_throw<std::invalid_argument>("NaN uv", [&] { make(nan_uv, {{0, identity_transform(), mat}}, {}); });
        Mesh nan_normal = quad;
        nan_normal.normals.assign(4, Vec3{0, 0, 1});
        nan_normal.normals[2].y = inf;
        expect_throw<std::invalid_argument>("infinite normal", [&] { make(nan_normal, {{0, identity_transform(), mat}}, {}); });
        expect_throw<std::invalid_argument>("texture index past the end", [&] { make(quad, {{0, identity_transform(), mat, 1}}, {ok}); });
        expect_throw<std::invalid_argument>("texture on a mesh with no uvs", [&] {
            Mesh no_uvs = quad;
            no_uvs.uvs.clear();
            make(no_uvs, {{0, identity_transform(), mat, 0}}, {ok});
        });
        Texture short_pixels = ok;
        short_pixels.rgba.pop_back();
        expect_throw<std::invalid_argument>("texture with too few bytes", [&] { make(quad, {}, {short_pixels}); });
        Texture empty;
        expect_throw<std::invalid_argument>("texture of size 0", [&] { make(quad, {}, {empty}); });

        bool accepted = true;
        try { make(quad, {{0, identity_transform(), mat}}, {ok}); } catch (...) { accepted = false; }
        check(accepted, "a texture nobody uses is accepted");
    }



    std::puts("devices");
    {
        // The CPU backend runs the same rendering code as the GPU kernel, so the two should draw
        // the same picture. A scene with every feature at once, rendered on each and compared.
        const Material mat = make_material({0.7f, 0.5f, 0.3f}, 0.9f, 0.4f, 0.0f, 60.0f);
        const Material textured = make_material({1, 1, 1}, 0.9f, 0.2f, 0.0f, 30.0f);
        Mesh smooth = make_icosphere(1.0f, 3);
        for (const Vec3 &v : smooth.vertices) smooth.normals.push_back(v);
        Mesh quad = make_quad(1.5f, 2.0f);
        const Texture tex = solid_2x2({{{255, 40, 40}, {40, 255, 40}, {40, 40, 255}, {240, 240, 60}}});
        std::vector<Light> lights = {{{-20, 20, 20}, 1.5f, 0.0f}, {{30, 50, 30}, 1.7f, 2.0f}};  // one soft: random shadow rays
        const std::vector<Instance> instances = {
            {0, make_transform({4, 0.5f, -12}, {0.1f, 0.3f, 0, 1}, 2.0f), mat},
            {1, make_transform({-4, 0, 0}, {0, 0.4f, 0, 1}), textured, 0}};

        auto build = [&](const Device d) {
            Renderer r(W, H, s.spheres, lights, s.floor, {smooth, quad}, instances, {tex}, d);
            r.set_ambient(0.15f);
            return r;
        };
        Renderer cpu = build(Device::Cpu);
        check(cpu.device() == Device::Cpu, "Device::Cpu gives a CPU renderer");

        bool have_gpu = true;
        try { Renderer probe(8, 8, {}, {}, Plane{}, {}, {}, {}, Device::Cuda); } catch (const CudaError &) { have_gpu = false; }
        if (have_gpu) {
            Renderer gpu = build(Device::Cuda);
            check(gpu.device() == Device::Cuda, "Device::Cuda gives a GPU renderer");
            const Image a = frame(gpu), b = frame(cpu);
            const Diff d = compare(a, b);
            std::printf("  GPU vs CPU, every feature at once: mean error %.3f, %.3f%% of pixels visibly off\n", d.mean, d.bad_fraction * 100.0);
            check(d.mean < 0.5 && d.bad_fraction < 0.005, "the CPU and the GPU draw the same picture");

            // And they stay together as the scene moves and the image accumulates.
            const std::vector<Transform> moved = {make_transform({3, 1, -11}, {0, 0.2f, 0, 1}, 2.0f), make_transform({-3, 0, 0.5f})};
            gpu.set_instance_transforms(moved);
            cpu.set_instance_transforms(moved);
            Image ga, ca;
            for (int f = 0; f < 6; ++f) { ga = frame(gpu, f); ca = frame(cpu, f); }
            const Diff after = compare(ga, ca);
            std::printf("  after moving instances and 6 accumulated frames: mean error %.3f\n", after.mean);
            check(after.mean < 0.5 && after.bad_fraction < 0.005, "they still agree after instances move and samples accumulate");
        } else {
            std::puts("  (no CUDA device: skipping the comparison with the GPU)");
        }

        // The CPU renderer alone: the same scene gives the same picture, however the threads share the rows.
        Renderer first = build(Device::Cpu), second = build(Device::Cpu);
        check(frame(first, 0) == frame(second, 0), "the CPU renders deterministically");
    }


    std::puts("depth, normal and object-ID images");
    {
        // A sphere, a textured quad (the sizes and places are what the numbers below come from), a floor.
        const Material mat = make_material({0.7f, 0.5f, 0.3f}, 0.9f, 0.2f, 0.0f, 40.0f);
        Plane floor{};
        floor.y = -2;
        floor.half_extent_x = 50;
        floor.z_min = -100;
        floor.z_max = 100;
        floor.checker_scale = 1;
        floor.material = mat;
        const Camera cam = camera_at_origin();
        auto depth_at = [&](const HostFrame &f, const int x, const int y) { return f.depth[y * W + x]; };
        auto id_at = [&](const HostFrame &f, const int x, const int y) { return f.object_id[y * W + x]; };
        auto normal_at = [&](const HostFrame &f, const int x, const int y) {
            return Vec3{f.normal[3 * (y * W + x)], f.normal[3 * (y * W + x) + 1], f.normal[3 * (y * W + x) + 2]};
        };
        const unsigned all = OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID;

        // The sphere: 8 away in front of the camera at its nearest point, so the depth there is 8.
        Renderer sphere_only(W, H, {{{0, 0, -10}, 2, mat}}, {{{0, 0, 2}, 1.0f, 0.0f}});
        HostFrame f = sphere_only.render_frame_host(cam, 0, 1, all);
        check(f.width == W && f.height == H && f.depth.size() == static_cast<size_t>(W) * H, "the images are width x height");
        check(std::fabs(depth_at(f, W / 2, H / 2) - 8.0f) < 0.02f, "depth at a sphere's near point is its distance (8)");
        const Vec3 n = normal_at(f, W / 2, H / 2);
        check(std::fabs(n.x) < 0.05f && std::fabs(n.y) < 0.05f && n.z > 0.99f, "the normal there faces the camera");
        check(id_at(f, W / 2, H / 2) == OBJECT_SPHERE, "the first sphere has ID OBJECT_SPHERE");
        check(is_inf(depth_at(f, 3, 3)) && depth_at(f, 3, 3) > 0 && id_at(f, 3, 3) == OBJECT_NONE &&
                normal_at(f, 3, 3).x == 0 && normal_at(f, 3, 3).z == 0,
            "where nothing is hit: depth +infinity, normal 0, ID OBJECT_NONE");

        // A flat quad 5 away: planar depth is 5 across the whole quad, even towards its edge, where the
        // ray to it is longer. A depth camera reports the former.
        Mesh quad = make_quad(3.0f, 1.0f);
        Renderer wall(W, H, {}, {{{0, 0, 2}, 1.0f, 0.0f}}, Plane{}, {quad}, {{0, identity_transform(), mat}});
        HostFrame g = wall.render_frame_host(cam, 0, 1, all);
        check(std::fabs(depth_at(g, W / 2 + 40, H / 2 + 20) - 5.0f) < 1e-3f && std::fabs(depth_at(g, W / 2 - 30, H / 2 - 15) - 5.0f) < 1e-3f,
            "depth is along the view axis: a flat wall is 5 deep everywhere, not farther at its edges");
        check(id_at(g, W / 2, H / 2) == OBJECT_INSTANCE, "the first mesh instance has ID OBJECT_INSTANCE");
        const Vec3 wn = normal_at(g, W / 2 + 40, H / 2 + 20);
        check(std::fabs(wn.z - 1.0f) < 1e-4f, "a wall's normal faces the camera");

        // The floor, whose depth at a row of pixels follows from the geometry: 2 / (2(y + 0.5)/H - 1).
        Renderer ground(W, H, {}, {{{0, 0, 2}, 1.0f, 0.0f}}, floor);
        HostFrame h = ground.render_frame_host(cam, 0, 1, all);
        const int row = H * 3 / 4;
        const float expect = 2.0f / (2.0f * (static_cast<float>(row) + 0.5f) / static_cast<float>(H) - 1.0f);
        check(id_at(h, W / 2, row) == OBJECT_FLOOR && std::fabs(depth_at(h, W / 2, row) - expect) < 0.01f * expect,
            "the floor has ID OBJECT_FLOOR and the depth the geometry says (" + std::to_string(expect) + ")");
        check(std::fabs(normal_at(h, W / 2, row).y - 1.0f) < 1e-5f, "the floor's normal is up");

        // The nearest object wins, and each has its own ID.
        Renderer both(W, H, {{{0, 0, -3}, 1, mat}, {{0, 0, -8}, 2, mat}}, {{{0, 0, 2}, 1.0f, 0.0f}});
        HostFrame b = both.render_frame_host(cam, 0, 1, all);
        check(id_at(b, W / 2, H / 2) == OBJECT_SPHERE && std::fabs(depth_at(b, W / 2, H / 2) - 2.0f) < 0.02f,
            "the nearer of two spheres is the one reported");

        // Only what is asked for is produced.
        const Frame depth_only = sphere_only.render_frame(cam, 0, 1, OUTPUT_DEPTH);
        check(depth_only.depth != nullptr && depth_only.rgb == nullptr && depth_only.normal == nullptr && depth_only.object_id == nullptr,
            "a frame has only the images that were asked for");
        check(depth_only.memory == sphere_only.memory() && depth_only.width == W && depth_only.height == H, "a frame says where it is and how big");
        expect_throw<std::invalid_argument>("no outputs", [&] { sphere_only.render_frame(cam, 0, 1, 0); });
        expect_throw<std::invalid_argument>("an unknown output", [&] { sphere_only.render_frame(cam, 0, 1, 64); });

        // Two features agreeing with each other: the RGB output is the picture render() draws.
        const uint32_t *argb = sphere_only.render(cam, 0, 1);
        const HostFrame rgb = sphere_only.render_frame_host(cam, 0, 1, OUTPUT_RGB);
        bool same = rgb.rgb.size() == static_cast<size_t>(W) * H * 3;
        for (size_t i = 0; same && i < static_cast<size_t>(W) * H; ++i)
            same = rgb.rgb[3 * i] == ((argb[i] >> 16) & 0xFF) && rgb.rgb[3 * i + 1] == ((argb[i] >> 8) & 0xFF) && rgb.rgb[3 * i + 2] == (argb[i] & 0xFF);
        check(same, "the RGB output is the picture render() draws, as red, green, blue bytes");
    }

    std::puts("ray queries");
    {
        const Material mat = make_material({0.7f, 0.5f, 0.3f}, 0.9f, 0.2f, 0.0f, 40.0f);
        Renderer r(W, H, {{{0, 0, -10}, 2, mat}}, {{{0, 0, 2}, 1.0f, 0.0f}}, Plane{}, {make_quad(3.0f, 1.0f)},
            {{0, make_transform({10, 0, -4}), mat}});
        auto cast = [&](const std::vector<float> &o, const std::vector<float> &d, const float max = INFINITY) {
            return r.cast_rays_host(o, d, max);
        };
        // Straight at the sphere: its front is 8 away, whatever the length of the direction.
        HostRayHits hit = cast({0, 0, 0, 0, 0, 0}, {0, 0, -1, 0, 0, -40});
        check(std::fabs(hit.distance[0] - 8.0f) < 1e-3f && std::fabs(hit.distance[1] - 8.0f) < 1e-3f,
            "a ray hits at its distance along the direction, however long the direction is");
        check(hit.object_id[0] == OBJECT_SPHERE && std::fabs(hit.normal[2] - 1.0f) < 1e-3f, "and reports the object and its normal");

        hit = cast({0, 0, 0}, {0, 0, -1}, 7.9f);
        check(is_inf(hit.distance[0]) && hit.object_id[0] == OBJECT_NONE, "a hit beyond max_distance is a miss");
        hit = cast({0, 0, 0}, {0, 0, -1}, 8.1f);
        check(hit.object_id[0] == OBJECT_SPHERE, "and a hit within it is not");

        hit = cast({0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 1, 0, 0});
        check(is_inf(hit.distance[0]) && is_inf(hit.distance[1]) && hit.object_id[1] == OBJECT_NONE,
            "a ray into empty space, and one with a zero direction, are misses");

        // The quad is modelled at z = -5 and the instance moves it by (10, 0, -4), so it is at z = -9 around x = 10.
        hit = cast({10, 0, 0}, {0, 0, -1});
        check(hit.object_id[0] == OBJECT_INSTANCE && std::fabs(hit.distance[0] - 9.0f) < 1e-3f, "mesh instances are hit, with their own IDs");

        // A ray query and a depth image are the same question: the ray through a pixel, and that pixel's depth.
        const Camera cam = camera_at_origin();
        const HostFrame depth = r.render_frame_host(cam, 0, 1, OUTPUT_DEPTH);
        const int px = W / 2 + 7, py = H / 2 - 5;
        const float sx = (2.0f * (static_cast<float>(px) + 0.5f) / static_cast<float>(W) - 1.0f) * cam.fov_scale * cam.aspect;
        const float sy = -(2.0f * (static_cast<float>(py) + 0.5f) / static_cast<float>(H) - 1.0f) * cam.fov_scale;
        const Vec3 d = {cam.right.x * sx + cam.up.x * sy + cam.forward.x, cam.right.y * sx + cam.up.y * sy + cam.forward.y,
            cam.right.z * sx + cam.up.z * sy + cam.forward.z};
        hit = cast({0, 0, 0}, {d.x, d.y, d.z});
        const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        const float along_axis = hit.distance[0] * (d.x * cam.forward.x + d.y * cam.forward.y + d.z * cam.forward.z) / len;
        check(is_finite_value(hit.distance[0]) && std::fabs(along_axis - depth.depth[py * W + px]) < 1e-3f,
            "the depth image is the ray query's distance, measured along the view axis");

        expect_throw<std::invalid_argument>("negative count", [&] { r.cast_rays(nullptr, nullptr, -1, 10.0f); });
        expect_throw<std::invalid_argument>("null pointers", [&] { r.cast_rays(nullptr, nullptr, 2, 10.0f); });
        expect_throw<std::invalid_argument>("max_distance 0", [&] { const float z[3] = {0, 0, 0}; r.cast_rays(z, z, 1, 0.0f); });
        expect_throw<std::invalid_argument>("NaN max_distance", [&] { const float z[3] = {0, 0, 0}; r.cast_rays(z, z, 1, nan); });
        expect_throw<std::invalid_argument>("NaN origin", [&] { const float o[3] = {nan, 0, 0}, dd[3] = {0, 0, -1}; r.cast_rays(o, dd, 1, 10.0f); });
        check(r.cast_rays_host({}, {}, 10.0f).distance.empty(), "no rays is fine");
        if (r.memory() == Memory::Host)
            expect_throw<std::invalid_argument>("device pointers on a CPU renderer", [&] { const float z[3] = {0, 0, 1}; r.cast_rays(z, z, 1, 10.0f, Memory::Device); });
    }

    std::puts("the CPU and GPU agree on these images");
    {
        const Material mat = make_material({0.7f, 0.5f, 0.3f}, 0.9f, 0.2f, 0.0f, 40.0f);
        Mesh smooth = make_icosphere(1.0f, 3);
        for (const Vec3 &v : smooth.vertices) smooth.normals.push_back(v);
        auto build = [&](const Device d) {
            return Renderer(W, H, s.spheres, s.lights, s.floor, {smooth, make_quad(1.5f, 1.0f)},
                {{0, make_transform({4, 0.5f, -12}, {0.1f, 0.3f, 0, 1}, 2.0f), mat}, {1, make_transform({-4, 0, 0}), mat}}, {}, d);
        };
        bool have_gpu = true;
        try { Renderer probe(8, 8, {}, {}, Plane{}, {}, {}, {}, Device::Cuda); } catch (const CudaError &) { have_gpu = false; }
        if (have_gpu) {
            Renderer gpu = build(Device::Cuda), cpu = build(Device::Cpu);
            const unsigned all = OUTPUT_RGB | OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID;
            const HostFrame a = gpu.render_frame_host(camera(), 0, 1, all), b = cpu.render_frame_host(camera(), 0, 1, all);
            size_t id_mismatch = 0;
            double depth_err = 0, normal_err = 0;
            size_t both_hit = 0;
            for (size_t i = 0; i < a.object_id.size(); ++i) {
                if (a.object_id[i] != b.object_id[i]) ++id_mismatch;
                if (is_finite_value(a.depth[i]) && is_finite_value(b.depth[i])) {
                    ++both_hit;
                    depth_err = std::max(depth_err, static_cast<double>(std::fabs(a.depth[i] - b.depth[i])));
                    for (int k = 0; k < 3; ++k) normal_err = std::max(normal_err, static_cast<double>(std::fabs(a.normal[3 * i + k] - b.normal[3 * i + k])));
                }
            }
            std::printf("  GPU vs CPU: %zu of %zu pixels differ in ID, worst depth difference %.5f, worst normal difference %.5f\n",
                id_mismatch, a.object_id.size(), depth_err, normal_err);
            check(id_mismatch <= a.object_id.size() / 2000 && depth_err < 0.01 && normal_err < 0.01 && both_hit > 1000,
                "depth, normals and object IDs are the same on the CPU and the GPU (to within a few edge pixels)");

            // The same rays, both devices.
            std::vector<float> origins, dirs;
            Lcg rng;
            for (int i = 0; i < 4000; ++i) {
                origins.insert(origins.end(), {rng.next() * 4 - 2, rng.next() * 4 - 2, rng.next() * 2});
                dirs.insert(dirs.end(), {rng.next() * 2 - 1, rng.next() * 2 - 1, -rng.next() - 0.1f});
            }
            const HostRayHits ra = gpu.cast_rays_host(origins, dirs, 100.0f), rb = cpu.cast_rays_host(origins, dirs, 100.0f);
            size_t mismatches = 0, hits = 0;
            for (size_t i = 0; i < ra.object_id.size(); ++i) {
                if (ra.object_id[i] != rb.object_id[i]) ++mismatches;
                if (ra.object_id[i] != OBJECT_NONE) ++hits;
            }
            std::printf("  4000 random rays: %zu hit something, %zu disagree\n", hits, mismatches);
            check(hits > 100 && mismatches <= 8, "ray queries give the same answers on the CPU and the GPU");
        } else {
            std::puts("  (no CUDA device: skipping the comparison with the GPU)");
        }
    }

    std::puts("ambient light");
    {
        // A sphere lit only from behind: the face the camera sees gets no direct light at all.
        const Material mat = make_material({0.8f, 0.4f, 0.2f}, 0.9f, 0.5f, 0.0f, 20.0f);
        Renderer r(W, H, {{{0, 0, -10}, 2, mat}}, {{{0, 0, -40}, 1.0f, 0.0f}});
        auto centre = [&] {
            const uint32_t *p = r.render(camera_at_origin(), 0, 1);
            return p[(H / 2) * W + W / 2];
        };
        check(r.ambient() == 0.0f, "ambient defaults to 0");
        check((centre() & 0xFFFFFF) == 0, "with no ambient, a face turned from every light is black");
        r.set_ambient(0.3f);
        const uint32_t lit = centre();
        const int red = static_cast<int>((lit >> 16) & 0xFF), blue = static_cast<int>(lit & 0xFF);
        // 0.3 * 0.9 * 0.8 = 0.216 linear, gamma-encoded: about 0.5 -> 128. Blue is a quarter of red's colour.
        std::printf("  centre pixel with ambient 0.3: red %d, blue %d\n", red, blue);
        check(red > 100 && red < 150 && blue < red, "ambient adds diffuse colour times the setting");
        r.set_ambient(0.0f);
        check((centre() & 0xFFFFFF) == 0, "setting it back to 0 restores black");

        expect_throw<std::invalid_argument>("negative ambient", [&] { r.set_ambient(-0.1f); });
        expect_throw<std::invalid_argument>("NaN ambient", [&] { r.set_ambient(nan); });
        expect_throw<std::invalid_argument>("infinite ambient", [&] { r.set_ambient(inf); });
        check(r.ambient() == 0.0f, "a rejected value changes nothing");
    }

    std::puts("meshes reject bad input");
    {
        const Material mat = s.spheres[0].material;
        const Mesh box = make_box({1, 1, 1});
        auto make = [&](const std::vector<Mesh> &m, const std::vector<Instance> &i) { Renderer r(W, H, {}, {}, Plane{}, m, i); };

        Mesh no_tris = box;
        no_tris.indices.clear();
        expect_throw<std::invalid_argument>("mesh with no triangles", [&] { make({no_tris}, {{0, identity_transform(), mat}}); });
        Mesh ragged = box;
        ragged.indices.pop_back();
        expect_throw<std::invalid_argument>("indices not a multiple of 3", [&] { make({ragged}, {{0, identity_transform(), mat}}); });
        Mesh out_of_range = box;
        out_of_range.indices[4] = 99;
        expect_throw<std::invalid_argument>("index past the last vertex", [&] { make({out_of_range}, {{0, identity_transform(), mat}}); });
        Mesh nan_vertex = box;
        nan_vertex.vertices[2].x = nan;
        expect_throw<std::invalid_argument>("NaN vertex", [&] { make({nan_vertex}, {{0, identity_transform(), mat}}); });
        expect_throw<std::invalid_argument>("instance names a missing mesh", [&] { make({box}, {{1, identity_transform(), mat}}); });
        expect_throw<std::invalid_argument>("negative mesh index", [&] { make({box}, {{-1, identity_transform(), mat}}); });
        expect_throw<std::invalid_argument>("singular instance transform", [&] {
            make({box}, {{0, Transform{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, mat}});
        });
        expect_throw<std::invalid_argument>("zero quaternion", [&] { (void)make_transform({0, 0, 0}, {0, 0, 0, 0}); });
        expect_throw<std::invalid_argument>("icosphere radius 0", [&] { (void)make_icosphere(0.0f, 2); });
        expect_throw<std::invalid_argument>("icosphere subdivisions 9", [&] { (void)make_icosphere(1.0f, 9); });

        // Meshes nobody instances cost nothing and are accepted.
        bool ok = true;
        try { make({box}, {}); } catch (...) { ok = false; }
        check(ok, "a mesh with no instances is accepted");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
        failures == 1 ? "" : "s");
    return failures;
}
