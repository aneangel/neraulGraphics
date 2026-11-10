// Tests for neraulgraphics::BatchRenderer: many worlds, each with its own cameras, drawn at once.
// Every batched view is compared with what an ordinary Renderer draws for that world from the same
// camera, and the worlds are checked not to see one another. Runs on every device available.

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
#include <neraulgraphics/batch.h>

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

// These tests are built with -ffast-math, so infinities are read from their bits.
bool is_inf(const float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7fffffffu) == 0x7f800000u;
}

constexpr int W = 96, H = 54;
constexpr float FOV = 1.5708f;

struct Lcg {
    uint32_t state = 777;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1 << 24);
    }
};

// A camera-to-world matrix (OpenGL convention: x right, y up, looking along -z) from a Camera.
std::vector<float> matrix_of(const Camera &c) {
    return {c.right.x, c.up.x, -c.forward.x, c.position.x, c.right.y, c.up.y, -c.forward.y, c.position.y,
        c.right.z, c.up.z, -c.forward.z, c.position.z};
}

// Appends a camera's matrix to a list of them.
void add_camera(std::vector<float> &list, const Camera &c) {
    const std::vector<float> m = matrix_of(c);
    list.insert(list.end(), m.begin(), m.end());
}

// Appends a transform's 12 floats to a list of them.
void add_transform(std::vector<float> &list, const Transform &t) { list.insert(list.end(), t.m, t.m + 12); }

Camera look(const Vec3 &from, const Vec3 &to) { return make_camera(from, to, {0, 1, 0}, FOV, W, H); }

const Material MAT = make_material({0.7f, 0.5f, 0.3f}, 0.9f, 0.3f, 0.0f, 40.0f);

// What every world holds: a box, a ball and a smooth ball.
struct Template {
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
};

Template make_template() {
    Mesh smooth = make_icosphere(1.0f, 2);
    for (const Vec3 &v : smooth.vertices) smooth.normals.push_back(v);
    Template t;
    t.meshes = {make_box({1, 1, 1}), make_icosphere(1.0f, 2), smooth};
    t.instances = {{0, identity_transform(), MAT}, {1, identity_transform(), make_material({0.2f, 0.6f, 0.3f}, 0.9f, 0.3f, 0, 40)},
        {2, identity_transform(), make_material({0.3f, 0.3f, 0.8f}, 0.9f, 0.3f, 0, 40)}};
    return t;
}

// Where world w puts instance i: different in every world, all within sight of a camera at the origin.
Transform place(const int w, const int i) {
    Lcg rng;
    rng.state = 1000 + 31 * w + i;
    return make_transform({-3.0f + 2.5f * static_cast<float>(i) + rng.next() - 0.5f, rng.next() - 0.5f - 0.3f * w, -6.0f - 3.0f * rng.next()},
        {rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f, 1.0f}, 0.6f + 0.6f * rng.next());
}

std::vector<float> all_matrices(const int worlds, const int per_world) {
    std::vector<float> m;
    for (int w = 0; w < worlds; ++w)
        for (int i = 0; i < per_world; ++i) add_transform(m, place(w, i));
    return m;
}

std::vector<Sphere> spheres() { return {{{0, -1.0f, -9}, 1.0f, make_material({0.8f, 0.8f, 0.9f}, 0.0f, 10, 0.8f, 800)}}; }
std::vector<Light> lights() { return {{{-10, 10, 10}, 1.4f, 0.0f}, {{15, 20, -5}, 0.8f, 0.0f}}; }
Plane floor_plane() {
    Plane f{};
    f.y = -2;
    f.half_extent_x = 30;
    f.z_min = -60;
    f.z_max = 5;
    f.checker_scale = 1;
    f.material = make_material({1, 1, 1}, 1, 0, 0, 1);
    return f;
}

// An ordinary Renderer holding what world w holds.
Renderer single_world(const Template &t, const int w, const Device d) {
    std::vector<Instance> inst = t.instances;
    for (size_t i = 0; i < inst.size(); ++i) inst[i].transform = place(w, static_cast<int>(i));
    return Renderer(W, H, spheres(), lights(), floor_plane(), t.meshes, inst, {}, d);
}

// Every device to test on: the CPU, and a GPU if there is one. NERAULGRAPHICS_DEVICE=cpu leaves the GPU out.
std::vector<Device> devices() {
    std::vector<Device> d = {Device::Cpu};
    if (const char *only = std::getenv("NERAULGRAPHICS_DEVICE"); only != nullptr && std::strcmp(only, "cpu") == 0) return d;
    try { Renderer probe(8, 8, {}, {}, Plane{}, {}, {}, {}, Device::Cuda); d.push_back(Device::Cuda); } catch (const CudaError &) {}
    return d;
}

const char *name(const Device d) { return d == Device::Cuda ? "GPU" : "CPU"; }

}  // namespace

int main() {
    const Template t = make_template();
    const unsigned all = OUTPUT_RGB | OUTPUT_DEPTH | OUTPUT_NORMAL | OUTPUT_OBJECT_ID;
    volatile float zero = 0.0f;
    const float nan = zero / zero;

    for (const Device dev : devices()) {
        std::printf("== %s ==\n", name(dev));

        std::puts("each view matches an ordinary renderer for its world");
        {
            const int worlds = 6;
            BatchRenderer b(W, H, worlds, 1, spheres(), lights(), floor_plane(), t.meshes, t.instances, {}, dev);
            check(b.world_count() == 6 && b.view_count() == 6 && b.instances_per_world() == 3 && b.device() == dev, "the batch knows its shape and device");
            const std::vector<float> poses = all_matrices(worlds, 3);
            b.set_instance_transforms(poses.data());

            const Camera cam = look({0, 0.5f, 0}, {0, 0, -8});
            std::vector<float> cams;
            for (int w = 0; w < worlds; ++w) add_camera(cams, cam);
            const HostBatchFrame f = b.render_host(cams.data(), FOV, 1, all);
            check(f.views == worlds && f.width == W && f.height == H && f.depth.size() == static_cast<size_t>(worlds) * W * H, "the batch holds views x height x width images");

            size_t id_diff = 0, total = 0;
            double depth_err = 0, rgb_err = 0;
            for (int w = 0; w < worlds; ++w) {
                Renderer one = single_world(t, w, dev);
                const HostFrame g = one.render_frame_host(cam, 0, 1, all);
                const size_t off = static_cast<size_t>(w) * W * H;
                for (size_t p = 0; p < static_cast<size_t>(W) * H; ++p, ++total) {
                    if (g.object_id[p] != f.object_id[off + p]) ++id_diff;
                    if (!is_inf(g.depth[p]) && !is_inf(f.depth[off + p])) depth_err = std::max(depth_err, static_cast<double>(std::fabs(g.depth[p] - f.depth[off + p])));
                    for (int k = 0; k < 3; ++k) rgb_err += std::abs(static_cast<int>(g.rgb[3 * p + k]) - static_cast<int>(f.rgb[3 * (off + p) + k]));
                }
            }
            std::printf("  %zu of %zu pixels differ in object ID, worst depth difference %.5f, mean colour difference %.4f\n",
                id_diff, total, depth_err, rgb_err / (3.0 * static_cast<double>(total)));
            check(id_diff <= total / 5000 && depth_err < 1e-3 && rgb_err / (3.0 * static_cast<double>(total)) < 0.05,
                "every batched view draws what a Renderer draws for that world");
            check(std::find(f.object_id.begin(), f.object_id.end(), OBJECT_INSTANCE + 2) != f.object_id.end(),
                "object IDs name an instance by its place in the world's list, the same in every world");
        }

        std::puts("worlds do not see one another");
        {
            // Two worlds, one camera each, at the same place. World 1's box sits right in front of that
            // camera; world 0's is farther away. World 0 must see only its own.
            Template one_box;
            one_box.meshes = {make_box({1, 1, 1})};
            one_box.instances = {{0, identity_transform(), MAT}};
            BatchRenderer b(W, H, 2, 1, {}, {{{0, 5, 5}, 1.0f, 0.0f}}, Plane{}, one_box.meshes, one_box.instances, {}, dev);
            std::vector<float> poses;
            for (const float z : {-8.0f, -3.0f}) add_transform(poses, make_transform({0, 0, z}));
            b.set_instance_transforms(poses.data());
            const Camera cam = look({0, 0, 0}, {0, 0, -1});
            std::vector<float> cams = matrix_of(cam);
            { const std::vector<float> once = cams; cams.insert(cams.end(), once.begin(), once.end()); }
            const HostBatchFrame f = b.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH | OUTPUT_OBJECT_ID);
            const size_t centre = static_cast<size_t>(H / 2) * W + W / 2, world1 = static_cast<size_t>(W) * H;
            std::printf("  the middle pixel's depth: world 0 %.3f (its box is at 8, front face 7), world 1 %.3f (3, front face 2)\n", f.depth[centre], f.depth[world1 + centre]);
            check(std::fabs(f.depth[centre] - 7.0f) < 0.02f && std::fabs(f.depth[world1 + centre] - 2.0f) < 0.02f,
                "each camera sees its own world's box, not the nearer one in the other world");

            // The same for a ray query, which is how a LiDAR is done.
            const std::vector<float> origins = {0, 0, 0, 0, 0, 0}, dirs = {0, 0, -1, 0, 0, -1};
            const HostRayHits hits = b.cast_rays_host(origins, dirs, 1, 100.0f);
            check(std::fabs(hits.distance[0] - 7.0f) < 1e-3f && std::fabs(hits.distance[1] - 2.0f) < 1e-3f && hits.object_id[0] == OBJECT_INSTANCE && hits.object_id[1] == OBJECT_INSTANCE,
                "ray queries see only their own world as well");
        }

        std::puts("poses from simulator bodies");
        {
            // Two bodies per world, and three instances: one on each body with an offset and a scale, one fixed.
            const int worlds = 4;
            BodyBinding bind;
            bind.bodies_per_world = 2;
            bind.body_of_instance = {0, 1, -1};
            bind.local_pose = {0.5f, 0, 0, 0, 0, 0, 1,   0, 0.4f, 0, 0, 0.3826834f, 0, 0.9238795f,   2, 0, -7, 0, 0, 0, 1};
            bind.scale = {1, 1, 1,   0.5f, 0.5f, 1.0f,   1, 1, 1};
            BatchRenderer b(W, H, worlds, 1, spheres(), lights(), floor_plane(), t.meshes, t.instances, {}, dev);

            Lcg rng;
            std::vector<float> bodies;
            for (int w = 0; w < worlds; ++w)
                for (int k = 0; k < 2; ++k)
                    bodies.insert(bodies.end(), {-1.5f + 3.0f * static_cast<float>(k) + 0.2f * static_cast<float>(w), 0.0f, -6.0f - rng.next(),
                        0.0f, 0.2588190f * (k ? 1.0f : -1.0f), 0.0f, 0.9659258f});
            b.bind_bodies(bind);
            b.set_body_poses(bodies.data());

            // The same placement worked out by hand, through the matrix path.
            BatchRenderer m(W, H, worlds, 1, spheres(), lights(), floor_plane(), t.meshes, t.instances, {}, dev);
            std::vector<float> matrices(static_cast<size_t>(worlds) * 3 * 12);
            for (int w = 0; w < worlds; ++w)
                for (int i = 0; i < 3; ++i) {
                    const float *lp = &bind.local_pose[7 * i], *sc = &bind.scale[3 * i];
                    float r[9], rb[9], rl[9], p[3] = {lp[0], lp[1], lp[2]};
                    auto q2m = [](const float *q, float *out) {
                        const float x = q[0], y = q[1], z = q[2], w2 = q[3];
                        const float v[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w2), 2 * (x * z + y * w2), 2 * (x * y + z * w2), 1 - 2 * (x * x + z * z),
                            2 * (y * z - x * w2), 2 * (x * z - y * w2), 2 * (y * z + x * w2), 1 - 2 * (x * x + y * y)};
                        std::memcpy(out, v, sizeof(v));
                    };
                    q2m(lp + 3, rl);
                    if (bind.body_of_instance[i] < 0) std::memcpy(r, rl, sizeof(r));
                    else {
                        const float *bp = &bodies[7 * (w * 2 + bind.body_of_instance[i])];
                        q2m(bp + 3, rb);
                        for (int a = 0; a < 3; ++a) {
                            for (int c = 0; c < 3; ++c) r[3 * a + c] = rb[3 * a] * rl[c] + rb[3 * a + 1] * rl[3 + c] + rb[3 * a + 2] * rl[6 + c];
                            p[a] = bp[a] + rb[3 * a] * lp[0] + rb[3 * a + 1] * lp[1] + rb[3 * a + 2] * lp[2];
                        }
                    }
                    float *out = &matrices[12 * (w * 3 + i)];
                    for (int a = 0; a < 3; ++a) {
                        for (int c = 0; c < 3; ++c) out[4 * a + c] = r[3 * a + c] * sc[c];
                        out[4 * a + 3] = p[a];
                    }
                }
            m.set_instance_transforms(matrices.data());

            const Camera cam = look({0, 0.5f, 0}, {0, 0, -8});
            std::vector<float> cams;
            for (int w = 0; w < worlds; ++w) add_camera(cams, cam);
            const HostBatchFrame a = b.render_host(cams.data(), FOV, 1, all), c = m.render_host(cams.data(), FOV, 1, all);
            size_t diff = 0;
            for (size_t i = 0; i < a.object_id.size(); ++i) diff += a.object_id[i] != c.object_id[i];
            check(diff <= a.object_id.size() / 5000 && std::find(a.object_id.begin(), a.object_id.end(), OBJECT_INSTANCE + 1) != a.object_id.end(),
                "body poses give the same picture as the matrices worked out by hand (body pose, then offset, then scale)");
        }

        std::puts("many worlds, several cameras each");
        {
            const int worlds = 200, cpw = 2;
            BatchRenderer b(W, H, worlds, cpw, spheres(), lights(), floor_plane(), t.meshes, t.instances, {}, dev);
            const std::vector<float> poses = all_matrices(worlds, 3);
            b.set_instance_transforms(poses.data());
            std::vector<float> cams;
            for (int v = 0; v < worlds * cpw; ++v) {
                const Camera c = look({0.3f * static_cast<float>(v % cpw), 0.5f, 0}, {0, 0, -8});
                add_camera(cams, c);
            }
            const HostBatchFrame f = b.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH | OUTPUT_OBJECT_ID);
            check(f.views == worlds * cpw, "200 worlds with 2 cameras each give 400 views");
            // Spot-check views from across the batch against ordinary renderers.
            bool all_match = true;
            for (const int v : {0, 1, 57, 198, 255, 399}) {
                const int w = v / cpw;
                Renderer one = single_world(t, w, dev);
                const Camera c = look({0.3f * static_cast<float>(v % cpw), 0.5f, 0}, {0, 0, -8});
                const HostFrame g = one.render_frame_host(c, 0, 1, OUTPUT_OBJECT_ID);
                size_t diff = 0;
                for (size_t p = 0; p < static_cast<size_t>(W) * H; ++p) diff += g.object_id[p] != f.object_id[static_cast<size_t>(v) * W * H + p];
                all_match = all_match && diff <= static_cast<size_t>(W) * H / 2000;
            }
            check(all_match, "views drawn from across the batch match ordinary renderers");
        }

        std::puts("edges: no instances, one instance, thousands");
        {
            BatchRenderer none(W, H, 3, 1, spheres(), lights(), floor_plane(), {}, {}, {}, dev);
            const Camera cam = look({0, 0, 0}, {0, 0, -8});
            std::vector<float> cams;
            for (int w = 0; w < 3; ++w) add_camera(cams, cam);
            none.set_instance_transforms(nullptr);
            check(none.render_host(cams.data(), FOV, 1, OUTPUT_RGB).rgb.size() == 3u * W * H * 3, "worlds with no instances are just the shared spheres and floor");

            Template one;
            one.meshes = {make_box({1, 1, 1})};
            one.instances = {{0, identity_transform(), MAT}};
            BatchRenderer single(W, H, 4, 1, {}, lights(), Plane{}, one.meshes, one.instances, {}, dev);
            std::vector<float> poses;
            for (int w = 0; w < 4; ++w) add_transform(poses, make_transform({0, 0, -4.0f - static_cast<float>(w)}));
            single.set_instance_transforms(poses.data());
            std::vector<float> cams4;
            for (int w = 0; w < 4; ++w) add_camera(cams4, cam);
            const HostBatchFrame f = single.render_host(cams4.data(), FOV, 1, OUTPUT_DEPTH);
            bool ok = true;
            for (int w = 0; w < 4; ++w) ok = ok && std::fabs(f.depth[static_cast<size_t>(w) * W * H + (H / 2) * W + W / 2] - (3.0f + static_cast<float>(w))) < 0.02f;
            check(ok, "a world with a single instance (a tree that is one leaf) renders right");

            // A thousand identical boxes in one world: the tree has to split coincident boxes.
            Template pile;
            pile.meshes = {make_box({0.2f, 0.2f, 0.2f})};
            pile.instances.assign(1000, Instance{0, identity_transform(), MAT});
            BatchRenderer many(W, H, 1, 1, {}, lights(), Plane{}, pile.meshes, pile.instances, {}, dev);
            std::vector<float> same;
            for (int i = 0; i < 1000; ++i) add_transform(same, make_transform({0.0f, 0.0f, -5.0f}));
            many.set_instance_transforms(same.data());
            const std::vector<float> one_camera = matrix_of(cam);
            const HostBatchFrame g = many.render_host(one_camera.data(), FOV, 1, OUTPUT_DEPTH);
            check(std::fabs(g.depth[(H / 2) * W + W / 2] - 4.8f) < 0.02f, "a thousand boxes in the same place are all handled (the nearest face is 4.8 away)");
        }


        std::puts("large scenes: the GPU's trees and the CPU's agree");
        if (dev == Device::Cuda) {
            // 20,000 boxes scattered through one world, and 300 small worlds of 40: the CPU builds its trees by the
            // surface-area heuristic and the GPU by sorted Morton codes, so agreeing is no accident.
            for (const auto &shape : {std::pair<int, int>{1, 20000}, {300, 40}}) {
                const int worlds = shape.first, per = shape.second;
                Template big;
                big.meshes = {make_box({0.4f, 0.4f, 0.4f}), make_icosphere(0.5f, 1)};
                for (int i = 0; i < per; ++i) big.instances.push_back({i % 2, identity_transform(), MAT});
                std::vector<float> poses;
                Lcg rng;
                const float spread = worlds == 1 ? 60.0f : 3.0f;
                for (int g = 0; g < worlds * per; ++g)
                    add_transform(poses, make_transform({spread * (rng.next() - 0.5f), spread * 0.5f * (rng.next() - 0.5f), -10.0f - spread * rng.next()},
                        {rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f, 1.0f}, 0.5f + rng.next()));
                BatchRenderer gpu(W, H, worlds, 1, {}, lights(), Plane{}, big.meshes, big.instances, {}, Device::Cuda);
                BatchRenderer cpu(W, H, worlds, 1, {}, lights(), Plane{}, big.meshes, big.instances, {}, Device::Cpu);
                gpu.set_instance_transforms(poses.data());
                cpu.set_instance_transforms(poses.data());
                std::vector<float> cams;
                for (int w = 0; w < worlds; ++w) add_camera(cams, look({0, 0, 5}, {0, 0, -30}));
                const HostBatchFrame a = gpu.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH | OUTPUT_OBJECT_ID);
                const HostBatchFrame b = cpu.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH | OUTPUT_OBJECT_ID);
                size_t diff = 0, hits = 0;
                double worst = 0;
                for (size_t i = 0; i < a.object_id.size(); ++i) {
                    diff += a.object_id[i] != b.object_id[i];
                    hits += a.object_id[i] != OBJECT_NONE;
                    // Depth only where both agree what was hit: a ray along an exact edge may choose differently.
                    if (a.object_id[i] == b.object_id[i] && !is_inf(a.depth[i]) && !is_inf(b.depth[i]))
                        worst = std::max(worst, static_cast<double>(std::fabs(a.depth[i] - b.depth[i])));
                }
                std::printf("  %d world%s of %d instances: %zu of %zu pixels hit something, %zu differ in ID, worst depth difference %.5f\n",
                    worlds, worlds == 1 ? "" : "s", per, hits, a.object_id.size(), diff, worst);
                check(hits > 1000 && diff <= a.object_id.size() / 2000 && worst < 0.01,
                    "the GPU's trees see the same scene as the CPU's (" + std::to_string(worlds) + " x " + std::to_string(per) + " instances)");
            }
        } else {
            std::puts("  (CPU only: compared with the GPU on the GPU's turn)");
        }

        std::puts("bad input is rejected, and changes nothing");
        {
            BatchRenderer b(W, H, 2, 1, spheres(), lights(), floor_plane(), t.meshes, t.instances, {}, dev);
            const std::vector<float> poses = all_matrices(2, 3);
            b.set_instance_transforms(poses.data());
            const Camera cam = look({0, 0, 0}, {0, 0, -8});
            std::vector<float> cams = matrix_of(cam);
            { const std::vector<float> once = cams; cams.insert(cams.end(), once.begin(), once.end()); }
            const HostBatchFrame before = b.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH);

            std::vector<float> singular = poses;
            std::fill(singular.begin() + 12 * 4, singular.begin() + 12 * 4 + 12, 0.0f);
            expect_throw<std::invalid_argument>("a singular instance matrix", [&] { b.set_instance_transforms(singular.data()); });
            std::vector<float> bad = poses;
            bad[7] = nan;
            expect_throw<std::invalid_argument>("a NaN instance matrix", [&] { b.set_instance_transforms(bad.data()); });
            expect_throw<std::invalid_argument>("no outputs", [&] { b.render(cams.data(), FOV, 1, 0); });
            expect_throw<std::invalid_argument>("an unknown output", [&] { b.render(cams.data(), FOV, 1, 64); });
            expect_throw<std::invalid_argument>("samples_per_pixel 0", [&] { b.render(cams.data(), FOV, 0, OUTPUT_RGB); });
            expect_throw<std::invalid_argument>("fov 0", [&] { b.render(cams.data(), 0.0f, 1, OUTPUT_RGB); });
            expect_throw<std::invalid_argument>("fov 4", [&] { b.render(cams.data(), 4.0f, 1, OUTPUT_RGB); });
            expect_throw<std::invalid_argument>("NaN fov", [&] { b.render(cams.data(), nan, 1, OUTPUT_RGB); });
            expect_throw<std::invalid_argument>("negative frame", [&] { b.render(cams.data(), FOV, 1, OUTPUT_RGB, -1); });
            std::vector<float> nan_cam = cams;
            nan_cam[3] = nan;
            expect_throw<std::invalid_argument>("a NaN camera", [&] { b.render(nan_cam.data(), FOV, 1, OUTPUT_RGB); });
            expect_throw<std::invalid_argument>("body poses before bind_bodies", [&] { const float q[14] = {}; b.set_body_poses(q); });
            BodyBinding wrong;
            wrong.bodies_per_world = 1;
            wrong.body_of_instance = {0, 0, 5};
            wrong.local_pose.assign(21, 0.0f);
            wrong.scale.assign(9, 1.0f);
            for (int i = 0; i < 3; ++i) wrong.local_pose[7 * i + 6] = 1.0f;
            expect_throw<std::invalid_argument>("an instance on a body that does not exist", [&] { b.bind_bodies(wrong); });
            wrong.body_of_instance = {0, 0};
            expect_throw<std::invalid_argument>("a binding with too few instances", [&] { b.bind_bodies(wrong); });
            expect_throw<std::invalid_argument>("rays of the wrong count", [&] { b.cast_rays_host({0, 0, 0}, {0, 0, -1}, 1, 10.0f); });
            expect_throw<std::invalid_argument>("zero worlds", [&] { BatchRenderer(W, H, 0, 1, {}, {}, Plane{}, {}, {}, {}, dev); });
            expect_throw<std::invalid_argument>("zero cameras", [&] { BatchRenderer(W, H, 1, 0, {}, {}, Plane{}, {}, {}, {}, dev); });
            check(b.render_host(cams.data(), FOV, 1, OUTPUT_DEPTH).depth == before.depth, "after all that rejected input, the scene is exactly as it was");
            if (dev == Device::Cpu)
                expect_throw<std::invalid_argument>("device pointers on a CPU renderer", [&] { b.set_instance_transforms(poses.data(), Memory::Device); });
        }
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures;
}
