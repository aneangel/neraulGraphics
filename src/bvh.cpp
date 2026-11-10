#include "bvh.h"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace neraulgraphics::bvh {

Aabb empty_aabb() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    return Aabb{{inf, inf, inf}, {-inf, -inf, -inf}};
}

void grow(Aabb &box, const Aabb &other) {
    for (int i = 0; i < 3; ++i) {
        box.lo[i] = std::min(box.lo[i], other.lo[i]);
        box.hi[i] = std::max(box.hi[i], other.hi[i]);
    }
}

float surface_area(const Aabb &box) {
    const float dx = box.hi[0] - box.lo[0];
    const float dy = box.hi[1] - box.lo[1];
    const float dz = box.hi[2] - box.lo[2];
    if (dx < 0.0f || dy < 0.0f || dz < 0.0f) return 0.0f;  // empty
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

namespace {

// More bins find better splits and cost more to evaluate. 16 is the usual trade.
constexpr int BINS = 16;

// Cost of visiting a node relative to testing one primitive.
constexpr float TRAVERSAL_COST = 1.0f;

struct Task {
    int node;
    uint32_t begin;
    uint32_t end;
    int depth;
};

}  // namespace

Bvh build(const std::vector<Aabb> &bounds, const int max_leaf_size) {
    Bvh out;
    const uint32_t n = static_cast<uint32_t>(bounds.size());
    if (n == 0) return out;

    out.order.resize(n);
    std::iota(out.order.begin(), out.order.end(), 0u);

    std::vector<std::array<float, 3>> centroid(n);
    for (uint32_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) centroid[i][k] = 0.5f * (bounds[i].lo[k] + bounds[i].hi[k]);

    // A binary tree with n leaves has 2n-1 nodes at most.
    out.nodes.reserve(2 * static_cast<size_t>(n));
    out.nodes.emplace_back();

    // Explicit stack instead of recursion, so a deep tree cannot overflow the call stack.
    std::vector<Task> tasks;
    tasks.push_back({0, 0, n, 0});

    while (!tasks.empty()) {
        const Task task = tasks.back();
        tasks.pop_back();
        const uint32_t count = task.end - task.begin;
        uint32_t *prims = out.order.data() + task.begin;

        Aabb box = empty_aabb();
        Aabb cbox = empty_aabb();  // centroid bounds: what the bins are laid over
        for (uint32_t i = 0; i < count; ++i) {
            grow(box, bounds[prims[i]]);
            for (int k = 0; k < 3; ++k) {
                cbox.lo[k] = std::min(cbox.lo[k], centroid[prims[i]][k]);
                cbox.hi[k] = std::max(cbox.hi[k], centroid[prims[i]][k]);
            }
        }

        Node node{};
        for (int k = 0; k < 3; ++k) {
            node.lo[k] = box.lo[k];
            node.hi[k] = box.hi[k];
        }

        // Try each axis; keep the split with the lowest surface-area cost.
        int best_axis = -1;
        int best_bin = 0;
        float best_cost = std::numeric_limits<float>::infinity();
        if (count > 1) {
            for (int axis = 0; axis < 3; ++axis) {
                const float extent = cbox.hi[axis] - cbox.lo[axis];
                if (!(extent > 0.0f)) continue;
                const float scale = static_cast<float>(BINS) / extent;

                int bin_count[BINS] = {};
                Aabb bin_box[BINS];
                for (auto &b : bin_box) b = empty_aabb();
                for (uint32_t i = 0; i < count; ++i) {
                    const int b = std::min(BINS - 1,
                        static_cast<int>((centroid[prims[i]][axis] - cbox.lo[axis]) * scale));
                    ++bin_count[b];
                    grow(bin_box[b], bounds[prims[i]]);
                }

                // Cost of every possible split plane: a sweep from each end gives the
                // area and count on each side without rescanning the primitives.
                float right_area[BINS - 1];
                int right_count[BINS - 1];
                Aabb acc = empty_aabb();
                int running = 0;
                for (int i = BINS - 1; i >= 1; --i) {
                    grow(acc, bin_box[i]);
                    running += bin_count[i];
                    right_area[i - 1] = surface_area(acc);
                    right_count[i - 1] = running;
                }
                acc = empty_aabb();
                running = 0;
                for (int i = 0; i < BINS - 1; ++i) {
                    grow(acc, bin_box[i]);
                    running += bin_count[i];
                    if (running == 0 || right_count[i] == 0) continue;
                    const float cost = static_cast<float>(running) * surface_area(acc) +
                        static_cast<float>(right_count[i]) * right_area[i];
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_axis = axis;
                        best_bin = i;
                    }
                }
            }
        }

        // A leaf if it is small enough and splitting would not pay for itself.
        const float leaf_cost = static_cast<float>(count) * surface_area(box);
        const float split_cost = TRAVERSAL_COST * surface_area(box) + best_cost;
        const bool small = count <= static_cast<uint32_t>(max_leaf_size);
        if (count == 1 || task.depth >= MAX_DEPTH || (small && (best_axis < 0 || leaf_cost <= split_cost))) {
            node.a = static_cast<int32_t>(task.begin);
            node.count = static_cast<int32_t>(count);
            out.nodes[task.node] = node;
            continue;
        }

        uint32_t mid;
        if (best_axis >= 0) {
            const float extent = cbox.hi[best_axis] - cbox.lo[best_axis];
            const float scale = static_cast<float>(BINS) / extent;
            // Same expression as the binning above, so primitives land on the side they were counted on.
            const auto *p = std::partition(prims, prims + count, [&](const uint32_t prim) {
                const int b = std::min(BINS - 1,
                    static_cast<int>((centroid[prim][best_axis] - cbox.lo[best_axis]) * scale));
                return b <= best_bin;
            });
            mid = task.begin + static_cast<uint32_t>(p - prims);
            if (mid == task.begin || mid == task.end) mid = task.begin + count / 2;
        } else {
            // Every centroid is the same point: no plane separates them, so halve by index.
            mid = task.begin + count / 2;
        }

        const int left = static_cast<int>(out.nodes.size());
        out.nodes.resize(out.nodes.size() + 2);
        node.a = left;
        node.count = 0;
        out.nodes[task.node] = node;
        tasks.push_back({left + 1, mid, task.end, task.depth + 1});
        tasks.push_back({left, task.begin, mid, task.depth + 1});
    }
    return out;
}

void refit(Bvh &tree, const std::vector<Aabb> &bounds) {
    for (size_t i = tree.nodes.size(); i-- > 0;) {
        Node &node = tree.nodes[i];
        Aabb box = empty_aabb();
        if (node.count > 0) {
            for (int k = 0; k < node.count; ++k) grow(box, bounds[tree.order[node.a + k]]);
        } else {
            for (const Node &child : {tree.nodes[node.a], tree.nodes[node.a + 1]})
                grow(box, Aabb{{child.lo[0], child.lo[1], child.lo[2]}, {child.hi[0], child.hi[1], child.hi[2]}});
        }
        for (int k = 0; k < 3; ++k) {
            node.lo[k] = box.lo[k];
            node.hi[k] = box.hi[k];
        }
    }
}

float sah_cost(const Bvh &tree) {
    if (tree.nodes.empty()) return 0.0f;
    double cost = 0.0;
    for (const Node &node : tree.nodes) {
        const float area = surface_area(Aabb{{node.lo[0], node.lo[1], node.lo[2]}, {node.hi[0], node.hi[1], node.hi[2]}});
        cost += node.count > 0 ? static_cast<double>(area) * node.count : static_cast<double>(area) * TRAVERSAL_COST;
    }
    const Node &root = tree.nodes[0];
    const float root_area = surface_area(Aabb{{root.lo[0], root.lo[1], root.lo[2]}, {root.hi[0], root.hi[1], root.hi[2]}});
    return root_area > 0.0f ? static_cast<float>(cost / root_area) : 0.0f;
}

}  // namespace neraulgraphics::bvh
