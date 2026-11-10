#include "instances.h"
#include <algorithm>
#include <cmath>

namespace neraulgraphics::detail {

InstanceSet::InstanceSet(const BuiltMeshes &built, const std::vector<Instance> &instances, const int worlds,
    const bool build_trees)
    : worlds_(worlds), build_trees_(build_trees), per_world_(instances.size()), mesh_root_(built.roots),
      mesh_bounds_(built.bounds) {
    records.resize(size());
    trees_.resize(worlds_);
    built_cost_.assign(worlds_, 0.0f);
    roots.assign(worlds_, 0);
    std::vector<float> matrices;
    matrices.reserve(size() * 12);
    for (int w = 0; w < worlds_; ++w) {
        for (size_t i = 0; i < per_world_; ++i) {
            if (w == 0) mesh_.push_back(instances[i].mesh);
            InstanceRecord &r = records[w * per_world_ + i];
            r.texture = instances[i].texture;
            r.material = instances[i].material;
            matrices.insert(matrices.end(), instances[i].transform.m, instances[i].transform.m + 12);
        }
    }
    update(matrices.data());
}

void InstanceSet::update(const float *matrices) {
    boxes_.resize(size());
    for (size_t g = 0; g < size(); ++g) {
        Transform t, inverse;
        for (int k = 0; k < 12; ++k) t.m[k] = matrices[12 * g + k];
        invert_affine(t, inverse);  // checked invertible before this point
        InstanceRecord &r = records[g];
        for (int k = 0; k < 12; ++k) r.inv[k] = inverse.m[k];
        const size_t i = g % per_world_;
        r.blas_root = mesh_root_[mesh_[i]];
        boxes_[g] = transform_aabb(t, mesh_bounds_[mesh_[i]]);
    }

    nodes.clear();
    for (int w = 0; w < worlds_ && build_trees_; ++w) {
        const std::vector<bvh::Aabb> boxes(boxes_.begin() + static_cast<std::ptrdiff_t>(w * per_world_),
            boxes_.begin() + static_cast<std::ptrdiff_t>((w + 1) * per_world_));
        bvh::Bvh &tree = trees_[w];
        // Refit is one cheap pass; a full build searches for good splits. Refit while the tree is
        // still good, and build again once the boxes have loosened too far. 1.3x the cost it had
        // when built is where traces start to visibly slow.
        constexpr float REBUILD_ABOVE = 1.3f;
        bool rebuilt = tree.nodes.empty();
        if (!rebuilt) {
            bvh::refit(tree, boxes);
            rebuilt = bvh::sah_cost(tree) > REBUILD_ABOVE * built_cost_[w];
        }
        if (rebuilt) {
            tree = bvh::build(boxes, 1);
            built_cost_[w] = bvh::sah_cost(tree);
        }
        flatten(w, tree);
    }
}

// A tree of the builder's layout (children side by side, leaves holding runs) as nodes of the
// shared forest (children named, one instance per leaf, a leaf a node of its own).
void InstanceSet::flatten(const int world, const bvh::Bvh &tree) {
    const int base = static_cast<int>(nodes.size());
    nodes.resize(nodes.size() + tree.nodes.size());
    roots[world] = base;  // the builder's root is its first node
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const bvh::Node &n = tree.nodes[i];
        const int self = base + static_cast<int>(i);
        for (int k = 0; k < 3; ++k) {
            nodes[self].lo[k] = n.lo[k];
            nodes[self].hi[k] = n.hi[k];
        }
        if (n.count == 0) {
            nodes[self].left = base + n.a;
            nodes[self].right = base + n.a + 1;
        } else if (n.count == 1) {
            nodes[self].left = ~static_cast<int>(world * per_world_ + tree.order[n.a]);
            nodes[self].right = -1;
        } else {
            // A run of several instances, which only a tree cut off at its depth limit has: this node
            // takes the two halves of the run as children, below it.
            const uint32_t first = static_cast<uint32_t>(n.a), half = static_cast<uint32_t>(n.count) / 2;
            const int left = append_run(world, tree.order, first, half);
            const int right = append_run(world, tree.order, first + half, static_cast<uint32_t>(n.count) - half);
            nodes[self].left = left;
            nodes[self].right = right;
        }
    }
}

// Appends a balanced subtree over order[first, first + count) and returns its root's index.
int InstanceSet::append_run(const int world, const std::vector<uint32_t> &order, const uint32_t first, const uint32_t count) {
    const int self = static_cast<int>(nodes.size());
    nodes.emplace_back();
    if (count == 1) {
        const bvh::Aabb &b = boxes_[world * per_world_ + order[first]];
        for (int k = 0; k < 3; ++k) {
            nodes[self].lo[k] = b.lo[k];
            nodes[self].hi[k] = b.hi[k];
        }
        nodes[self].left = ~static_cast<int>(world * per_world_ + order[first]);
        nodes[self].right = -1;
        return self;
    }
    const uint32_t half = count / 2;
    const int left = append_run(world, order, first, half);
    const int right = append_run(world, order, first + half, count - half);
    for (int k = 0; k < 3; ++k) {
        nodes[self].lo[k] = std::min(nodes[left].lo[k], nodes[right].lo[k]);
        nodes[self].hi[k] = std::max(nodes[left].hi[k], nodes[right].hi[k]);
    }
    nodes[self].left = left;
    nodes[self].right = right;
    return self;
}

namespace {

// Rotation matrix of a quaternion given as x, y, z, w, row-major.
void quat_to_matrix(const float *q, double r[9]) {
    double x = q[0], y = q[1], z = q[2], w = q[3];
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    if (n > 0.0) { x /= n; y /= n; z /= n; w /= n; } else { x = y = z = 0.0; w = 1.0; }
    r[0] = 1 - 2 * (y * y + z * z); r[1] = 2 * (x * y - z * w);     r[2] = 2 * (x * z + y * w);
    r[3] = 2 * (x * y + z * w);     r[4] = 1 - 2 * (x * x + z * z); r[5] = 2 * (y * z - x * w);
    r[6] = 2 * (x * z - y * w);     r[7] = 2 * (y * z + x * w);     r[8] = 1 - 2 * (x * x + y * y);
}

}  // namespace

void body_poses_to_matrices(const BodyBinding &b, const int worlds, const float *body_poses, float *out) {
    const size_t count = b.body_of_instance.size();
    for (int w = 0; w < worlds; ++w) {
        for (size_t i = 0; i < count; ++i) {
            const float *local = &b.local_pose[7 * i];
            const float *scale = &b.scale[3 * i];
            double rl[9], r[9], p[3] = {local[0], local[1], local[2]};
            quat_to_matrix(local + 3, rl);
            if (b.body_of_instance[i] < 0) {
                for (int k = 0; k < 9; ++k) r[k] = rl[k];
            } else {
                const float *body = &body_poses[7 * (static_cast<size_t>(w) * b.bodies_per_world + b.body_of_instance[i])];
                double rb[9];
                quat_to_matrix(body + 3, rb);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col)
                        r[3 * row + col] = rb[3 * row] * rl[col] + rb[3 * row + 1] * rl[3 + col] + rb[3 * row + 2] * rl[6 + col];
                    p[row] = body[row] + rb[3 * row] * local[0] + rb[3 * row + 1] * local[1] + rb[3 * row + 2] * local[2];
                }
            }
            float *m = out + 12 * (static_cast<size_t>(w) * count + i);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) m[4 * row + col] = static_cast<float>(r[3 * row + col] * scale[col]);
                m[4 * row + 3] = static_cast<float>(p[row]);
            }
        }
    }
}

}  // namespace neraulgraphics::detail
