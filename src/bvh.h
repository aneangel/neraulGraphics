#ifndef NERAULGRAPHICS_BVH_H
#define NERAULGRAPHICS_BVH_H

// Bounding volume hierarchy builder. Plain C++ with no CUDA, so the same tree
// layout is built on the host and read by the kernel, and the builder can be
// tested without a GPU.
//
// The tree is binary and stored flat. The two children of an interior node sit
// next to each other (right = left + 1), so a node only has to name one index.

#include <cstdint>
#include <vector>

namespace neraulgraphics::bvh {

struct Aabb {
    float lo[3];
    float hi[3];
};

// An inverted box that any real box grows from.
Aabb empty_aabb();
void grow(Aabb &box, const Aabb &other);
float surface_area(const Aabb &box);

// 32 bytes, 16-aligned: the kernel reads a node as two float4 loads, and the
// field order below is what makes those loads line up.
//   interior: count == 0, a is the index of the left child
//   leaf:     count > 0,  a is the first primitive (a position in Bvh::order)
struct alignas(16) Node {
    float lo[3];
    int32_t a;
    float hi[3];
    int32_t count;
};
static_assert(sizeof(Node) == 32, "the kernel reads Node as two float4");

struct Bvh {
    std::vector<Node> nodes;     // nodes[0] is the root; empty if there were no primitives
    std::vector<uint32_t> order; // primitive indices, arranged so each leaf is a contiguous run
};

// Binned surface-area-heuristic build. A leaf holds at most max_leaf_size
// primitives unless the depth limit is hit first. Primitives with identical
// centroids are still split (by index), so a pile of coincident triangles does
// not become one huge leaf.
Bvh build(const std::vector<Aabb> &bounds, int max_leaf_size);

// Recomputes every box from new primitive bounds, keeping the tree's shape. One
// pass, no sorting and no cost search, so it is far cheaper than build(). It works
// because a parent's index is always lower than its children's, so walking the
// nodes backwards visits children first. The boxes stay correct but loosen as
// primitives drift from where the shape was chosen: check sah_cost() and call
// build() again when it has grown too much.
void refit(Bvh &tree, const std::vector<Aabb> &bounds);

// Surface-area-heuristic cost of the tree: the expected work per ray, relative to
// the root's area. Lower is better; compare a refitted tree against its own cost
// when freshly built to see how much quality has been lost.
float sah_cost(const Bvh &tree);

// The deepest a tree can get. The kernel's traversal stack is sized from this.
constexpr int MAX_DEPTH = 60;

}  // namespace neraulgraphics::bvh

#endif  // NERAULGRAPHICS_BVH_H
