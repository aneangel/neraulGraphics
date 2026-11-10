#ifndef NERAULGRAPHICS_CUDA_TLAS_CUH
#define NERAULGRAPHICS_CUDA_TLAS_CUH

// Builds the trees over instances on the GPU, from the instances' transforms, every time they move.
// Included only by cuda_backend.cu, after it defines CUDA_CHECK.
//
// The method is a linear BVH (Karras 2012): each instance gets a Morton code, its position along a space-filling
// curve, with its world's number in the bits above, so that sorting the codes puts each world's instances
// together and neighbours in space next to one another. A binary radix tree over the sorted codes is then
// built with every node found independently of the rest, so all of them run at once. The tree is the
// same shape whatever order the work happens in. Boxes are filled in from the leaves up, and one more
// pass finds the node that roots each world's tree.
//
// This replaces the host builder for a GPU renderer. It is built from scratch on every update, not refitted: it costs
// a few hundred microseconds for 100,000 instances, so there is no tree to grow stale, and no cost to
// watch. The trees are not as good as the host's surface-area-heuristic ones, and that is the price.

#include <cstdint>
#include <cub/cub.cuh>
#include "bvh.h"
#include "mesh_dev.h"

namespace neraulgraphics::device {
namespace tlas {

// A float as an int whose order is the float's order, so atomicMin and atomicMax can work on it.
__device__ __forceinline__ int to_ordered(const float f) {
    const int i = __float_as_int(f);
    return i ^ ((i >> 31) & 0x7fffffff);
}
__device__ __forceinline__ float from_ordered(const int i) { return __int_as_float(i ^ ((i >> 31) & 0x7fffffff)); }

// The inverse of the affine transform with top rows m (12 floats, row-major), into out. A singular
// matrix gives infinities: the caller promised an invertible one.
__device__ __forceinline__ void invert_affine(const float *m, float *out) {
    const float c00 = m[5] * m[10] - m[6] * m[9];
    const float c01 = m[6] * m[8] - m[4] * m[10];
    const float c02 = m[4] * m[9] - m[5] * m[8];
    const float inv = 1.0f / (m[0] * c00 + m[1] * c01 + m[2] * c02);
    const float r[9] = {c00 * inv, (m[2] * m[9] - m[1] * m[10]) * inv, (m[1] * m[6] - m[2] * m[5]) * inv,
        c01 * inv, (m[0] * m[10] - m[2] * m[8]) * inv, (m[2] * m[4] - m[0] * m[6]) * inv,
        c02 * inv, (m[1] * m[8] - m[0] * m[9]) * inv, (m[0] * m[5] - m[1] * m[4]) * inv};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) out[4 * row + col] = r[3 * row + col];
        out[4 * row + 3] = -(r[3 * row] * m[3] + r[3 * row + 1] * m[7] + r[3 * row + 2] * m[11]);
    }
}

// Spreads the low 10 bits of v so there are two zero bits between each.
__device__ __forceinline__ uint32_t spread_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// One thread per world: bounds that any centre will move.
__global__ void init_world_bounds(int *bounds, const int worlds) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= worlds) return;
    for (int k = 0; k < 3; ++k) {
        bounds[6 * w + k] = to_ordered(1e30f);        // minimum of the centres
        bounds[6 * w + 3 + k] = to_ordered(-1e30f);   // maximum of the centres
    }
}

// One thread per instance: its inverse transform for the tracer, its box in the world, and a vote on its world's bounds.
__global__ void place_instances(const float *matrices, const float *local_bounds, const int per_world, const int total,
    Instance *instances, float *aabb, int *world_bounds) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const float *m = matrices + 12 * static_cast<size_t>(g);
    float inv[12];
    invert_affine(m, inv);
    for (int row = 0; row < 3; ++row) instances[g].inv[row] = make_float4(inv[4 * row], inv[4 * row + 1], inv[4 * row + 2], inv[4 * row + 3]);

    const float *lb = local_bounds + 6 * (g % per_world);
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int corner = 0; corner < 8; ++corner) {
        const float p[3] = {(corner & 1) ? lb[3] : lb[0], (corner & 2) ? lb[4] : lb[1], (corner & 4) ? lb[5] : lb[2]};
        for (int a = 0; a < 3; ++a) {
            const float w = m[4 * a] * p[0] + m[4 * a + 1] * p[1] + m[4 * a + 2] * p[2] + m[4 * a + 3];
            lo[a] = fminf(lo[a], w);
            hi[a] = fmaxf(hi[a], w);
        }
    }
    const int world = g / per_world;
    for (int a = 0; a < 3; ++a) {
        aabb[6 * static_cast<size_t>(g) + a] = lo[a];
        aabb[6 * static_cast<size_t>(g) + 3 + a] = hi[a];
        const int centre = to_ordered(0.5f * (lo[a] + hi[a]));
        atomicMin(&world_bounds[6 * world + a], centre);
        atomicMax(&world_bounds[6 * world + 3 + a], centre);
    }
}

// One thread per instance: its sort key, the world in the top bits and the Morton code of its centre below.
__global__ void make_keys(const float *aabb, const int *world_bounds, const int per_world, const int total,
    uint64_t *keys, int *ids) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const int world = g / per_world;
    uint32_t q[3];
    for (int a = 0; a < 3; ++a) {
        const float lo = from_ordered(world_bounds[6 * world + a]), hi = from_ordered(world_bounds[6 * world + 3 + a]);
        const float centre = 0.5f * (aabb[6 * static_cast<size_t>(g) + a] + aabb[6 * static_cast<size_t>(g) + 3 + a]);
        const float extent = hi - lo;
        const float t = extent > 1e-20f ? (centre - lo) / extent : 0.0f;
        q[a] = static_cast<uint32_t>(fminf(fmaxf(t, 0.0f), 1.0f) * 1023.0f);
    }
    const uint32_t morton = (spread_bits(q[0]) << 2) | (spread_bits(q[1]) << 1) | spread_bits(q[2]);
    keys[g] = (static_cast<uint64_t>(world) << 32) | morton;
    ids[g] = g;
}

// The number of leading bits two keys share; keys that are equal are told apart by their positions, so
// every pair of leaves differs. -1 for a position outside the array.
__device__ __forceinline__ int common_prefix(const uint64_t *keys, const int n, const int i, const int j) {
    if (j < 0 || j >= n) return -1;
    const uint64_t a = keys[i], b = keys[j];
    if (a == b) return 64 + __clz(static_cast<uint32_t>(i ^ j));
    return __clzll(static_cast<long long>(a ^ b));
}

// One thread per internal node (n - 1 of them): works out which leaves it covers, where it splits, and names its two children.
// Internal nodes are numbered 0 to n-2, and leaf k is node n-1+k. Node 0 is the root of the whole sorted array.
__global__ void build_radix_tree(const uint64_t *keys, const int n, bvh::TlasNode *nodes, int *parent, int2 *range) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // Which way the node's range runs from i, and how far it reaches.
    const int d = common_prefix(keys, n, i, i + 1) > common_prefix(keys, n, i, i - 1) ? 1 : -1;
    const int min_prefix = common_prefix(keys, n, i, i - d);
    int max_length = 2;
    while (common_prefix(keys, n, i, i + max_length * d) > min_prefix) max_length *= 2;
    int length = 0;
    for (int t = max_length / 2; t >= 1; t /= 2)
        if (common_prefix(keys, n, i, i + (length + t) * d) > min_prefix) length += t;
    const int j = i + length * d;

    // Where the range splits in two: the last position that still shares more than the whole range does.
    const int node_prefix = common_prefix(keys, n, i, j);
    int s = 0;
    for (int t = (length + 1) / 2;; t = (t + 1) / 2) {
        if (common_prefix(keys, n, i, i + (s + t) * d) > node_prefix) s += t;
        if (t == 1) break;
    }
    const int gamma = i + s * d + (d < 0 ? d : 0);

    const int lo = min(i, j), hi = max(i, j);
    const int left = (lo == gamma) ? n - 1 + gamma : gamma;
    const int right = (hi == gamma + 1) ? n - 1 + gamma + 1 : gamma + 1;
    nodes[i].left = left;
    nodes[i].right = right;
    parent[left] = i;
    parent[right] = i;
    range[i] = make_int2(lo, hi);
}

// One thread per leaf (position k in the sorted order): the leaf's node, boxed as its instance is.
__global__ void make_leaves(const int *sorted_ids, const float *aabb, const int n, const int per_world,
    bvh::TlasNode *nodes, int *leaf_world) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int id = sorted_ids[k];
    bvh::TlasNode &leaf = nodes[n - 1 + k];
    for (int a = 0; a < 3; ++a) {
        leaf.lo[a] = aabb[6 * static_cast<size_t>(id) + a];
        leaf.hi[a] = aabb[6 * static_cast<size_t>(id) + 3 + a];
    }
    leaf.left = ~id;
    leaf.right = -1;
    leaf_world[k] = id / per_world;
}

// One thread per leaf, climbing: a node's box is the union of its children's, and the second of the two
// threads to reach a node is the one that knows both children are done, so it computes the box and climbs on.
__global__ void fit_boxes(bvh::TlasNode *nodes, const int *parent, int *arrivals, const int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int node = n - 1 + k;
    while (true) {
        const int p = parent[node];
        if (p < 0) return;
        __threadfence();  // this thread's box is visible before it is counted
        if (atomicAdd(&arrivals[p], 1) == 0) return;
        __threadfence();
        const int l = nodes[p].left, r = nodes[p].right;
        for (int a = 0; a < 3; ++a) {
            nodes[p].lo[a] = fminf(__ldcg(&nodes[l].lo[a]), __ldcg(&nodes[r].lo[a]));
            nodes[p].hi[a] = fmaxf(__ldcg(&nodes[l].hi[a]), __ldcg(&nodes[r].hi[a]));
        }
        node = p;
    }
}

// One thread per node: the node that roots a world's tree is the highest one whose leaves all belong to that world.
__global__ void find_roots(const int *parent, const int2 *range, const int *leaf_world, const int n, int *roots) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= 2 * n - 1) return;
    const int2 r = v < n - 1 ? range[v] : make_int2(v - (n - 1), v - (n - 1));
    const int world = leaf_world[r.x];
    if (leaf_world[r.y] != world) return;  // spans several worlds
    const int p = parent[v];
    if (p < 0) {
        roots[world] = v;
        return;
    }
    if (leaf_world[range[p].x] != leaf_world[range[p].y]) roots[world] = v;  // its parent spans several worlds
}

// Writes the instance records' inverses and the trees from instance matrices; see the file comment.
// Owns every buffer the build needs, sized once from the number of worlds and instances.
class Builder {
public:
    // local_bounds: 6 floats (the box's low corner, then its high corner) for each of the per_world
    // instances of a world, in the mesh's own space.
    Builder(const int worlds, const int per_world, const std::vector<float> &local_bounds)
        : worlds_(worlds), per_world_(per_world), total_(worlds * per_world) {
        const size_t n = static_cast<size_t>(total_);
        CUDA_CHECK(cudaMalloc(&d_local_bounds_, local_bounds.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_local_bounds_, local_bounds.data(), local_bounds.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_aabb_, 6 * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_world_bounds_, 6 * static_cast<size_t>(worlds_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_keys_in_, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_out_, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_ids_in_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ids_out_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&nodes, (2 * n - 1) * sizeof(bvh::TlasNode)));
        CUDA_CHECK(cudaMalloc(&roots, static_cast<size_t>(worlds_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_parent_, (2 * n - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_range_, n * sizeof(int2)));
        CUDA_CHECK(cudaMalloc(&d_arrivals_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_leaf_world_, n * sizeof(int)));

        // Keys have the world in bits 32 and up, and a 30-bit Morton code below.
        int world_bits = 1;
        while ((1 << world_bits) < worlds_) ++world_bits;
        end_bit_ = 32 + world_bits;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_, d_keys_in_, d_keys_out_, d_ids_in_, d_ids_out_,
            total_, 0, end_bit_));
        CUDA_CHECK(cudaMalloc(&d_temp_, temp_bytes_));
    }

    ~Builder() {
        cudaFree(d_local_bounds_);
        cudaFree(d_aabb_);
        cudaFree(d_world_bounds_);
        cudaFree(d_keys_in_);
        cudaFree(d_keys_out_);
        cudaFree(d_ids_in_);
        cudaFree(d_ids_out_);
        cudaFree(nodes);
        cudaFree(roots);
        cudaFree(d_parent_);
        cudaFree(d_range_);
        cudaFree(d_arrivals_);
        cudaFree(d_leaf_world_);
        cudaFree(d_temp_);
    }
    Builder(const Builder &) = delete;
    Builder &operator=(const Builder &) = delete;

    // matrices and instances are device arrays of total() entries. Queued on the default stream, not waited for.
    void build(const float *matrices, Instance *instances) {
        constexpr int T = 128;
        auto blocks = [](const int count) { return (count + T - 1) / T; };
        init_world_bounds<<<blocks(worlds_), T>>>(d_world_bounds_, worlds_);
        place_instances<<<blocks(total_), T>>>(matrices, d_local_bounds_, per_world_, total_, instances, d_aabb_, d_world_bounds_);
        make_keys<<<blocks(total_), T>>>(d_aabb_, d_world_bounds_, per_world_, total_, d_keys_in_, d_ids_in_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_, temp_bytes_, d_keys_in_, d_keys_out_, d_ids_in_, d_ids_out_,
            total_, 0, end_bit_));

        CUDA_CHECK(cudaMemsetAsync(d_parent_, 0xFF, (2 * static_cast<size_t>(total_) - 1) * sizeof(int)));  // -1: no parent
        CUDA_CHECK(cudaMemsetAsync(d_arrivals_, 0, static_cast<size_t>(total_) * sizeof(int)));
        make_leaves<<<blocks(total_), T>>>(d_ids_out_, d_aabb_, total_, per_world_, nodes, d_leaf_world_);
        if (total_ > 1) build_radix_tree<<<blocks(total_ - 1), T>>>(d_keys_out_, total_, nodes, d_parent_, d_range_);
        fit_boxes<<<blocks(total_), T>>>(nodes, d_parent_, d_arrivals_, total_);
        find_roots<<<blocks(2 * total_ - 1), T>>>(d_parent_, d_range_, d_leaf_world_, total_, roots);
        CUDA_CHECK(cudaGetLastError());
    }

    int total() const { return total_; }

    bvh::TlasNode *nodes = nullptr;  // 2 * total() - 1 of them
    int *roots = nullptr;            // per world

private:
    int worlds_;
    int per_world_;
    int total_;
    int end_bit_ = 0;
    size_t temp_bytes_ = 0;
    float *d_local_bounds_ = nullptr;
    float *d_aabb_ = nullptr;
    int *d_world_bounds_ = nullptr;
    uint64_t *d_keys_in_ = nullptr, *d_keys_out_ = nullptr;
    int *d_ids_in_ = nullptr, *d_ids_out_ = nullptr;
    int *d_parent_ = nullptr;
    int2 *d_range_ = nullptr;
    int *d_arrivals_ = nullptr;
    int *d_leaf_world_ = nullptr;
    void *d_temp_ = nullptr;
};

}  // namespace tlas
}  // namespace neraulgraphics::device

#endif  // NERAULGRAPHICS_CUDA_TLAS_CUH
