#ifndef NERAULGRAPHICS_MESH_GPU_CUH
#define NERAULGRAPHICS_MESH_GPU_CUH

// Ray queries against triangle meshes, on the device. A query walks two levels:
//
//   top level     a BVH over the instances' world-space boxes
//     instance    the ray is moved into the mesh's own space
//   bottom level  the mesh's BVH, built once and shared by all its instances
//     triangle    Moller-Trumbore
//
// Moving an instance only changes its matrix and the top level, never the bottom
// one. That is why thousands of moving bodies stay cheap.

#include "bvh.h"
#include "device_scene.cuh"

namespace neraulgraphics::device {

// Deeper than any tree the builder makes (bvh::MAX_DEPTH), because a traversal
// stack never holds more entries than the tree is deep.
constexpr int TRAVERSAL_STACK = bvh::MAX_DEPTH + 4;

constexpr float NO_HIT = 1e30f;

struct Instance {
    float4 inv[3];   // world -> mesh space, top three rows; only the inverse is needed on the device
    int blas_root;   // root of this instance's mesh in the shared node array
    Material material;
};

struct MeshScene {
    const bvh::Node *tlas;       // top level; root is node 0
    const int *tlas_instance;    // top-level leaf slot -> instance index
    const Instance *instances;
    const bvh::Node *blas;       // every mesh's nodes, concatenated
    const float4 *triangles;     // 3 float4 per triangle: v0, e1, e2
    int instance_count;
};

__device__ __forceinline__ float3 xyz3(const float4 &v) { return make_float3(v.x, v.y, v.z); }

// 1/x that never produces infinity, so a ray parallel to a slab gives a large
// finite number instead of 0 * inf = NaN in the box test.
__device__ __forceinline__ float3 safe_inverse(const float3 &d) {
    auto inv = [](const float x) { return 1.0f / (fabsf(x) < 1e-12f ? copysignf(1e-12f, x) : x); };
    return make_float3(inv(d.x), inv(d.y), inv(d.z));
}

// Slab test. Returns the distance where the ray enters the box, or NO_HIT if it
// misses or the box starts beyond tmax (the closest hit found so far). The box
// arrives as the two float4 halves of a Node: lo.xyz|a and hi.xyz|count.
__device__ __forceinline__ float box_entry(const float4 &lo, const float4 &hi, const float3 &o,
    const float3 &inv_d, const float tmax) {
    const float tx0 = (lo.x - o.x) * inv_d.x, tx1 = (hi.x - o.x) * inv_d.x;
    const float ty0 = (lo.y - o.y) * inv_d.y, ty1 = (hi.y - o.y) * inv_d.y;
    const float tz0 = (lo.z - o.z) * inv_d.z, tz1 = (hi.z - o.z) * inv_d.z;
    const float tn = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), 0.0f));
    const float tf = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), tmax));
    return tn <= tf ? tn : NO_HIT;
}

// Two-sided ray-triangle test (Moller-Trumbore). d need not be normalized; t is
// in units of d. Not watertight: a ray through an exact shared edge can slip
// between two triangles.
__device__ __forceinline__ bool triangle_hit(const float4 *tris, const int i, const float3 &o,
    const float3 &d, const float tmax, float &t) {
    const float3 v0 = xyz3(tris[3 * i]);
    const float3 e1 = xyz3(tris[3 * i + 1]);
    const float3 e2 = xyz3(tris[3 * i + 2]);
    const float3 p = cross(d, e2);
    const float det = dot(e1, p);
    if (fabsf(det) < 1e-14f) return false;  // parallel to the triangle, or degenerate
    const float inv_det = 1.0f / det;
    const float3 s = o - v0;
    const float u = dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const float3 q = cross(s, e1);
    const float v = dot(d, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = dot(e2, q) * inv_det;
    return t > 0.0f && t < tmax;
}

// Walks one BVH. For each leaf the ray reaches, calls leaf(first, count, tmax);
// the callback tests that leaf's primitives and lowers tmax when it finds a
// closer hit, which in turn lets this walk skip boxes that are now too far.
// Returns true if a callback asked to stop (any-hit queries).
//
// The nearer child is visited first and the farther one is pushed, with its entry
// distance so it can be dropped if a closer hit is found before it is popped.
template <class LeafFn>
__device__ bool traverse(const bvh::Node *nodes, const int root, const float3 &o, const float3 &inv_d,
    float &tmax, LeafFn &&leaf) {
    const float4 *raw = reinterpret_cast<const float4 *>(nodes);  // node i = raw[2i], raw[2i+1]
    if (box_entry(raw[2 * root], raw[2 * root + 1], o, inv_d, tmax) >= NO_HIT) return false;

    int stack[TRAVERSAL_STACK];
    float stack_t[TRAVERSAL_STACK];
    int sp = 0;
    int node = root;
    while (true) {
        const float4 n0 = raw[2 * node];
        const float4 n1 = raw[2 * node + 1];
        const int count = __float_as_int(n1.w);
        const int a = __float_as_int(n0.w);

        bool descended = false;
        if (count > 0) {
            if (leaf(a, count, tmax)) return true;
        } else {
            const int right = a + 1;
            const float dl = box_entry(raw[2 * a], raw[2 * a + 1], o, inv_d, tmax);
            const float dr = box_entry(raw[2 * right], raw[2 * right + 1], o, inv_d, tmax);
            if (dl < NO_HIT || dr < NO_HIT) {
                const bool left_first = dl <= dr;
                const int near_node = left_first ? a : right;
                const int far_node = left_first ? right : a;
                const float far_t = left_first ? dr : dl;
                if (far_t < NO_HIT) {
                    stack[sp] = far_node;
                    stack_t[sp] = far_t;
                    ++sp;
                }
                node = near_node;
                descended = true;
            }
        }
        if (descended) continue;

        // Dead end: resume from the stack, skipping anything a later hit has put out of reach.
        while (sp > 0 && stack_t[sp - 1] >= tmax) --sp;
        if (sp == 0) return false;
        node = stack[--sp];
    }
}

// Moves a world-space ray into an instance's mesh space. The direction is not
// renormalized, which keeps t identical in both spaces: the point at distance t
// is the same point either way, so no conversion is needed when comparing hits.
__device__ __forceinline__ float3 to_local_point(const Instance &inst, const float3 &p) {
    return make_float3(dot(xyz3(inst.inv[0]), p) + inst.inv[0].w, dot(xyz3(inst.inv[1]), p) + inst.inv[1].w,
        dot(xyz3(inst.inv[2]), p) + inst.inv[2].w);
}

__device__ __forceinline__ float3 to_local_dir(const Instance &inst, const float3 &d) {
    return make_float3(dot(xyz3(inst.inv[0]), d), dot(xyz3(inst.inv[1]), d), dot(xyz3(inst.inv[2]), d));
}

// Closest hit. tmax is the distance to beat on entry (the sphere or floor hit found
// already) and the hit distance on a true return. Writes which instance and which
// triangle (an index into the global triangle array).
__device__ bool mesh_intersect(const MeshScene &m, const float3 &o, const float3 &d, float &tmax,
    int &hit_instance, int &hit_triangle) {
    bool found = false;
    traverse(m.tlas, 0, o, safe_inverse(d), tmax, [&](const int first, const int count, float &top_tmax) {
        for (int k = 0; k < count; ++k) {
            const int id = m.tlas_instance[first + k];
            const Instance &inst = m.instances[id];
            const float3 lo = to_local_point(inst, o);
            const float3 ld = to_local_dir(inst, d);
            traverse(m.blas, inst.blas_root, lo, safe_inverse(ld), top_tmax,
                [&](const int tri_first, const int tri_count, float &t_best) {
                    for (int j = tri_first; j < tri_first + tri_count; ++j) {
                        float t;
                        if (triangle_hit(m.triangles, j, lo, ld, t_best, t)) {
                            t_best = t;
                            hit_instance = id;
                            hit_triangle = j;
                            found = true;
                        }
                    }
                    return false;
                });
        }
        return false;
    });
    return found;
}

// Any hit closer than max_dist. Stops at the first one, which is all a shadow ray needs.
__device__ bool mesh_occluded(const MeshScene &m, const float3 &o, const float3 &d, const float max_dist) {
    float tmax = max_dist;
    return traverse(m.tlas, 0, o, safe_inverse(d), tmax, [&](const int first, const int count, float &top_tmax) {
        for (int k = 0; k < count; ++k) {
            const Instance &inst = m.instances[m.tlas_instance[first + k]];
            const float3 lo = to_local_point(inst, o);
            const float3 ld = to_local_dir(inst, d);
            const bool blocked = traverse(m.blas, inst.blas_root, lo, safe_inverse(ld), top_tmax,
                [&](const int tri_first, const int tri_count, float &t_best) {
                    for (int j = tri_first; j < tri_first + tri_count; ++j) {
                        float t;
                        if (triangle_hit(m.triangles, j, lo, ld, t_best, t)) return true;
                    }
                    return false;
                });
            if (blocked) return true;
        }
        return false;
    });
}

// The surface normal at a triangle hit, facing the incoming ray (triangles are two-sided).
// Normals are covectors: they transform by the inverse transpose, which for the stored
// inverse M^-1 means taking its rows as the basis, not its columns.
__device__ float3 mesh_normal(const MeshScene &m, const int instance, const int triangle, const float3 &dir) {
    const Instance &inst = m.instances[instance];
    const float3 nl = cross(xyz3(m.triangles[3 * triangle + 1]), xyz3(m.triangles[3 * triangle + 2]));
    const float3 n = normalize(xyz3(inst.inv[0]) * nl.x + xyz3(inst.inv[1]) * nl.y + xyz3(inst.inv[2]) * nl.z);
    return dot(n, dir) > 0.0f ? -n : n;
}

}  // namespace neraulgraphics::device

#endif  // NERAULGRAPHICS_MESH_GPU_CUH
