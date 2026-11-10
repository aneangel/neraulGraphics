#ifndef NERAULGRAPHICS_MESH_BUILD_H
#define NERAULGRAPHICS_MESH_BUILD_H

// Host-side preparation of meshes and instances for the renderer. No CUDA here:
// this produces the flat arrays render_gpu.cu uploads, and can be tested alone.

#include <string>
#include <vector>
#include <neraulgraphics/mesh.h>
#include "bvh.h"

namespace neraulgraphics::detail {

// Every mesh's bottom-level structure, concatenated. Child and primitive indices
// inside nodes are already global, so one array and one root per mesh is enough.
struct BuiltMeshes {
    std::vector<bvh::Node> nodes;
    // Triangles in BVH leaf order, 12 floats each: v0, e1 = v1 - v0, e2 = v2 - v0,
    // each padded to 4 so the kernel reads them as float4. This is the layout
    // Moller-Trumbore wants, and it avoids chasing an index buffer per hit.
    std::vector<float> triangles;
    // Per triangle, 16 floats, in the same order as `triangles`: three vertices' normals and
    // uvs as (nx, ny, nz, u) x 3 then (v0, v1, v2, flags). flags is an int stored in the last
    // float's bits: ATTR_NORMALS and ATTR_UVS say which of the two the triangle has. Empty if
    // no mesh has either, so a scene without them pays nothing.
    std::vector<float> attributes;
    std::vector<int> roots;           // per mesh: index of its root in nodes
    std::vector<bvh::Aabb> bounds;    // per mesh: bounds in the mesh's own space
};

constexpr int ATTR_NORMALS = 1;
constexpr int ATTR_UVS = 2;

// Throws std::invalid_argument naming the offending mesh, instance or index.
void check_meshes(const std::vector<Mesh> &meshes, const std::vector<Instance> &instances,
    const std::vector<Texture> &textures);
void check_transform(const Transform &t, const std::string &name);

// Whether t is finite and invertible, with no message built: for checking many transforms, where almost
// every one is fine and the explanation is only wanted for the one that is not.
bool transform_ok(const Transform &t);

// Meshes must already have passed check_meshes.
BuiltMeshes build_meshes(const std::vector<Mesh> &meshes);

// The inverse of an affine transform. Returns false if it is singular.
bool invert_affine(const Transform &t, Transform &out);

// Bounds of a box after a transform: the box around its eight moved corners.
bvh::Aabb transform_aabb(const Transform &t, const bvh::Aabb &box);

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_MESH_BUILD_H
