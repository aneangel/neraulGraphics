#ifndef NERAULGRAPHICS_INSTANCES_H
#define NERAULGRAPHICS_INSTANCES_H

// The host-side state of a scene's instances, shared by both backends: what each instance
// needs to be traced (its inverse transform, its mesh's root, texture and material) and the
// top-level BVH over them. A backend converts the records to its own layout and uploads or
// keeps them; this decides what they are and when the tree must be rebuilt.

#include <vector>
#include <neraulgraphics/mesh.h>
#include "bvh.h"
#include "mesh_build.h"

namespace neraulgraphics::detail {

struct InstanceRecord {
    float inv[12];  // world -> mesh space, top three rows of the 4x4, row-major
    int blas_root;  // root of this instance's mesh in BuiltMeshes::nodes
    int texture;    // index into the textures, or -1
    Material material;
};

class InstanceSet {
public:
    // instances must be non-empty and have passed check_meshes.
    InstanceSet(const BuiltMeshes &built, const std::vector<Instance> &instances);

    // Moves the instances to new transforms (already checked invertible). The top-level BVH is
    // refitted in place, which is one cheap pass, and rebuilt from scratch only once its boxes
    // have loosened enough to slow tracing down. Returns true if it was rebuilt, which is when
    // the order of its leaves, tlas.order, changes.
    bool update(const std::vector<Transform> &transforms);

    size_t size() const { return mesh_.size(); }

    std::vector<InstanceRecord> records;
    bvh::Bvh tlas;

private:
    std::vector<int> mesh_;
    std::vector<int> mesh_root_;
    std::vector<bvh::Aabb> mesh_bounds_;
    float built_cost_ = 0.0f;  // bvh::sah_cost() right after the last full build
};

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_INSTANCES_H
