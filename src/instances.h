#ifndef NERAULGRAPHICS_INSTANCES_H
#define NERAULGRAPHICS_INSTANCES_H

// The host-side state of a scene's instances, shared by both backends: what each instance
// needs to be traced (its inverse transform, its mesh's root, texture and material) and the
// top-level trees over them. A backend converts the records to its own layout and uploads or
// keeps them; this decides what they are and when a tree must be rebuilt.
//
// A scene may hold several worlds: independent copies of the same instances, each placed on its
// own, and each with its own tree. A ray cast in one world only ever sees that world's instances.

#include <vector>
#include <neraulgraphics/batch.h>
#include <neraulgraphics/mesh.h>
#include "bvh.h"
#include "mesh_build.h"

namespace neraulgraphics::detail {

struct InstanceRecord {
    float inv[12];  // world -> mesh space, top three rows of the 4x4, row-major
    int blas_root;  // root of this instance's mesh in BuiltMeshes::nodes
    int texture;    // index into the textures, or -1
    int normal_texture;  // a tangent-space normal map, or -1
    int orm_texture;     // occlusion, roughness and metal maps, or -1
    Material material;
};

class InstanceSet {
public:
    // instances (non-empty, and already through check_meshes) is what one world contains; every
    // world starts with those transforms. Instances are numbered world-major: world w's instance i
    // is w * instances.size() + i. With build_trees false only the records are kept up to date, for a
    // backend that builds the trees itself.
    InstanceSet(const BuiltMeshes &built, const std::vector<Instance> &instances, int worlds = 1, bool build_trees = true);

    // Moves the instances to new transforms: 12 floats each (the top three rows of the 4x4,
    // row-major), world-major, already checked invertible. Each world's tree is refitted in place,
    // which is one cheap pass, and rebuilt from scratch only once its boxes have loosened enough to
    // slow tracing down.
    void update(const float *matrices);
    void update(const std::vector<Transform> &transforms) { update(transforms.empty() ? nullptr : transforms.front().m); }

    int worlds() const { return worlds_; }
    size_t per_world() const { return per_world_; }
    size_t size() const { return per_world_ * static_cast<size_t>(worlds_); }

    std::vector<InstanceRecord> records;  // size() of them
    std::vector<bvh::TlasNode> nodes;     // every world's tree, in one array
    std::vector<int> roots;               // per world: the node that roots its tree

private:
    void flatten(int world, const bvh::Bvh &tree);
    int append_run(int world, const std::vector<uint32_t> &order, uint32_t first, uint32_t count);

    int worlds_;
    bool build_trees_;
    size_t per_world_;
    std::vector<int> mesh_;
    std::vector<int> mesh_root_;
    std::vector<bvh::Aabb> mesh_bounds_;
    std::vector<bvh::Aabb> boxes_;     // each instance's box in the world, from the last update
    std::vector<bvh::Bvh> trees_;      // per world, before flattening, kept so an update can refit them
    std::vector<float> built_cost_;    // per world: bvh::sah_cost() right after its last full build
};

// The matrix of every instance of every world from body poses: body_poses holds worlds *
// bodies_per_world poses of 7 floats (x, y, z, quaternion x, y, z, w), world-major, and out receives
// worlds * instance_count matrices of 12 floats. An instance's matrix is its body's pose, then its
// own offset, then its scale along the mesh's own axes; an instance with no body (-1) keeps its offset
// as its place in the world. Quaternions are normalized.
void body_poses_to_matrices(const BodyBinding &binding, int worlds, const float *body_poses, float *out);

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_INSTANCES_H
