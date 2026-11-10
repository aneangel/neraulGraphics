#include "instances.h"

namespace neraulgraphics::detail {

InstanceSet::InstanceSet(const BuiltMeshes &built, const std::vector<Instance> &instances)
    : mesh_root_(built.roots), mesh_bounds_(built.bounds) {
    std::vector<Transform> transforms;
    transforms.reserve(instances.size());
    records.resize(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        mesh_.push_back(instances[i].mesh);
        records[i].texture = instances[i].texture;
        records[i].material = instances[i].material;
        transforms.push_back(instances[i].transform);
    }
    update(transforms);
}

bool InstanceSet::update(const std::vector<Transform> &transforms) {
    const size_t n = transforms.size();
    std::vector<bvh::Aabb> boxes(n);
    for (size_t i = 0; i < n; ++i) {
        Transform inverse;
        invert_affine(transforms[i], inverse);  // checked invertible before this point
        InstanceRecord &r = records[i];
        for (int k = 0; k < 12; ++k) r.inv[k] = inverse.m[k];
        r.blas_root = mesh_root_[mesh_[i]];
        boxes[i] = transform_aabb(transforms[i], mesh_bounds_[mesh_[i]]);
    }

    // Refit is one cheap pass; a full build searches for good splits. Refit while the tree is
    // still good, and build again once the boxes have loosened too far. 1.3x the cost it had
    // when built is where traces start to visibly slow.
    constexpr float REBUILD_ABOVE = 1.3f;
    bool rebuilt = tlas.nodes.empty();
    if (!rebuilt) {
        bvh::refit(tlas, boxes);
        rebuilt = bvh::sah_cost(tlas) > REBUILD_ABOVE * built_cost_;
    }
    if (rebuilt) {
        tlas = bvh::build(boxes, 1);
        built_cost_ = bvh::sah_cost(tlas);
    }
    return rebuilt;
}

}  // namespace neraulgraphics::detail
