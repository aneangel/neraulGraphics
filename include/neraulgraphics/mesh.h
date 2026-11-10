#ifndef NERAULGRAPHICS_MESH_H
#define NERAULGRAPHICS_MESH_H

// Triangle geometry. A Mesh is built once and shared; an Instance places one in
// the world with its own Transform and Material. Plain data with no CUDA types,
// like scene.h.
//
// The split is what makes dynamic scenes cheap. A mesh's acceleration structure
// is built once, in the mesh's own space. Moving an instance changes only its
// Transform, so the renderer rebuilds just the small structure over instances.

#include <cstdint>
#include <vector>
#include <neraulgraphics/scene.h>

namespace neraulgraphics {

struct Mesh {
    std::vector<Vec3> vertices;
    // Three indices per triangle. Winding does not matter: triangles are two-sided.
    std::vector<uint32_t> indices;
};

// Rotation as a unit quaternion, vector part first (x, y, z, w), the order Warp,
// MuJoCo and Newton use for poses.
struct Quat {
    float x;
    float y;
    float z;
    float w;
};

// An affine map from a mesh's space to the world, as the top three rows of a 4x4
// matrix, row-major:  [ m0 m1 m2 | m3 ]  with the translation in m3, m7 and m11.
// Scale, rotation, shear and mirroring are all allowed; it must be invertible.
struct Transform {
    float m[12];
};

Transform identity_transform();

// Scale first, then rotate, then translate. rotation is normalized, so it need
// not be exactly unit length; a zero quaternion throws std::invalid_argument.
Transform make_transform(const Vec3 &position, const Quat &rotation = Quat{0.0f, 0.0f, 0.0f, 1.0f},
    float scale = 1.0f);

struct Instance {
    int mesh;  // index into the meshes the Renderer was given
    Transform transform;
    Material material;
};

// An axis-aligned box centred on the origin, 12 triangles.
Mesh make_box(const Vec3 &half_extents);

// A sphere centred on the origin, built by subdividing an icosahedron:
// 20 * 4^subdivisions triangles. 3 gives 1280, a smooth-looking ball.
Mesh make_icosphere(float radius, int subdivisions);

}  // namespace neraulgraphics

#endif  // NERAULGRAPHICS_MESH_H
