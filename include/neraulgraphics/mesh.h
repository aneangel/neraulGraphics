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
    // Optional, one per vertex or empty. With normals, shading is smooth: the normal is
    // interpolated across each triangle instead of being the triangle's own. They need not
    // be unit length. With uvs, an Instance's texture can be mapped on.
    std::vector<Vec3> normals;
    // Texture coordinates with the origin at the image's top left and v pointing down,
    // as glTF has them (an OBJ's v is flipped by the loader). Outside 0..1 they repeat.
    std::vector<Vec2> uvs;
};

// An RGBA image, 8 bits per channel, rows from top to bottom, colour sRGB-encoded as
// it is in a PNG or JPEG. The renderer converts to linear light when it samples, and
// filters bilinearly. Alpha is stored but not used.
struct Texture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;  // width * height * 4 bytes
    // Data, not colour: a normal map or a roughness map. The bytes are used as they are (0..255 is 0..1), without the sRGB decode.
    bool linear = false;
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
    // Index into the textures the Renderer was given, or -1 for none. The texture colour is
    // multiplied into material.diffuse_color, so a textured material usually wants a white
    // diffuse_color. The mesh must have uvs.
    int texture = -1;
    // A tangent-space normal map (a texture made with linear = true): red is +u, green is up the image, blue is out of the surface,
    // each 0..255 for -1..1. The mesh needs uvs. Fine detail in the shading without fine geometry.
    int normal_texture = -1;
    // Occlusion, roughness and metal in the red, green and blue of a linear texture, as glTF packs them. Roughness and metal
    // multiply the material's values (or stand in for them when the material derives them); occlusion darkens the diffuse colour.
    // Path tracing uses roughness and metal; the classic renderer ignores them. The mesh needs uvs.
    int orm_texture = -1;
};

// An axis-aligned box centred on the origin, 12 triangles.
Mesh make_box(const Vec3 &half_extents);

// A sphere centred on the origin, built by subdividing an icosahedron:
// 20 * 4^subdivisions triangles. 3 gives 1280, a smooth-looking ball.
Mesh make_icosphere(float radius, int subdivisions);

}  // namespace neraulgraphics

#endif  // NERAULGRAPHICS_MESH_H
