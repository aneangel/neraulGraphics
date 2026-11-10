#include "mesh_build.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// Compiled without -ffast-math (only the CUDA sources get it), so std::isfinite
// here still sees NaN. The same reason the other input checks live in a library file.

namespace neraulgraphics::detail {
namespace {

void require(const bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument(message);
}

bool finite(const float v) { return std::isfinite(v); }

bool finite(const Vec3 &v) { return finite(v.x) && finite(v.y) && finite(v.z); }

// Triangle count that keeps every index the kernel forms (3 * tri + 2) inside an int.
constexpr size_t MAX_TRIANGLES = 100000000;

}  // namespace

void check_transform(const Transform &t, const std::string &name) {
    for (const float v : t.m) require(finite(v), name + " has a NaN or infinite value");
    Transform inverse;
    require(invert_affine(t, inverse), name + " is singular (it flattens space, so it cannot be inverted)");
}

void check_meshes(const std::vector<Mesh> &meshes, const std::vector<Instance> &instances,
    const std::vector<Texture> &textures) {
    for (size_t t = 0; t < textures.size(); ++t) {
        const Texture &tex = textures[t];
        const std::string name = "texture " + std::to_string(t);
        require(tex.width > 0 && tex.height > 0, name + " size must be positive, got " +
            std::to_string(tex.width) + "x" + std::to_string(tex.height));
        require(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) <= (size_t{1} << 28),
            name + " is too large");
        require(tex.rgba.size() == static_cast<size_t>(tex.width) * tex.height * 4,
            name + " has " + std::to_string(tex.rgba.size()) + " bytes, but " + std::to_string(tex.width) + "x" +
            std::to_string(tex.height) + " RGBA needs " + std::to_string(static_cast<size_t>(tex.width) * tex.height * 4));
    }
    for (size_t m = 0; m < meshes.size(); ++m) {
        const Mesh &mesh = meshes[m];
        const std::string name = "mesh " + std::to_string(m);
        require(!mesh.indices.empty(), name + " has no triangles");
        require(mesh.indices.size() % 3 == 0, name + " has " + std::to_string(mesh.indices.size()) +
            " indices, which is not a multiple of 3");
        require(mesh.indices.size() / 3 <= MAX_TRIANGLES, name + " has too many triangles");
        for (size_t i = 0; i < mesh.vertices.size(); ++i)
            require(finite(mesh.vertices[i]), name + " vertex " + std::to_string(i) + " has a NaN or infinite value");
        require(mesh.normals.empty() || mesh.normals.size() == mesh.vertices.size(),
            name + " has " + std::to_string(mesh.normals.size()) + " normals for " +
            std::to_string(mesh.vertices.size()) + " vertices (need one each, or none)");
        require(mesh.uvs.empty() || mesh.uvs.size() == mesh.vertices.size(),
            name + " has " + std::to_string(mesh.uvs.size()) + " uvs for " +
            std::to_string(mesh.vertices.size()) + " vertices (need one each, or none)");
        for (size_t i = 0; i < mesh.normals.size(); ++i)
            require(finite(mesh.normals[i]), name + " normal " + std::to_string(i) + " has a NaN or infinite value");
        for (size_t i = 0; i < mesh.uvs.size(); ++i)
            require(finite(mesh.uvs[i].u) && finite(mesh.uvs[i].v),
                name + " uv " + std::to_string(i) + " has a NaN or infinite value");
        for (size_t i = 0; i < mesh.indices.size(); ++i)
            require(mesh.indices[i] < mesh.vertices.size(), name + " index " + std::to_string(i) +
                " is " + std::to_string(mesh.indices[i]) + ", but there are only " +
                std::to_string(mesh.vertices.size()) + " vertices");
    }
    for (size_t i = 0; i < instances.size(); ++i) {
        const Instance &inst = instances[i];
        const std::string name = "instance " + std::to_string(i);
        require(inst.mesh >= 0 && static_cast<size_t>(inst.mesh) < meshes.size(),
            name + " uses mesh " + std::to_string(inst.mesh) + ", but there are " +
            std::to_string(meshes.size()) + " meshes");
        check_transform(inst.transform, name + " transform");
        require(inst.texture >= -1 && inst.texture < static_cast<int>(textures.size()),
            name + " uses texture " + std::to_string(inst.texture) + ", but there are " +
            std::to_string(textures.size()) + " textures");
        require(inst.texture < 0 || !meshes[inst.mesh].uvs.empty(),
            name + " has a texture, but its mesh " + std::to_string(inst.mesh) + " has no uvs");
        const Material &mat = inst.material;
        require(finite(mat.diffuse_color) && finite(mat.diffuse_albedo) && finite(mat.specular_albedo) &&
            finite(mat.reflectivity) && finite(mat.specular_exponent),
            name + " material has a NaN or infinite value");
    }
}

BuiltMeshes build_meshes(const std::vector<Mesh> &meshes) {
    BuiltMeshes out;
    bool any_attributes = false;
    for (const Mesh &mesh : meshes) any_attributes = any_attributes || !mesh.normals.empty() || !mesh.uvs.empty();
    for (const Mesh &mesh : meshes) {
        const size_t tri_count = mesh.indices.size() / 3;

        std::vector<bvh::Aabb> tri_bounds(tri_count);
        for (size_t t = 0; t < tri_count; ++t) {
            bvh::Aabb box = bvh::empty_aabb();
            for (int k = 0; k < 3; ++k) {
                const Vec3 &v = mesh.vertices[mesh.indices[3 * t + k]];
                const float p[3] = {v.x, v.y, v.z};
                for (int a = 0; a < 3; ++a) {
                    box.lo[a] = std::fmin(box.lo[a], p[a]);
                    box.hi[a] = std::fmax(box.hi[a], p[a]);
                }
            }
            tri_bounds[t] = box;
        }

        // Four triangles per leaf: enough to amortise a node visit, few enough to stay cheap.
        const bvh::Bvh tree = bvh::build(tri_bounds, 4);

        const int node_base = static_cast<int>(out.nodes.size());
        const int tri_base = static_cast<int>(out.triangles.size() / 12);
        out.roots.push_back(node_base);
        out.bounds.push_back(bvh::Aabb{{tree.nodes[0].lo[0], tree.nodes[0].lo[1], tree.nodes[0].lo[2]},
            {tree.nodes[0].hi[0], tree.nodes[0].hi[1], tree.nodes[0].hi[2]}});

        // Leaves index triangles, interior nodes index nodes. Both become global.
        for (bvh::Node n : tree.nodes) {
            n.a += n.count > 0 ? tri_base : node_base;
            out.nodes.push_back(n);
        }

        for (const uint32_t t : tree.order) {
            const Vec3 &a = mesh.vertices[mesh.indices[3 * t + 0]];
            const Vec3 &b = mesh.vertices[mesh.indices[3 * t + 1]];
            const Vec3 &c = mesh.vertices[mesh.indices[3 * t + 2]];
            const float packed[12] = {a.x, a.y, a.z, 0.0f, b.x - a.x, b.y - a.y, b.z - a.z, 0.0f,
                c.x - a.x, c.y - a.y, c.z - a.z, 0.0f};
            out.triangles.insert(out.triangles.end(), packed, packed + 12);

            if (any_attributes) {
                const bool has_n = !mesh.normals.empty(), has_uv = !mesh.uvs.empty();
                float rec[16] = {};
                for (int k = 0; k < 3; ++k) {
                    const uint32_t vi = mesh.indices[3 * t + k];
                    if (has_n) {
                        rec[4 * k] = mesh.normals[vi].x;
                        rec[4 * k + 1] = mesh.normals[vi].y;
                        rec[4 * k + 2] = mesh.normals[vi].z;
                    }
                    if (has_uv) {
                        rec[4 * k + 3] = mesh.uvs[vi].u;
                        rec[12 + k] = mesh.uvs[vi].v;
                    }
                }
                const int32_t flags = (has_n ? ATTR_NORMALS : 0) | (has_uv ? ATTR_UVS : 0);
                std::memcpy(&rec[15], &flags, sizeof(flags));
                out.attributes.insert(out.attributes.end(), rec, rec + 16);
            }
        }
    }
    return out;
}

bool invert_affine(const Transform &t, Transform &out) {
    const float *m = t.m;
    // Cofactors of the 3x3 linear part.
    const float c00 = m[5] * m[10] - m[6] * m[9];
    const float c01 = m[6] * m[8] - m[4] * m[10];
    const float c02 = m[4] * m[9] - m[5] * m[8];
    const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    // Relative to the matrix's own size, so a small but valid scale is not rejected.
    float mag = 0.0f;
    for (const int i : {0, 1, 2, 4, 5, 6, 8, 9, 10}) mag = std::fmax(mag, std::fabs(m[i]));
    if (!std::isfinite(det) || std::fabs(det) <= 1e-12f * mag * mag * mag) return false;

    const float inv = 1.0f / det;
    const float r[9] = {
        c00 * inv, (m[2] * m[9] - m[1] * m[10]) * inv, (m[1] * m[6] - m[2] * m[5]) * inv,
        c01 * inv, (m[0] * m[10] - m[2] * m[8]) * inv, (m[2] * m[4] - m[0] * m[6]) * inv,
        c02 * inv, (m[1] * m[8] - m[0] * m[9]) * inv, (m[0] * m[5] - m[1] * m[4]) * inv,
    };
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) out.m[4 * row + col] = r[3 * row + col];
        // The inverse of x -> Ax + t is x -> A^-1 x - A^-1 t.
        out.m[4 * row + 3] = -(r[3 * row] * m[3] + r[3 * row + 1] * m[7] + r[3 * row + 2] * m[11]);
    }
    return true;
}

bvh::Aabb transform_aabb(const Transform &t, const bvh::Aabb &box) {
    bvh::Aabb out = bvh::empty_aabb();
    for (int corner = 0; corner < 8; ++corner) {
        const float p[3] = {(corner & 1) ? box.hi[0] : box.lo[0], (corner & 2) ? box.hi[1] : box.lo[1],
            (corner & 4) ? box.hi[2] : box.lo[2]};
        for (int a = 0; a < 3; ++a) {
            const float w = t.m[4 * a] * p[0] + t.m[4 * a + 1] * p[1] + t.m[4 * a + 2] * p[2] + t.m[4 * a + 3];
            out.lo[a] = std::fmin(out.lo[a], w);
            out.hi[a] = std::fmax(out.hi[a], w);
        }
    }
    return out;
}

}  // namespace neraulgraphics::detail
