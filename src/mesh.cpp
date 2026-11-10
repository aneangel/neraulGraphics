#include <neraulgraphics/mesh.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace neraulgraphics {

Transform identity_transform() {
    return Transform{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};
}

Transform make_transform(const Vec3 &position, const Quat &rotation, const float scale) {
    const float norm = std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y +
        rotation.z * rotation.z + rotation.w * rotation.w);
    if (!(norm > 0.0f) || !std::isfinite(norm))
        throw std::invalid_argument("rotation quaternion must be finite and not zero");
    const float x = rotation.x / norm, y = rotation.y / norm, z = rotation.z / norm,
                w = rotation.w / norm;

    // The standard unit-quaternion to rotation-matrix expansion.
    return Transform{{
        scale * (1 - 2 * (y * y + z * z)), scale * (2 * (x * y - z * w)), scale * (2 * (x * z + y * w)), position.x,
        scale * (2 * (x * y + z * w)), scale * (1 - 2 * (x * x + z * z)), scale * (2 * (y * z - x * w)), position.y,
        scale * (2 * (x * z - y * w)), scale * (2 * (y * z + x * w)), scale * (1 - 2 * (x * x + y * y)), position.z,
    }};
}

Mesh make_box(const Vec3 &h) {
    Mesh mesh;
    for (int i = 0; i < 8; ++i)
        mesh.vertices.push_back(Vec3{(i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z});
    // Two triangles per face. Vertex i has x from bit 0, y from bit 1, z from bit 2.
    mesh.indices = {
        0, 2, 1, 1, 2, 3,  // -z
        4, 5, 6, 5, 7, 6,  // +z
        0, 1, 4, 1, 5, 4,  // -y
        2, 6, 3, 3, 6, 7,  // +y
        0, 4, 2, 2, 4, 6,  // -x
        1, 3, 5, 3, 7, 5,  // +x
    };
    return mesh;
}

Mesh make_icosphere(const float radius, const int subdivisions) {
    if (!(radius > 0.0f) || !std::isfinite(radius))
        throw std::invalid_argument("icosphere radius must be positive, got " + std::to_string(radius));
    // 20 * 4^n triangles: past 8 that is over a billion.
    if (subdivisions < 0 || subdivisions > 8)
        throw std::invalid_argument("icosphere subdivisions must be 0 to 8, got " +
            std::to_string(subdivisions));

    Mesh mesh;
    auto add = [&](const float x, const float y, const float z) {
        const float len = std::sqrt(x * x + y * y + z * z);
        mesh.vertices.push_back(Vec3{x / len, y / len, z / len});  // unit length; scaled at the end
        return static_cast<uint32_t>(mesh.vertices.size() - 1);
    };

    // The icosahedron: 12 vertices on three mutually perpendicular golden rectangles.
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    add(-1, t, 0); add(1, t, 0); add(-1, -t, 0); add(1, -t, 0);
    add(0, -1, t); add(0, 1, t); add(0, -1, -t); add(0, 1, -t);
    add(t, 0, -1); add(t, 0, 1); add(-t, 0, -1); add(-t, 0, 1);
    mesh.indices = {
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
    };

    for (int level = 0; level < subdivisions; ++level) {
        // Each edge is shared by two triangles; the cache makes both use one midpoint.
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        auto midpoint = [&](const uint32_t a, const uint32_t b) {
            const std::pair<uint32_t, uint32_t> key{std::min(a, b), std::max(a, b)};
            if (const auto it = midpoints.find(key); it != midpoints.end()) return it->second;
            const Vec3 va = mesh.vertices[a], vb = mesh.vertices[b];
            return midpoints[key] = add(va.x + vb.x, va.y + vb.y, va.z + vb.z);
        };
        std::vector<uint32_t> next;
        next.reserve(mesh.indices.size() * 4);
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            const uint32_t a = mesh.indices[i], b = mesh.indices[i + 1], c = mesh.indices[i + 2];
            const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        mesh.indices = std::move(next);
    }

    for (Vec3 &v : mesh.vertices) v = Vec3{v.x * radius, v.y * radius, v.z * radius};
    return mesh;
}

}  // namespace neraulgraphics
