// Python bindings: the neraulgraphics._core extension module.
//
// Vectors cross the boundary as plain 3-sequences (tuples, lists, numpy arrays),
// pixels come back as a numpy array, and the library's exceptions become Python
// ones: std::invalid_argument is ValueError, CudaError is neraulgraphics.CudaError.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <neraulgraphics/renderer.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace neraulgraphics;

// Vec3 <-> any 3-sequence of numbers. Only stable-ABI calls, since the module is
// built against Python's limited API.
namespace nanobind::detail {

template <> struct type_caster<Vec3> {
    NB_TYPE_CASTER(Vec3, const_name("tuple[float, float, float]"))

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *o = src.ptr();
        // Strings are sequences too, just not of numbers.
        if (PyUnicode_Check(o) || PyBytes_Check(o) || !PySequence_Check(o)) return false;
        if (PySequence_Size(o) != 3) {
            PyErr_Clear();
            return false;
        }
        float v[3];
        for (Py_ssize_t i = 0; i < 3; ++i) {
            PyObject *item = PySequence_GetItem(o, i);
            if (!item) {
                PyErr_Clear();
                return false;
            }
            const double d = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (d == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return false;
            }
            v[i] = static_cast<float>(d);
        }
        value = Vec3{v[0], v[1], v[2]};
        return true;
    }

    static handle from_cpp(const Vec3 &v, rv_policy, cleanup_list *) noexcept {
        return Py_BuildValue("(ddd)", static_cast<double>(v.x), static_cast<double>(v.y),
            static_cast<double>(v.z));
    }
};

}  // namespace nanobind::detail

namespace {

// The C++ Renderer is not safe to call from two threads at once. render() lets go
// of the GIL while the GPU works, so the Python object carries a lock of its own.
struct PyRenderer {
    Renderer renderer;
    std::mutex mutex;

    PyRenderer(const int width, const int height, const std::vector<Sphere> &spheres,
        const std::vector<Light> &lights, const Plane &floor, const std::vector<Mesh> &meshes,
        const std::vector<Instance> &instances, const std::vector<Texture> &textures, const Device device)
        : renderer(width, height, spheres, lights, floor, meshes, instances, textures, device) {}
};

Device parse_device(const std::string &name) {
    if (name == "auto") return Device::Auto;
    if (name == "cuda") return Device::Cuda;
    if (name == "cpu") return Device::Cpu;
    throw std::invalid_argument("device must be 'auto', 'cuda' or 'cpu', got '" + name + "'");
}

// Arrays arrive as C-contiguous float32 or uint32 (anything else is converted),
// which is what lets the C++ side read them with no per-element Python calls.
using Vertices = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Triangles = nb::ndarray<const uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrices = nb::ndarray<const float, nb::shape<-1, 3, 4>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const float, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;

using Uvs = nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using Pixels = nb::ndarray<const uint8_t, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cpu>;

Mesh make_mesh(const Vertices &vertices, const Triangles &triangles, const std::optional<Vertices> &normals,
    const std::optional<Uvs> &uvs) {
    Mesh mesh;
    mesh.vertices.resize(vertices.shape(0));
    std::memcpy(mesh.vertices.data(), vertices.data(), vertices.shape(0) * 3 * sizeof(float));
    mesh.indices.resize(triangles.shape(0) * 3);
    std::memcpy(mesh.indices.data(), triangles.data(), triangles.shape(0) * 3 * sizeof(uint32_t));
    if (normals) {
        mesh.normals.resize(normals->shape(0));
        std::memcpy(mesh.normals.data(), normals->data(), normals->shape(0) * 3 * sizeof(float));
    }
    if (uvs) {
        mesh.uvs.resize(uvs->shape(0));
        std::memcpy(mesh.uvs.data(), uvs->data(), uvs->shape(0) * 2 * sizeof(float));
    }
    return mesh;
}

// A (height, width, channels) uint8 image with 1, 3 or 4 channels, widened to RGBA.
Texture make_texture(const Pixels &image) {
    const size_t h = image.shape(0), w = image.shape(1), c = image.shape(2);
    if (c != 1 && c != 3 && c != 4)
        throw std::invalid_argument("a texture image needs 1 (grey), 3 (RGB) or 4 (RGBA) channels, got " +
            std::to_string(c));
    if (w == 0 || h == 0) throw std::invalid_argument("a texture image cannot be empty");
    if (w > 65536 || h > 65536) throw std::invalid_argument("a texture image is at most 65536 pixels on a side");
    Texture t;
    t.width = static_cast<int>(w);
    t.height = static_cast<int>(h);
    t.rgba.resize(w * h * 4);
    const uint8_t *src = image.data();
    for (size_t i = 0; i < w * h; ++i) {
        uint8_t *dst = &t.rgba[4 * i];
        dst[0] = src[c * i];
        dst[1] = c >= 3 ? src[c * i + 1] : src[c * i];
        dst[2] = c >= 3 ? src[c * i + 2] : src[c * i];
        dst[3] = c == 4 ? src[c * i + 3] : 255;
    }
    return t;
}

// A 3x4 or 4x4 matrix; the bottom row of a 4x4 is ignored.
Transform transform_from_matrix(const Matrix &m) {
    if (m.shape(0) != 3 && m.shape(0) != 4)
        throw std::invalid_argument("transform matrix must be 3x4 or 4x4, got " +
            std::to_string(m.shape(0)) + "x4");
    Transform t;
    std::memcpy(t.m, m.data(), 12 * sizeof(float));
    return t;
}

void set_instance_transforms(PyRenderer &self, const Matrices &matrices) {
    std::vector<Transform> transforms(matrices.shape(0));
    std::memcpy(transforms.data(), matrices.data(), matrices.shape(0) * 12 * sizeof(float));
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(self.mutex);
    self.renderer.set_instance_transforms(transforms);
}

void set_instance_transform_list(PyRenderer &self, const std::vector<Transform> &transforms) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(self.mutex);
    self.renderer.set_instance_transforms(transforms);
}

using Image = nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::c_contig>;

// An array owned by the caller from here on: a vector's data, freed when the array is.
template <class T>
nb::object own(std::vector<T> &&values, std::initializer_list<size_t> shape) {
    auto *keep = new std::vector<T>(std::move(values));
    nb::capsule owner(keep, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T>(keep->data(), shape.size(), shape.begin(), owner));
}

// A view of the renderer's own buffer, kept alive by holding the renderer: a NumPy array for the CPU, and for a
// GPU an object that speaks DLPack (torch.from_dlpack, jax.dlpack.from_dlpack, wp.from_dlpack, cupy.from_dlpack).
// It is overwritten by the next render or ray query, so the caller must use or copy it first.
template <class T>
nb::object view(nb::handle owner, const T *data, std::initializer_list<size_t> shape, const Memory memory, const int device_id) {
    T *mutable_data = const_cast<T *>(data);
    if (memory == Memory::Device)
        return nb::cast(nb::ndarray<T, nb::device::cuda>(mutable_data, shape.size(), shape.begin(), owner, nullptr,
            nb::dtype<T>(), nb::device::cuda::value, device_id));
    return nb::cast(nb::ndarray<nb::numpy, T>(mutable_data, shape.size(), shape.begin(), owner));
}

// render_frame: which images to make, as keyword arguments, and where the result lives.
nb::dict render_frame(nb::handle self_object, PyRenderer &self, const Camera &camera, const int frame,
    const int samples_per_frame, const bool rgb, const bool depth, const bool normals, const bool object_ids,
    const bool zero_copy) {
    const unsigned outputs = (rgb ? OUTPUT_RGB : 0u) | (depth ? OUTPUT_DEPTH : 0u) | (normals ? OUTPUT_NORMAL : 0u) |
        (object_ids ? OUTPUT_OBJECT_ID : 0u);
    nb::dict result;
    if (!zero_copy) {
        HostFrame f;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(self.mutex);
            f = self.renderer.render_frame_host(camera, frame, samples_per_frame, outputs);
        }
        const size_t h = static_cast<size_t>(f.height), w = static_cast<size_t>(f.width);
        if (rgb) result["rgb"] = own(std::move(f.rgb), {h, w, 3});
        if (depth) result["depth"] = own(std::move(f.depth), {h, w});
        if (normals) result["normals"] = own(std::move(f.normal), {h, w, 3});
        if (object_ids) result["object_ids"] = own(std::move(f.object_id), {h, w});
        return result;
    }
    Frame f;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(self.mutex);
        f = self.renderer.render_frame(camera, frame, samples_per_frame, outputs);
    }
    const size_t h = static_cast<size_t>(f.height), w = static_cast<size_t>(f.width);
    if (rgb) result["rgb"] = view(self_object, f.rgb, {h, w, 3}, f.memory, f.device_id);
    if (depth) result["depth"] = view(self_object, f.depth, {h, w}, f.memory, f.device_id);
    if (normals) result["normals"] = view(self_object, f.normal, {h, w, 3}, f.memory, f.device_id);
    if (object_ids) result["object_ids"] = view(self_object, f.object_id, {h, w}, f.memory, f.device_id);
    return result;
}

using Rays = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using DeviceRays = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>;

nb::dict rays_to_dict(nb::handle self_object, const RayHits &h) {
    nb::dict result;
    const size_t n = static_cast<size_t>(h.count);
    result["distance"] = view(self_object, h.distance, {n}, h.memory, h.device_id);
    result["normal"] = view(self_object, h.normal, {n, 3}, h.memory, h.device_id);
    result["object_id"] = view(self_object, h.object_id, {n}, h.memory, h.device_id);
    return result;
}

// A copy, not a view of the renderer's buffer: the next render() overwrites that,
// which would silently change an image the caller still holds.
Image render(PyRenderer &self, const Camera &camera, const int frame, const int samples_per_frame) {
    const size_t h = static_cast<size_t>(self.renderer.height());
    const size_t w = static_cast<size_t>(self.renderer.width());
    auto *rgb = new uint8_t[h * w * 3];
    // Owns rgb from here on, so it is freed even if render() throws.
    nb::capsule owner(rgb, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(self.mutex);
        const uint32_t *argb = self.renderer.render(camera, frame, samples_per_frame);
        for (size_t i = 0; i < h * w; ++i) {
            rgb[3 * i + 0] = static_cast<uint8_t>(argb[i] >> 16);
            rgb[3 * i + 1] = static_cast<uint8_t>(argb[i] >> 8);
            rgb[3 * i + 2] = static_cast<uint8_t>(argb[i]);
        }
    }
    return Image(rgb, {h, w, 3}, owner);
}

std::string fmt(const Vec3 &v) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(%g, %g, %g)", v.x, v.y, v.z);
    return buf;
}

std::string fmt(const float f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", f);
    return buf;
}

std::string repr(const Material &m) {
    return "Material(diffuse_color=" + fmt(m.diffuse_color) + ", diffuse_albedo=" +
        fmt(m.diffuse_albedo) + ", specular_albedo=" + fmt(m.specular_albedo) +
        ", reflectivity=" + fmt(m.reflectivity) + ", specular_exponent=" +
        fmt(m.specular_exponent) + ")";
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "GPU raytracer. Build a scene, make a Renderer, call render().";
    m.attr("__version__") = NERAULGRAPHICS_VERSION;
    m.attr("MAX_SPHERES") = MAX_SPHERES;
    m.attr("MAX_LIGHTS") = MAX_LIGHTS;
    // What an object-ID image or a ray query reports.
    m.attr("OBJECT_NONE") = OBJECT_NONE;
    m.attr("OBJECT_SPHERE") = OBJECT_SPHERE;
    m.attr("OBJECT_FLOOR") = OBJECT_FLOOR;
    m.attr("OBJECT_INSTANCE") = OBJECT_INSTANCE;

    nb::exception<CudaError>(m, "CudaError", PyExc_RuntimeError);

    m.def("cuda_available", &cuda_available,
        "True if a GPU is usable, so device='auto' would use it. The environment variable "
        "NERAULGRAPHICS_DEVICE=cpu (or cuda) overrides what 'auto' chooses.");

    nb::class_<Material>(m, "Material", "How a surface reflects light.")
        .def("__init__",
            [](Material *self, const Vec3 &diffuse_color, const float diffuse_albedo,
                const float specular_albedo, const float reflectivity,
                const float specular_exponent) {
                new (self) Material{diffuse_color, diffuse_albedo, specular_albedo, reflectivity,
                    specular_exponent};
            },
            "diffuse_color"_a = Vec3{0.8f, 0.8f, 0.8f}, "diffuse_albedo"_a = 0.9f,
            "specular_albedo"_a = 0.1f, "reflectivity"_a = 0.0f, "specular_exponent"_a = 50.0f,
            "reflectivity: 0 is matte, 1 is a perfect mirror. specular_exponent: higher gives "
            "a smaller, sharper highlight.")
        .def_rw("diffuse_color", &Material::diffuse_color)
        .def_rw("diffuse_albedo", &Material::diffuse_albedo)
        .def_rw("specular_albedo", &Material::specular_albedo)
        .def_rw("reflectivity", &Material::reflectivity)
        .def_rw("specular_exponent", &Material::specular_exponent)
        .def("__repr__", &repr);

    nb::class_<Sphere>(m, "Sphere")
        .def("__init__",
            [](Sphere *self, const Vec3 &center, const float radius, const Material &material) {
                new (self) Sphere{center, radius, material};
            },
            "center"_a, "radius"_a, "material"_a = Material{{0.8f, 0.8f, 0.8f}, 0.9f, 0.1f, 0.0f, 50.0f})
        .def_rw("center", &Sphere::center)
        .def_rw("radius", &Sphere::radius)
        .def_rw("material", &Sphere::material)
        .def("__repr__", [](const Sphere &s) {
            return "Sphere(center=" + fmt(s.center) + ", radius=" + fmt(s.radius) + ", material=" +
                repr(s.material) + ")";
        });

    nb::class_<Light>(m, "Light", "A spherical light. radius 0 gives hard shadows, above 0 soft ones.")
        .def("__init__",
            [](Light *self, const Vec3 &position, const float intensity, const float radius) {
                new (self) Light{position, intensity, radius};
            },
            "position"_a, "intensity"_a = 1.0f, "radius"_a = 0.0f)
        .def_rw("position", &Light::position)
        .def_rw("intensity", &Light::intensity)
        .def_rw("radius", &Light::radius)
        .def("__repr__", [](const Light &l) {
            return "Light(position=" + fmt(l.position) + ", intensity=" + fmt(l.intensity) +
                ", radius=" + fmt(l.radius) + ")";
        });

    nb::class_<Plane>(m, "Plane",
        "A finite checkerboard floor at height y, spanning |x| < half_extent_x and "
        "z_min < z < z_max. Plane() is no floor at all.")
        .def("__init__",
            [](Plane *self, const float y, const float half_extent_x, const float z_min,
                const float z_max, const float checker_scale, const Material &material) {
                new (self) Plane{y, half_extent_x, z_min, z_max, checker_scale, material};
            },
            "y"_a = 0.0f, "half_extent_x"_a = 0.0f, "z_min"_a = 0.0f, "z_max"_a = 0.0f,
            "checker_scale"_a = 0.5f, "material"_a = Material{{1.0f, 1.0f, 1.0f}, 1.0f, 0.0f, 0.0f, 1.0f})
        .def_rw("y", &Plane::y)
        .def_rw("half_extent_x", &Plane::half_extent_x)
        .def_rw("z_min", &Plane::z_min)
        .def_rw("z_max", &Plane::z_max)
        .def_rw("checker_scale", &Plane::checker_scale)
        .def_rw("material", &Plane::material)
        .def("__repr__", [](const Plane &p) {
            return "Plane(y=" + fmt(p.y) + ", half_extent_x=" + fmt(p.half_extent_x) + ", z_min=" +
                fmt(p.z_min) + ", z_max=" + fmt(p.z_max) + ", checker_scale=" +
                fmt(p.checker_scale) + ")";
        });

    nb::class_<Transform>(m, "Transform",
        "Where a mesh instance is in the world: an affine map as a 3x4 matrix. "
        "Make one with make_transform(), or from a 3x4 or 4x4 numpy matrix.")
        .def("__init__",
            [](Transform *self, const Matrix &matrix) { new (self) Transform(transform_from_matrix(matrix)); },
            "matrix"_a)
        .def_prop_ro("matrix", [](const Transform &t) {
            auto *data = new float[12];
            std::memcpy(data, t.m, sizeof(t.m));
            nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<float *>(p); });
            return nb::ndarray<nb::numpy, float, nb::shape<3, 4>>(data, {3, 4}, owner);
        // The array brings its own owner, so the property's default reference_internal policy would clash.
        }, nb::rv_policy::move, "The top three rows of the 4x4 matrix, as a (3, 4) float32 array.")
        .def("__repr__", [](const Transform &t) {
            return "Transform(position=" + fmt(Vec3{t.m[3], t.m[7], t.m[11]}) + ", ...)";
        });

    m.def("identity_transform", &identity_transform, "The transform that leaves a mesh where it is.");
    m.def("make_transform",
        [](const Vec3 &position, const std::array<float, 4> &rotation, const float scale) {
            return make_transform(position, Quat{rotation[0], rotation[1], rotation[2], rotation[3]}, scale);
        },
        "position"_a = Vec3{0.0f, 0.0f, 0.0f}, "rotation"_a = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f},
        "scale"_a = 1.0f,
        "Scale, then rotate, then translate. rotation is a quaternion (x, y, z, w), the order "
        "Warp, MuJoCo and Newton use; it is normalized for you.");

    nb::class_<Mesh>(m, "Mesh",
        "Triangle geometry, shared by every instance that uses it. Its BVH is built once, "
        "when a Renderer is made.")
        .def("__init__",
            [](Mesh *self, const Vertices &vertices, const Triangles &triangles,
                const std::optional<Vertices> &normals, const std::optional<Uvs> &uvs) {
                new (self) Mesh(make_mesh(vertices, triangles, normals, uvs));
            },
            "vertices"_a, "triangles"_a, "normals"_a = nb::none(), "uvs"_a = nb::none(),
            "vertices: (N, 3) float array. triangles: (M, 3) integer array of vertex indices. "
            "Winding does not matter; triangles are two-sided.\n\n"
            "normals: optional (N, 3), one per vertex, for smooth shading. uvs: optional (N, 2) "
            "texture coordinates with the origin at the image's top left (v down, as in glTF), "
            "needed to put a Texture on the mesh.")
        .def_prop_ro("vertex_count", [](const Mesh &mesh) { return mesh.vertices.size(); })
        .def_prop_ro("has_normals", [](const Mesh &mesh) { return !mesh.normals.empty(); })
        .def_prop_ro("has_uvs", [](const Mesh &mesh) { return !mesh.uvs.empty(); })
        .def_prop_ro("triangle_count", [](const Mesh &mesh) { return mesh.indices.size() / 3; })
        .def("__repr__", [](const Mesh &mesh) {
            return "Mesh(" + std::to_string(mesh.vertices.size()) + " vertices, " +
                std::to_string(mesh.indices.size() / 3) + " triangles)";
        });

    nb::class_<Texture>(m, "Texture",
        "An image that instances can be painted with. Colour is sRGB, as in a PNG or JPEG; the "
        "renderer converts to linear light when it samples, and filters bilinearly. uvs outside "
        "0..1 repeat.")
        .def("__init__", [](Texture *self, const Pixels &image) { new (self) Texture(make_texture(image)); },
            "image"_a,
            "image: a (height, width, channels) uint8 array with 1 (grey), 3 (RGB) or 4 (RGBA) "
            "channels, rows from the top.")
        .def_ro("width", &Texture::width)
        .def_ro("height", &Texture::height)
        .def("__repr__", [](const Texture &t) {
            return "Texture(" + std::to_string(t.width) + "x" + std::to_string(t.height) + ")";
        });

    m.def("make_box", &make_box, "half_extents"_a, "A box centred on the origin, 12 triangles.");
    m.def("make_icosphere", &make_icosphere, "radius"_a, "subdivisions"_a = 3,
        "A sphere centred on the origin: 20 * 4**subdivisions triangles.");

    nb::class_<Instance>(m, "Instance",
        "One placement of a mesh: which mesh (an index into the Renderer's meshes), where, and "
        "what it is made of.")
        .def("__init__",
            [](Instance *self, const int mesh, const Transform &transform, const Material &material,
                const int texture) { new (self) Instance{mesh, transform, material, texture}; },
            "mesh"_a, "transform"_a = identity_transform(),
            "material"_a = Material{{0.8f, 0.8f, 0.8f}, 0.9f, 0.1f, 0.0f, 50.0f}, "texture"_a = -1,
            "texture: index into the Renderer's textures, or -1 for none. The texture colour is "
            "multiplied into the material's diffuse_color, so a textured material usually wants "
            "diffuse_color=(1, 1, 1). The mesh needs uvs.")
        .def_rw("mesh", &Instance::mesh)
        .def_rw("texture", &Instance::texture)
        .def_rw("transform", &Instance::transform)
        .def_rw("material", &Instance::material)
        .def("__repr__", [](const Instance &i) {
            return "Instance(mesh=" + std::to_string(i.mesh) + ", material=" + repr(i.material) +
                (i.texture >= 0 ? ", texture=" + std::to_string(i.texture) : "") + ")";
        });

    nb::class_<Camera>(m, "Camera", "Create with make_camera().")
        .def_ro("position", &Camera::position)
        .def_ro("right", &Camera::right)
        .def_ro("up", &Camera::up)
        .def_ro("forward", &Camera::forward)
        .def_ro("fov_scale", &Camera::fov_scale)
        .def_ro("aspect", &Camera::aspect)
        .def("__repr__", [](const Camera &c) {
            return "Camera(position=" + fmt(c.position) + ", forward=" + fmt(c.forward) + ")";
        });

    m.def("make_camera",
        [](const Vec3 &position, const Vec3 &target, const int width, const int height,
            const Vec3 &world_up, const float fov) {
            return make_camera(position, target, world_up, fov, width, height);
        },
        "position"_a, "target"_a, nb::kw_only(), "width"_a, "height"_a,
        "world_up"_a = Vec3{0.0f, 1.0f, 0.0f}, "fov"_a = 1.5707964f,
        "A camera at position looking at target. width and height give the aspect ratio; "
        "fov is the vertical field of view in radians.");

    nb::class_<PyRenderer>(m, "Renderer",
        "One scene and its frame buffers on the GPU. Safe to share between threads.")
        .def("__init__",
            [](PyRenderer *self, const int width, const int height, const std::vector<Sphere> &spheres,
                const std::vector<Light> &lights, const Plane &floor, const std::vector<Mesh> &meshes,
                const std::vector<Instance> &instances, const std::vector<Texture> &textures,
                const std::string &device) {
                new (self) PyRenderer(width, height, spheres, lights, floor, meshes, instances, textures,
                    parse_device(device));
            },
            "width"_a, "height"_a, "spheres"_a, "lights"_a, "floor"_a = Plane{},
            "meshes"_a = std::vector<Mesh>{}, "instances"_a = std::vector<Instance>{},
            "textures"_a = std::vector<Texture>{}, "device"_a = "auto",
            "device: where to trace. 'auto' (the default) uses an NVIDIA GPU if there is one and "
            "the CPU otherwise; 'cuda' requires a GPU and raises CudaError without one; 'cpu' "
            "always uses the CPU. The picture is the same either way.")
        .def_prop_ro("device", [](PyRenderer &self) { return self.renderer.device() == Device::Cuda ? "cuda" : "cpu"; },
            "'cuda' or 'cpu': where this renderer is tracing.")
        .def("set_instance_transforms", &set_instance_transforms, "transforms"_a,
            "Move the instances: an (N, 3, 4) float array, one 3x4 matrix per instance in the "
            "order they were given. Only the small BVH over instances is updated, so this is the "
            "cheap per-frame call for a simulation. Render with frame=0 afterwards.")
        .def("set_instance_transforms", &set_instance_transform_list, "transforms"_a,
            "The same, from a list of Transform objects.")
        .def_prop_ro("instance_count", [](PyRenderer &self) { return self.renderer.instance_count(); })
        .def_prop_rw("ambient",
            [](PyRenderer &self) {
                std::lock_guard<std::mutex> lock(self.mutex);
                return self.renderer.ambient();
            },
            [](PyRenderer &self, const float ambient) {
                std::lock_guard<std::mutex> lock(self.mutex);
                self.renderer.set_ambient(ambient);
            },
            "Light that reaches every surface regardless of the lights, as a fraction of its "
            "diffuse colour. 0 (the default) leaves surfaces facing away from every light black; "
            "0.1 to 0.3 keeps detail visible there. Render with frame=0 after changing it.")
        .def("render_frame",
            [](PyRenderer &self, const Camera &camera, const int frame, const int spp, const bool rgb,
                const bool depth, const bool normals, const bool object_ids, const bool zero_copy) {
                return render_frame(nb::find(&self), self, camera, frame, spp, rgb, depth, normals, object_ids, zero_copy);
            },
            "camera"_a, "frame"_a = 0, "samples_per_frame"_a = 1, nb::kw_only(), "rgb"_a = true, "depth"_a = false,
            "normals"_a = false, "object_ids"_a = false, "zero_copy"_a = false,
            "Render the images you ask for and return them as a dict: 'rgb' (H, W, 3) uint8, 'depth' "
            "(H, W) float32, 'normals' (H, W, 3) float32 and 'object_ids' (H, W) int32.\n\n"
            "depth is measured along the camera's view axis, as a depth camera reports it, and is "
            "+inf where nothing is hit. normals are world-space, facing the camera, and 0 where nothing is hit. "
            "object_ids say what was hit (OBJECT_NONE, OBJECT_SPHERE + i, OBJECT_FLOOR, OBJECT_INSTANCE + i). "
            "Depth, normals and ids are one exact ray through each pixel's middle, so object edges are not "
            "blurred into false depths; rgb is the picture render() draws. Asking for fewer images is faster: "
            "depth alone does no shading.\n\n"
            "By default the arrays are copies. With zero_copy=True they are views of the renderer's own "
            "buffers instead, with no copy at all: on a GPU renderer they are CUDA arrays you can hand to "
            "torch.from_dlpack, jax.dlpack.from_dlpack, warp.from_dlpack or cupy.from_dlpack, already "
            "finished and ready to read. They are overwritten by the next render, so use or clone them first.")
        .def("cast_rays",
            [](PyRenderer &self, const Rays &origins, const Rays &directions, const float max_distance) {
                if (origins.shape(0) != directions.shape(0))
                    throw std::invalid_argument("origins and directions must have the same number of rays");
                const int n = static_cast<int>(origins.shape(0));
                // Copied into vectors first, so the GIL can be released while the GPU works.
                std::vector<float> o(origins.data(), origins.data() + 3 * origins.shape(0));
                std::vector<float> d(directions.data(), directions.data() + 3 * directions.shape(0));
                HostRayHits hits;
                {
                    nb::gil_scoped_release release;
                    std::lock_guard<std::mutex> lock(self.mutex);
                    hits = self.renderer.cast_rays_host(o, d, max_distance);
                }
                const size_t count = static_cast<size_t>(n);
                nb::dict result;
                result["distance"] = own(std::move(hits.distance), {count});
                result["normal"] = own(std::move(hits.normal), {count, 3});
                result["object_id"] = own(std::move(hits.object_id), {count});
                return result;
            },
            "origins"_a, "directions"_a, "max_distance"_a = std::numeric_limits<float>::infinity(),
            "Cast rays into the scene: a LiDAR, a depth sensor, or a visibility query. origins and "
            "directions are (N, 3) float arrays; directions need not be unit length. Returns a dict of "
            "copies: 'distance' (N,) along the ray (+inf for a miss), 'normal' (N, 3) and 'object_id' (N,) "
            "(see render_frame). A hit farther than max_distance is a miss.\n\n"
            "Pass CUDA arrays instead (for example PyTorch tensors) and the rays are traced where they "
            "are, with no copy: the results come back as CUDA arrays too, valid until the next ray "
            "query, with the GIL released. That needs a renderer on a GPU.")
        .def("cast_rays",
            [](PyRenderer &self, const DeviceRays &origins, const DeviceRays &directions, const float max_distance) {
                if (origins.shape(0) != directions.shape(0))
                    throw std::invalid_argument("origins and directions must have the same number of rays");
                RayHits hits;
                {
                    nb::gil_scoped_release release;
                    std::lock_guard<std::mutex> lock(self.mutex);
                    hits = self.renderer.cast_rays(origins.data(), directions.data(), static_cast<int>(origins.shape(0)),
                        max_distance, Memory::Device);
                }
                return rays_to_dict(nb::find(&self), hits);
            },
            "origins"_a, "directions"_a, "max_distance"_a = std::numeric_limits<float>::infinity())
        .def("render", &render, "camera"_a, "frame"_a = 0, "samples_per_frame"_a = 1,
            "Trace one frame and return it as a (height, width, 3) uint8 RGB array.\n\n"
            "frame 0 starts a fresh image. Rendering again with frame 1, 2, ... while the "
            "camera holds still averages in more samples, so the image keeps getting "
            "smoother. samples_per_frame adds rays per pixel within one call.")
        .def_prop_ro("width", [](PyRenderer &self) { return self.renderer.width(); })
        .def_prop_ro("height", [](PyRenderer &self) { return self.renderer.height(); })
        .def_prop_ro("last_kernel_ms", [](PyRenderer &self) {
            std::lock_guard<std::mutex> lock(self.mutex);
            return self.renderer.last_kernel_ms();
        }, "Time of the last render, in milliseconds: GPU time on CUDA, wall-clock time on the CPU.");
}
