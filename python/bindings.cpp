// Python bindings: the neraulgraphics._core extension module.
//
// Vectors cross the boundary as plain 3-sequences (tuples, lists, numpy arrays),
// pixels come back as a numpy array, and the library's exceptions become Python
// ones: std::invalid_argument is ValueError, CudaError is neraulgraphics.CudaError.

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
        const std::vector<Light> &lights, const Plane &floor)
        : renderer(width, height, spheres, lights, floor) {}
};

using Image = nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::c_contig>;

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

    nb::exception<CudaError>(m, "CudaError", PyExc_RuntimeError);

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
        .def(nb::init<int, int, const std::vector<Sphere> &, const std::vector<Light> &, const Plane &>(),
            "width"_a, "height"_a, "spheres"_a, "lights"_a, "floor"_a = Plane{})
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
        }, "GPU time of the last render, in milliseconds.");
}
