#ifndef NERAULGRAPHICS_VALIDATE_H
#define NERAULGRAPHICS_VALIDATE_H

// Input checks shared by Renderer and BatchRenderer. They live in a compiled file rather than a header
// because main.cpp and the tests are built with -ffast-math, which lets the compiler assume NaN never
// occurs and delete std::isfinite; this one is built without it.

#include <string>
#include <vector>
#include <neraulgraphics/renderer.h>

namespace neraulgraphics::detail {

// Throws std::invalid_argument with the message unless ok.
void require(bool ok, const std::string &message);

bool finite(float v);
bool finite(const Vec3 &v);
bool finite(const Material &m);

// Throws std::invalid_argument saying what is wrong with a material, named `name` in the message.
void check_material(const Material &m, const std::string &name);

void check_frame_size(int width, int height);
void check_scene(const std::vector<Sphere> &spheres, const std::vector<Light> &lights, const Plane &floor);
void check_render_args(const Camera &camera, int frame, int samples_per_frame);

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_VALIDATE_H
