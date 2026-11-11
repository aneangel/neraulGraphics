#ifndef NERAULGRAPHICS_ENVIRONMENT_H
#define NERAULGRAPHICS_ENVIRONMENT_H

// What a backend needs of an Environment, worked out once on the host: the map as RGBA floats, and the tables
// that let a path tracer pick a direction on the sky in proportion to how bright it is there.

#include <vector>
#include <neraulgraphics/scene.h>

namespace neraulgraphics::detail {

struct EnvironmentTables {
    std::vector<float> rgba;         // map_width * map_height * 4 (alpha unused), empty if there is no map
    std::vector<float> marginal;     // map_height + 1: the running share of the brightness (weighted by the sphere's area) by row
    std::vector<float> conditional;  // map_height * (map_width + 1): the same along each row
    std::vector<float> density;      // map_width * map_height: each pixel's probability density over the unit square, in 0..
    // An orthonormal frame for the up axis: x = right, y = up, z = forward, so a direction is looked up as (x, y, z).
    float right[3];
    float up[3];
    float forward[3];
};

// Tables for env. A map must already have passed check_environment.
EnvironmentTables build_environment_tables(const Environment &env);

// Throws std::invalid_argument naming what is wrong.
void check_environment(const Environment &env);

}  // namespace neraulgraphics::detail

#endif  // NERAULGRAPHICS_ENVIRONMENT_H
