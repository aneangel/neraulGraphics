#include "environment.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "validate.h"

namespace neraulgraphics::detail {

void check_environment(const Environment &env) {
    require(finite(env.up) && env.up.x * env.up.x + env.up.y * env.up.y + env.up.z * env.up.z > 1e-12f,
        "environment up must be a nonzero vector with no NaN or infinite value");
    for (const Vec3 &c : {env.color_up, env.color_down})
        require(finite(c) && c.x >= 0.0f && c.y >= 0.0f && c.z >= 0.0f, "environment colours must be finite and not negative");
    require(finite(env.intensity) && env.intensity >= 0.0f, "environment intensity must be 0 or more, got " + std::to_string(env.intensity));
    require(finite(env.rotation), "environment rotation has a NaN or infinite value");
    if (env.map.empty() && env.map_width == 0 && env.map_height == 0) return;
    require(env.map_width > 0 && env.map_height > 0, "an environment map needs a positive size, got " +
        std::to_string(env.map_width) + "x" + std::to_string(env.map_height));
    require(static_cast<long long>(env.map_width) * env.map_height <= (1ll << 26), "the environment map is too large");
    require(env.map.size() == static_cast<size_t>(env.map_width) * env.map_height * 3,
        "an environment map of " + std::to_string(env.map_width) + "x" + std::to_string(env.map_height) + " needs " +
        std::to_string(static_cast<size_t>(env.map_width) * env.map_height * 3) + " floats, got " + std::to_string(env.map.size()));
    for (size_t i = 0; i < env.map.size(); ++i)
        require(finite(env.map[i]) && env.map[i] >= 0.0f, "environment map value " + std::to_string(i) + " is negative, NaN or infinite");
}

EnvironmentTables build_environment_tables(const Environment &env) {
    EnvironmentTables t{};
    // The frame: up is up; right is any perpendicular to it.
    double u[3] = {env.up.x, env.up.y, env.up.z};
    const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (double &v : u) v /= un;
    const double probe[3] = {std::fabs(u[1]) < 0.9 ? 0.0 : 1.0, std::fabs(u[1]) < 0.9 ? 1.0 : 0.0, 0.0};
    double r[3] = {u[1] * probe[2] - u[2] * probe[1], u[2] * probe[0] - u[0] * probe[2], u[0] * probe[1] - u[1] * probe[0]};
    const double rn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    for (double &v : r) v /= rn;
    const double f[3] = {r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0]};
    for (int k = 0; k < 3; ++k) {
        t.right[k] = static_cast<float>(r[k]);
        t.up[k] = static_cast<float>(u[k]);
        t.forward[k] = static_cast<float>(f[k]);
    }
    if (env.map.empty()) return t;

    const int w = env.map_width, h = env.map_height;
    t.rgba.resize(static_cast<size_t>(w) * h * 4);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        t.rgba[4 * i] = env.map[3 * i];
        t.rgba[4 * i + 1] = env.map[3 * i + 1];
        t.rgba[4 * i + 2] = env.map[3 * i + 2];
        t.rgba[4 * i + 3] = 1.0f;
    }

    // A pixel's weight is its brightness times the sine of its angle from straight up: rows near the poles cover
    // little of the sphere, so they should be picked less. A small floor keeps every direction samplable.
    // The map is read with bilinear filtering, so a bright pixel also lights the ones next to it: the brightness used is the
    // largest in the pixel's 3x3 neighbourhood, or the light spread onto a dark neighbour would almost never be sampled.
    std::vector<double> luminance(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < luminance.size(); ++i)
        luminance[i] = 0.2126 * env.map[3 * i] + 0.7152 * env.map[3 * i + 1] + 0.0722 * env.map[3 * i + 2];
    std::vector<double> weight(static_cast<size_t>(w) * h), row_sum(h, 0.0);
    double total = 0.0;
    for (int y = 0; y < h; ++y) {
        const double sin_theta = std::sin(3.14159265358979 * (y + 0.5) / h);
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            double lum = 0.0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qy = std::min(std::max(y + dy, 0), h - 1), qx = (x + dx + w) % w;
                    lum = std::max(lum, luminance[static_cast<size_t>(qy) * w + qx]);
                }
            weight[i] = (lum + 1e-4) * sin_theta;
            row_sum[y] += weight[i];
        }
        total += row_sum[y];
    }
    t.marginal.assign(h + 1, 0.0f);
    t.conditional.assign(static_cast<size_t>(h) * (w + 1), 0.0f);
    double running = 0.0;
    for (int y = 0; y < h; ++y) {
        t.marginal[y] = static_cast<float>(running / total);
        running += row_sum[y];
        double along = 0.0;
        for (int x = 0; x < w; ++x) {
            t.conditional[static_cast<size_t>(y) * (w + 1) + x] = static_cast<float>(along / row_sum[y]);
            along += weight[static_cast<size_t>(y) * w + x];
        }
        t.conditional[static_cast<size_t>(y) * (w + 1) + w] = 1.0f;
    }
    t.marginal[h] = 1.0f;
    // Probability density over the unit square: the weight over its mean.
    t.density.resize(static_cast<size_t>(w) * h);
    const double mean = total / (static_cast<double>(w) * h);
    for (size_t i = 0; i < t.density.size(); ++i) t.density[i] = static_cast<float>(weight[i] / mean);
    return t;
}

}  // namespace neraulgraphics::detail
