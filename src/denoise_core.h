#ifndef NERAULGRAPHICS_DENOISE_CORE_H
#define NERAULGRAPHICS_DENOISE_CORE_H

// The denoiser for path-traced pictures, shared by both backends. A path tracer given a few samples per pixel
// makes a noisy picture; this turns it into a clean one by reusing what other frames and neighbouring pixels know.
// It follows Schied et al., "Spatiotemporal Variance-Guided Filtering" (2017):
//
//   1. The colour of the first surface is divided out of the light, so the filter blurs only the lighting and never a texture.
//   2. Last frame's result is moved to where the same surface point is now (reprojection) and blended in, which
//      averages the noise over time. History is thrown away wherever the surface is not the one it was (a change of
//      depth or of normal), which is what stops ghosts trailing behind things that move.
//   3. An edge-aware wavelet filter (a trous) then smooths across space in five passes of widening reach. How far a
//      neighbour counts depends on how similar its normal, depth and brightness are, with brightness judged against
//      the variance measured over time, so noisy areas are smoothed hard and clean ones left alone.
//   4. The colour is put back.
//
// All buffers hold float4 per pixel. A "G-buffer" pixel is (normal.xyz, depth), with negative depth where no surface was hit.

#include "path_core.h"

namespace neraulgraphics::NG_NS {
namespace {

NG_FN float luminance(const float3 &c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Where a world point lands in a camera's picture, in pixels, and how far along the view axis it is. False if it is behind the camera.
NG_FN bool project_to_camera(const Camera &c, const float3 &p, const int width, const int height, float &px, float &py, float &z) {
    const float3 v = p - c.position;
    z = dot(v, c.forward);
    if (z <= 1e-4f) return false;
    const float sx = dot(v, c.right) / z, sy = dot(v, c.up) / z;
    px = (sx / (c.fov_scale * c.aspect) + 1.0f) * 0.5f * static_cast<float>(width);
    py = (1.0f - sy / c.fov_scale) * 0.5f * static_cast<float>(height);
    return true;
}

// Step 2: blend this frame's noisy light with the history from the last, and measure the variance over time.
//   noisy     this frame's light with the colour divided out (rgb)
//   gbuf      this frame's normals and depths; prev_gbuf the last frame's
//   hist_in   last frame's accumulated light (rgb) and how many frames it covers (w); mom_in its brightness and brightness squared
//   hist_out, mom_out   this frame's, to be read next frame
//   filtered  the light (rgb) and its variance (w) for the filter
// valid_history is false for a first frame or a cut, when nothing from before is to be trusted.
// blend_min is the smallest weight a new frame gets however long the history: lower is smoother and slower to follow change.
NG_FN void temporal_accumulate_pixel(const int x, const int y, const int width, const int height, const Camera &cam, const Camera &prev,
    const bool valid_history, const float4 *noisy, const float4 *gbuf, const float4 *prev_gbuf, const float4 *hist_in,
    const float4 *mom_in, float4 *hist_out, float4 *mom_out, float4 *filtered, const float blend_min) {
    const int idx = y * width + x;
    const float4 cur = noisy[idx];
    const float4 g = gbuf[idx];
    const float3 cur_rgb = make_float3(cur.x, cur.y, cur.z);
    const float lum = luminance(cur_rgb);

    float length = 0.0f;
    float3 hist = make_float3(0.0f, 0.0f, 0.0f);
    float m1 = 0.0f, m2 = 0.0f;
    if (valid_history && g.w > 0.0f) {
        const float3 dir = camera_ray(cam, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f, width, height);
        const float3 world = cam.position + dir * (g.w / dot(dir, cam.forward));
        float px, py, z;
        if (project_to_camera(prev, world, width, height, px, py, z)) {
            // The four pixels around where it was, weighted as a bilinear lookup would, but each only if it was the same surface.
            const float fx = px - 0.5f, fy = py - 0.5f;
            const int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
            const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
            float total = 0.0f, hl = 0.0f, a1 = 0.0f, a2 = 0.0f;
            float3 hc = make_float3(0.0f, 0.0f, 0.0f);
            for (int k = 0; k < 4; ++k) {
                const int tx_i = x0 + (k & 1), ty_i = y0 + (k >> 1);
                if (tx_i < 0 || ty_i < 0 || tx_i >= width || ty_i >= height) continue;
                const int ti = ty_i * width + tx_i;
                const float4 pg = prev_gbuf[ti];
                const bool same_surface = pg.w > 0.0f && fabsf(pg.w - z) < 0.05f * z + 1e-3f &&
                    dot(make_float3(pg.x, pg.y, pg.z), make_float3(g.x, g.y, g.z)) > 0.85f;
                if (!same_surface) continue;
                const float w = ((k & 1) ? tx : 1.0f - tx) * ((k >> 1) ? ty : 1.0f - ty);
                const float4 hh = hist_in[ti], mm = mom_in[ti];
                hc += make_float3(hh.x, hh.y, hh.z) * w;
                hl += hh.w * w;
                a1 += mm.x * w;
                a2 += mm.y * w;
                total += w;
            }
            if (total > 0.01f) {
                hist = hc * (1.0f / total);
                length = hl / total;
                m1 = a1 / total;
                m2 = a2 / total;
            }
        }
    }

    const float new_length = fminf(length + 1.0f, 1024.0f);
    const float alpha = fmaxf(1.0f / new_length, blend_min);
    const float3 color = length > 0.0f ? hist * (1.0f - alpha) + cur_rgb * alpha : cur_rgb;
    const float nm1 = length > 0.0f ? m1 * (1.0f - alpha) + lum * alpha : lum;
    const float nm2 = length > 0.0f ? m2 * (1.0f - alpha) + lum * lum * alpha : lum * lum;
    float variance = fmaxf(0.0f, nm2 - nm1 * nm1);

    // With only a few frames of history the variance over time is itself unreliable: use the spread of the neighbouring pixels.
    if (new_length < 4.0f) {
        float sum = 0.0f, sum2 = 0.0f, n = 0.0f;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int qx = x + dx, qy = y + dy;
                if (qx < 0 || qy < 0 || qx >= width || qy >= height) continue;
                const float4 q = noisy[qy * width + qx];
                const float ql = luminance(make_float3(q.x, q.y, q.z));
                sum += ql;
                sum2 += ql * ql;
                n += 1.0f;
            }
        const float mean = sum / n;
        variance = fmaxf(variance, fmaxf(0.0f, sum2 / n - mean * mean));
    }

    // What was measured is how much one sample varies; the pixel holds the average of new_length of them, whose uncertainty is that much smaller.
    variance /= new_length;

    hist_out[idx] = make_float4(color.x, color.y, color.z, new_length);
    mom_out[idx] = make_float4(nm1, nm2, 0.0f, 0.0f);
    filtered[idx] = make_float4(color.x, color.y, color.z, variance);
}

// Step 3, one pass: replace each pixel by a weighted average of the 5x5 pixels around it, spaced `step` apart. Neighbours count
// more the more their normal, depth and brightness resemble the centre's. The variance goes through the filter too, so
// the next pass knows how noisy what it is given still is.
NG_FN void atrous_pixel(const int x, const int y, const int width, const int height, const float4 *in, float4 *out, const float4 *gbuf,
    const int step) {
    const int idx = y * width + x;
    const float4 c = in[idx];
    const float4 g = gbuf[idx];
    if (g.w <= 0.0f) {  // not a surface (the sky): there is nothing to smooth against
        out[idx] = c;
        return;
    }
    const float3 n = make_float3(g.x, g.y, g.z);
    const float lum = luminance(make_float3(c.x, c.y, c.z));
    // How fast depth changes across the screen here, so a slanted surface's own depth change is not taken for an edge.
    auto depth_at = [&](const int qx, const int qy) {
        const float d = gbuf[qy * width + qx].w;
        return d > 0.0f ? d : g.w;
    };
    const float gx = 0.5f * (depth_at(x + 1 < width ? x + 1 : x, y) - depth_at(x > 0 ? x - 1 : x, y));
    const float gy = 0.5f * (depth_at(x, y + 1 < height ? y + 1 : y) - depth_at(x, y > 0 ? y - 1 : y));
    const float sigma_l = 4.0f, sigma_z = 1.0f, power_n = 64.0f;
    const float kernel[5] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float sum_var = 0.0f, sum_w = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int qx = x + dx * step, qy = y + dy * step;
            if (qx < 0 || qy < 0 || qx >= width || qy >= height) continue;
            const float4 qg = gbuf[qy * width + qx];
            if (qg.w <= 0.0f) continue;
            const float4 q = in[qy * width + qx];
            const float h = kernel[dx + 2] * kernel[dy + 2];
            const float w_n = powf(fmaxf(0.0f, dot(n, make_float3(qg.x, qg.y, qg.z))), power_n);
            const float expected = fabsf(gx * static_cast<float>(dx * step) + gy * static_cast<float>(dy * step));
            const float w_z = expf(-fabsf(g.w - qg.w) / (sigma_z * expected + 1e-3f * g.w + 1e-4f));
            const float ql = luminance(make_float3(q.x, q.y, q.z));
            const float w_l = expf(-fabsf(lum - ql) / (sigma_l * sqrtf(fmaxf(c.w, 0.0f)) + 1e-4f));
            const float w = h * w_n * w_z * w_l;
            sum += make_float3(q.x, q.y, q.z) * w;
            sum_var += q.w * w * w;
            sum_w += w;
        }
    }
    if (sum_w < 1e-6f) {
        out[idx] = c;
        return;
    }
    const float inv = 1.0f / sum_w;
    out[idx] = make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum_var * inv * inv);
}

// Step 4 and the last: the colour goes back in, the picture is exposed and tone mapped, and packed to 0xAARRGGBB.
NG_FN uint32_t resolve_pixel(const PathParams &p, const float3 &linear, const bool tonemap) {
    const float3 c = linear * p.exposure;
    return pack_argb8888(linear_to_srgb(tonemap ? tonemap_aces(c) : c));
}

}  // namespace
}  // namespace neraulgraphics::NG_NS

#endif  // NERAULGRAPHICS_DENOISE_CORE_H
