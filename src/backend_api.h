#ifndef NERAULGRAPHICS_BACKEND_API_H
#define NERAULGRAPHICS_BACKEND_API_H

// What differs between the two ways the rendering code is compiled. The shading, traversal and
// intersection code is written once, in geom_dev.h, device_scene.h, mesh_dev.h and
// render_core.h, and compiled twice:
//
//   as CUDA device code, in cuda_backend.cu, in namespace neraulgraphics::device
//   as plain C++ for the CPU, in cpu_backend.cpp (which defines NG_BACKEND_CPU first),
//   in namespace neraulgraphics::host
//
// Each gets its own namespace so the two translation units never share a symbol, even where
// they define types of the same name. This header supplies the few things that differ: the
// vector types, the function qualifiers, reading a float's bits, and sampling a texture.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef NG_BACKEND_CPU

#define NG_NS host
#define GPU_FN inline
#define NG_FN inline

namespace neraulgraphics::host {

struct float3 {
    float x, y, z;
};
struct alignas(16) float4 {
    float x, y, z, w;
};

inline float3 make_float3(const float x, const float y, const float z) { return float3{x, y, z}; }
inline float4 make_float4(const float x, const float y, const float z, const float w) {
    return float4{x, y, z, w};
}

inline int ng_float_as_int(const float f) {
    int i;
    std::memcpy(&i, &f, sizeof(i));
    return i;
}

inline float ng_infinity() { return std::numeric_limits<float>::infinity(); }

// A texture on the CPU: the bytes as given, decoded and filtered when sampled (see sample_texture).
struct CpuTexture {
    int width;
    int height;
    const uint8_t *rgba;  // width * height * 4, rows from the top
};
using TextureHandle = const CpuTexture *;

// Bilinear, repeating, sRGB-decoded before filtering: what the GPU's texture unit does.
// Defined in cpu_backend.cpp.
float4 sample_texture(TextureHandle texture, float u, float v);

}  // namespace neraulgraphics::host

#else  // CUDA

#include <cuda_runtime.h>

#define NG_NS device
// __host__ __device__ under nvcc, plain inline otherwise.
#ifdef __CUDACC__
#define GPU_FN __host__ __device__ __forceinline__
#define NG_FN __device__
#else
#define GPU_FN inline
#define NG_FN inline
#endif

namespace neraulgraphics::device {

using TextureHandle = cudaTextureObject_t;

#ifdef __CUDACC__
__device__ __forceinline__ int ng_float_as_int(const float f) { return __float_as_int(f); }

// Built from its bits, so it survives -use_fast_math, which otherwise assumes there is no infinity.
__device__ __forceinline__ float ng_infinity() { return __int_as_float(0x7f800000); }

// Hardware filtering and sRGB decode, set up when the texture object is created.
__device__ __forceinline__ float4 sample_texture(const TextureHandle texture, const float u, const float v) {
    return tex2D<float4>(texture, u, v);
}
#endif

}  // namespace neraulgraphics::device

#endif  // NG_BACKEND_CPU

#endif  // NERAULGRAPHICS_BACKEND_API_H
