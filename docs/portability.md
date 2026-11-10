# Where it runs, and how it could run elsewhere

## What exists

| Backend | Hardware | How it is chosen |
|---|---|---|
| **CUDA** | NVIDIA GPUs (Turing, RTX 20-series, and later) | `device="auto"` uses it when the library was built with CUDA and a GPU is present |
| **CPU** | Anything with a C++17 compiler: every core of the machine | `device="auto"` falls back to it; `device="cpu"` forces it |

Both run **the same rendering code** (`src/render_core.h`), compiled twice: once as a CUDA kernel,
once as plain C++ spread over threads. They therefore draw the same picture, and a test checks
it: a scene using every feature (spheres, meshes, smooth normals, textures, soft shadows,
reflections, ambient light) differs between the two by under 0.1 out of 255 per colour channel on
average, with no visibly different pixels.

Speed is the difference. At 1280x720, one sample per pixel, on a 16-core CPU and an RTX 5090:

| Model | GPU | CPU |
|---|---|---|
| 4k triangles, textured | 0.25 ms | 32 ms |
| 135k triangles | 0.79 ms | 41 ms |
| 138k triangles | 0.39 ms | 17 ms |

A CPU renders these interactively (24 to 58 frames a second here; roughly a quarter of that on a
4-core laptop), which is plenty for previews, stills, CI, headless servers and anyone without an
NVIDIA card. It is not a substitute for a GPU when rendering many frames of a large scene.

## Prebuilt wheels

| Platform | Backends | Notes |
|---|---|---|
| Linux x86_64 | CUDA and CPU | needs only the NVIDIA driver, and uses the CPU if there is no GPU |
| macOS (Apple Silicon) | CPU | |
| Windows (x64) | CPU | |
| Linux aarch64 | CPU | Raspberry Pi, Graviton, and similar |

Anything else installs from the source archive with `pip install`, which needs CMake and a C++17
compiler, and builds the CUDA backend too if the CUDA toolkit is found.

## Other GPUs: AMD, Intel, Apple

None of these is supported, and nothing here pretends otherwise. Each needs its own backend, and a
backend that cannot be tested on real hardware should not be shipped, so they wait for someone with
the hardware. This is what each would involve:

- **AMD (ROCm/HIP).** The CUDA backend is ordinary CUDA C++: one kernel, device buffers, and texture
  objects. HIP is source-compatible for most of it, so this is the closest port. The uncertain part is
  texture objects with hardware sRGB decoding, which would need checking, or replacing with the CPU
  backend's explicit sampler.
- **Apple (Metal).** Metal Shading Language is C++14-based and the rendering core is plain C++ built
  on a small vector-type layer (`src/backend_api.h`), so the core should port by supplying that layer
  for MSL. The host side is new work, in Objective-C++ or Swift.
- **Intel and any vendor (SYCL or Vulkan).** SYCL is single-source C++ like CUDA and would reuse the
  core the same way. Vulkan compute works everywhere but needs the core rewritten in a shading language.
- **Hardware ray tracing (OptiX, DXR, Vulkan RT).** A different kind of change: traversal would move
  into the GPU's fixed-function units and replace the BVH code in `src/mesh_dev.h`. It gives a large
  speed-up on RTX cards and is worth doing after the portable backends, not instead of them.

## Adding a backend

1. Implement the five methods of `detail::Backend` (`src/backend.h`): render, move instances, get and
   set ambient light, last frame time.
2. Compile `render_core.h` for the new target, supplying the pieces `backend_api.h` lists: the vector
   types, how a function is qualified, reading a float's bits, and sampling a texture.
3. Choose it in `Renderer`'s constructor.
4. Add it to the comparison test in `tests/test_renderer.cpp`. The CPU backend is the reference
   there, because it runs on every machine: a new backend is correct when it draws what the CPU draws.

## Environment variables

- `NERAULGRAPHICS_DEVICE=cpu` or `cuda`: what `device="auto"` chooses. Useful to leave the GPU to
  another program, or to test the CPU path on a machine that has a GPU.
- `NERAULGRAPHICS_THREADS=N`: how many threads the CPU backend uses (default: one per core).
