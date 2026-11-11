# Beyond CUDA: the portable backend

neraulGraphics renders on CUDA and on the CPU, which leaves every AMD, Intel and Apple GPU on the CPU path. The portable backend runs the
same renderer on any GPU through Vulkan. This page records what was measured, what was decided and why, and what comes next.

## What was measured (milestone V0)

One machine with three Vulkan devices, all of which expose `VK_KHR_ray_query`, `VK_KHR_acceleration_structure` and buffer device
addresses:

| device | Vulkan | subgroup | ray queries | 1 M rays, 2,000 boxes |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 | 1.4.351 | 32 | hardware (RT cores) | 0.19 ms (5,400 Mrays/s) |
| AMD Radeon iGPU (RADV, Raphael) | 1.4.318 | 64 | hardware (RDNA2 ray accelerators) | 20 ms (49 Mrays/s) |
| llvmpipe (software, LLVM 20) | 1.4.318 | 8 | software | 80 ms (12.5 Mrays/s) |

`tools/vulkan` builds a bottom- and top-level acceleration structure, traces rays with one compute shader, and `compare.py` checks every
hit against the CPU backend's `cast_rays`: hit or miss, which instance, and how far. On a million rays all three devices agree with
the CPU on 99.9996 to 99.9999 % of rays on every measure. The few that differ per million graze a silhouette edge, where two
intersection tests round differently. Rays that start within 0.1 mm of a surface are set aside: the CPU counts the self-hit and a
ray query starting at `tMin` does not.

Two things the first run taught:

- **Where the buffers live decides the speed.** With everything in host-visible memory the RTX 5090 took 10.7 ms for 100,000 rays (it was
  walking its BVH across PCIe) and was slower than the iGPU. In device-local memory, filled through a staging buffer, it takes 0.19 ms
  for ten times the rays.
- **Subgroup size is 8, 32 or 64.** Nothing may assume one. The work group is a fixed 64 and no code depends on lanes.

## What was decided

1. **Vulkan compute with inline ray queries, not the ray tracing pipeline.** One compute shader owns the whole path-tracing loop, which
   makes it a plain port of the CUDA kernel, runs on every device that has the extension, and lets the software device serve as a
   fallback. A shader binding table would add machinery and buy nothing here.
2. **The CPU stays the oracle.** Every portable kernel is judged against the CPU backend on the same scene, as V0 does, and the CPU/GPU
   agreement stays a test, not a hope.
3. **One acceleration structure per mesh, one for the instances,** as the CUDA backend already does (`cuda_tlas.cuh`): a simulation
   that moves thousands of bodies a frame rebuilds only the top level.
4. **Single-source shaders, eventually.** The CUDA and CPU backends share `render_core.h` through macros. The Vulkan shaders start as GLSL
   (it is what compiled and passed first); the plan is to move the shared core to [Slang](https://github.com/shader-slang/slang), which
   compiles one source to SPIR-V, CUDA and CPU code, so a fix lands once. `slangc` is already on the development machine.
5. **Rendering is allowed to differ in the last bits; the simulation is not.** See the physics section of [guide.md](guide.md). A portable backend is held
   to "the same picture to within a fraction of a code value", not to bit equality, because hardware ray tracing cannot be bit-identical
   across vendors.

## What comes next

| | milestone | done when |
|---|---|---|
| V0 | acceleration structures and ray queries on three devices | **done**: this page |
| V1 | the `Backend` interface (src/backend.h) implemented for Vulkan: classic mode, spheres, floor, meshes, instances, `cast_rays` | the Python test suite passes with `device="vulkan"` on all three devices |
| V2 | path tracing and the denoiser in the Vulkan backend | images match the CPU's within 1/255 on the existing tests |
| V3 | the instance hand-off from a simulation without a CPU round trip (external memory shared with CUDA on NVIDIA, a host copy elsewhere) | a Newton scene renders on the AMD device through the hand-off |
| V4 | tuned memory (Vulkan Memory Allocator), async upload, timeline semaphores; hardware-specific tuning behind the same interface | published numbers on an NVIDIA, an AMD and an Intel GPU |

Not yet verified: an Intel GPU, Apple's Metal-over-MoltenVK path, and Windows.

## Running V0

```bash
cmake -S . -B build -DNERAULGRAPHICS_BUILD_VULKAN=ON && cmake --build build --target vk_rayquery vk_rayquery_shader
python tools/vulkan/compare.py --tool build/vk_rayquery --spv build/rayquery.spv
NERAULGRAPHICS_VK_BUILD=build pytest tests/test_vulkan.py
```
