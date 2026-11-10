# Changelog

Each release has a section here. The release workflow refuses to publish a version
without one, and uses the section as the release notes.

Versions follow [semantic versioning](https://semver.org). Before 1.0, a minor
version (0.x.0) may change the API.

## [Unreleased]

### Added
- Batched worlds. `BatchRenderer` draws many worlds at once, each seen by its own camera(s) and
  holding the same instances placed differently, and returns `(views, H, W, ...)` batches of RGB,
  depth, normals and object IDs. Poses (`set_instance_transforms`, or simulator body poses through
  `bind_bodies` and `set_body_poses`), cameras and rays may be CUDA arrays, used where they are.
  Worlds are isolated: a camera never sees another world's instances.
- The trees over instances are built on the GPU every update (Morton-code sort and parallel Karras
  construction): 82,000 instances in 0.15 ms and 100,000 in under 1 ms, against 89 ms when built on the CPU.
  The layout is now a forest with explicit child pointers, one root per world.
- `NewtonBatchRenderer` and `MJWarpBatchRenderer`: batched Newton and MuJoCo Warp simulations, reading the
  simulators' GPU state in place. `neraulgraphics.shapes.look_at` makes camera matrices.
- Sensor outputs. `Renderer.render_frame(camera, rgb=, depth=, normals=, object_ids=)` returns depth
  (along the view axis, `inf` where nothing is hit), world-space normals and object-ID images, each
  one exact ray through the pixel's middle. IDs are fixed numbers: `OBJECT_NONE`, `OBJECT_SPHERE + i`,
  `OBJECT_FLOOR`, `OBJECT_INSTANCE + i`. Asking for fewer images is faster: depth alone does no shading.
- `Renderer.cast_rays(origins, directions, max_distance)`: ray queries for LiDAR and range sensors.
- Zero-copy output. With `zero_copy=True` the images are views of the renderer's buffers, and on a GPU
  they are CUDA arrays for `torch.from_dlpack`, `jax.dlpack.from_dlpack`, `warp.from_dlpack` or
  `cupy.from_dlpack`, already finished. `cast_rays` takes CUDA arrays such as PyTorch tensors and
  answers on the GPU. C++: `render_frame`, `render_frame_host`, `cast_rays`, `cast_rays_host`.
- The viewer shows the sensor images: keys `1` to `4` switch between the picture, depth, normals
  and object IDs, `--mode` starts in one, and `--snapshot FILE.ppm` saves a frame without a window.
- `render_frame` and `cast_rays` on the Newton, MuJoCo and loaded-model renderers, `update(state)` to
  place shapes without drawing, and `object_id_of_shape` / `object_id_of_geom`.

## [0.4.0]

Newton and MuJoCo adapters, model loading, textures, and a CPU backend so it runs on any machine.

### Added
- A CPU backend. The renderer runs on every core of the machine when there is no NVIDIA GPU,
  drawing the same picture as the GPU (the two run the same rendering code). `Renderer` takes
  `device="auto" | "cuda" | "cpu"` (C++: `Device`), `Renderer.device` says which was chosen, and
  `cuda_available()` says whether a GPU is usable. `NERAULGRAPHICS_DEVICE=cpu` overrides `auto`;
  `NERAULGRAPHICS_THREADS` sets the CPU thread count. A machine with no GPU used to raise
  `CudaError`; it now falls back to the CPU.
- CUDA is optional at build time (`-DNERAULGRAPHICS_CUDA=AUTO|ON|OFF`; AUTO uses it if a CUDA
  compiler is found). Without it the library needs only a C++17 compiler and depends on no CUDA
  library or driver.
- Wheels for macOS (Apple Silicon), Windows and Linux aarch64, CPU-only, alongside the Linux x86_64
  CUDA wheels. The release workflow can publish to PyPI by trusted publishing, once enabled.
- `docs/portability.md`: what runs where, and how other GPUs could be supported.
- `neraulgraphics.newton.NewtonRenderer` and `neraulgraphics.mujoco.MuJoCoRenderer`:
  render a Newton or MuJoCo simulation. Shapes become mesh instances once; each
  `render(state)` only moves them. Sphere, box, capsule, cylinder, cone, ellipsoid,
  plane and mesh shapes, with colours from the model.
- `neraulgraphics.shapes`: capsule, cylinder, cone, plane and sphere generators, and the
  pose math (`pose_matrices`, `compose_poses`) the adapters use.
- `examples/` with Newton, MuJoCo and model-viewing scripts, and `mujoco` and `images` extras.
- Smooth shading and textures. `Mesh` takes optional per-vertex `normals` and `uvs`; a new
  `Texture` (RGBA8, sRGB, bilinear filtered, repeating) can be put on an `Instance` with its
  `texture` index, and multiplies into the material's diffuse colour. `Renderer` takes `textures`.
- `neraulgraphics.load`, `load_gltf` and `load_obj`: glTF 2.0 (`.gltf` and `.glb`) and
  Wavefront OBJ with materials, colour textures, vertex normals and node transforms, returning a
  `LoadedScene` whose `.renderer()` frames the model and lights it.
- `Renderer.ambient` (C++: `set_ambient`): light reaching every surface regardless of the lights,
  so faces turned from every light are not black. Defaults to 0; the adapters and loaders
  default to 0.2.
- Simulator planes draw as a checkerboard ground.

### Changed
- `make_camera` computes in double precision on the CPU, so a camera is identical on both backends.
- Python wheel tests now render for real on build machines, on the CPU backend.

### Known limits
- Textures are not mip-mapped, so distant fine patterns grain until samples accumulate.
- The CPU backend is roughly 40 to 100 times slower than a recent GPU (at 1280x720 a 16-core CPU takes 17 to 41 ms for 138k-triangle models).
- glTF: no animation or skinning (bind pose is drawn), normal maps, transparency or Draco.

## [0.3.0]

Triangle meshes and dynamic scenes.

### Added
- Apache-2.0 license.
- Triangle meshes. `Mesh` is geometry, built into a BVH once; `Instance` places a
  mesh with its own `Transform` and `Material`. The number of instances is not
  capped, unlike spheres. Pass them to `Renderer(..., meshes, instances)`.
- `Renderer::set_instance_transforms()` (Python: an `(N, 3, 4)` array or a list of
  `Transform`) moves instances without touching the meshes. The tree over
  instances is refitted in place and rebuilt only when it has loosened too far.
- `make_transform` (position, `(x, y, z, w)` quaternion, scale), `make_box` and
  `make_icosphere`, in C++ and Python.
- `make raytracer` builds the viewer and links it as `./raytracer`; `make run`
  and `make test` do the obvious. The viewer takes `--instances N` to spawn
  spinning shapes.

### Known limits
- Triangles are shaded flat, with no vertex normals or textures.
- Updating instances runs on the CPU: about 9 ms for 10,000 and 90 ms for 100,000.
- Ray-triangle tests are not watertight along an exact shared edge.

## [0.2.0]

The renderer is now a package you can call from Python or C++.

### Added
- Python package: `pip install`, then `import neraulgraphics`. `Renderer.render()`
  returns a `(height, width, 3)` uint8 numpy array. Vectors are plain tuples,
  lists or numpy arrays. Bad input raises `ValueError`; CUDA failures raise
  `neraulgraphics.CudaError`. Type stubs included.
- Linux wheels for Python 3.10 and later that need only the NVIDIA driver, not
  the CUDA toolkit, built for every GPU generation from Turing to Blackwell.
- `neraulgraphics/version.h`, generated from the version in `CMakeLists.txt`.
- C++ tests, run with `ctest`.

### Changed
- Renamed everything to `neraulgraphics`: `find_package(neraulgraphics)`, the
  `neraulgraphics::core` target, `#include <neraulgraphics/renderer.h>` and
  `namespace neraulgraphics`. The viewer is now `neraulgraphics-viewer`.
- The public headers no longer include any CUDA header. Vectors are
  `neraulgraphics::Vec3`, and `make_camera` is computed inside the library, so
  cameras no longer depend on the caller's floating point flags.
- The `Renderer` floor argument is optional; the default is no floor.

## [0.1.0]

- The renderer as a library: a `gpu::Renderer` class that owns its device
  memory, reports errors as exceptions, and rejects bad input before it reaches
  the GPU.
- CMake install and `find_package(raytracer)` support.
