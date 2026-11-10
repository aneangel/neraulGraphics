# Changelog

Each release has a section here. The release workflow refuses to publish a version
without one, and uses the section as the release notes.

Versions follow [semantic versioning](https://semver.org). Before 1.0, a minor
version (0.x.0) may change the API.

## [Unreleased]

### Added
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
