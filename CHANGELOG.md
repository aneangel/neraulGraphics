# Changelog

Each release has a section here. The release workflow refuses to publish a version
without one, and uses the section as the release notes.

Versions follow [semantic versioning](https://semver.org). Before 1.0, a minor
version (0.x.0) may change the API.

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
