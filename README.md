# neraulGraphics

A GPU-accelerated rendering engine for games and simulation, built on CUDA raytracing
and usable as a library from Python or C++. Actively developed.

![Screenshot](docs/screenshot.png)

Hand it a scene and a camera, get an image back. Every pixel is traced with real
rays, so mirrors, shadows and reflections work without tricks, and the picture
sharpens the longer the camera holds still.

## Features

- **Library first.** `pip install` and `import neraulgraphics`, or `find_package(neraulgraphics)` in C++. No window or engine to adopt.
- **Meshes and instancing.** Share triangle meshes between any number of instances. Moving an instance only updates a small BVH, so thousands of moving bodies stay cheap.
- **Dynamic scenes.** `set_instance_transforms()` takes a pose per instance each frame, with poses as position plus `(x, y, z, w)` quaternion, the convention Warp, MuJoCo and Newton use.
- **Reflections, soft shadows, anti-aliasing**, plus progressive refinement while the camera is still.
- **Interactive viewer** with free-fly camera: `make raytracer`.
- **Clear errors.** Bad scenes raise `ValueError` (Python) or `std::invalid_argument` (C++) saying what is wrong, instead of crashing.

## Install

You need an NVIDIA GPU and driver, on Linux x86_64.

```bash
# built for your GPU (needs the CUDA toolkit)
pip install git+https://github.com/aneangel/neraulGraphics@v0.3.0
```

Prebuilt wheels that need only the driver are on the
[releases page](https://github.com/aneangel/neraulGraphics/releases).

## Quick start

```python
import neraulgraphics as ng

red = ng.Material(diffuse_color=(0.6, 0.15, 0.15))
renderer = ng.Renderer(
    800, 450, spheres=[],
    lights=[ng.Light((-10, 20, 10), intensity=1.6, radius=1.0)],
    floor=ng.Plane(y=-2, half_extent_x=12, z_min=-30, z_max=-2),
    meshes=[ng.make_icosphere(1.0)],
    instances=[ng.Instance(0, ng.make_transform((0, 0, -8), scale=2.0), red)],
)
camera = ng.make_camera((0, 1, 0), (0, 0, -8), width=800, height=450)

for frame in range(64):                  # more frames, smoother image
    image = renderer.render(camera, frame)   # (450, 800, 3) uint8 numpy array
```

Move things every frame with `renderer.set_instance_transforms(matrices)`, an
`(N, 3, 4)` float array, then render with `frame=0`.

C++:

```cmake
find_package(neraulgraphics 0.3 REQUIRED)
target_link_libraries(my_app PRIVATE neraulgraphics::core)
```

```cpp
#include <neraulgraphics/renderer.h>
neraulgraphics::Renderer renderer(width, height, spheres, lights, floor, meshes, instances);
const uint32_t *pixels = renderer.render(camera, frame, 1);   // ARGB, width*height
```

## Build and run

```bash
make raytracer                         # builds ./raytracer
./raytracer --instances 5000           # viewer with 5000 spinning bodies
make test                              # needs a GPU
```

Needs CUDA 12+, CMake 3.28+ and SDL2. Builds for RTX 50-series by default; use
`make raytracer CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native"` for other GPUs.

## Status

Pre-1.0: the API may change between minor versions (see the [changelog](CHANGELOG.md)).
Planned next, roughly in order: smooth shading and textures, glTF/OBJ loading, GPU-side
instance updates for 100k+ moving bodies, and a bridge for rendering Newton and MuJoCo scenes.

More detail, benchmarks and viewer controls are in the [guide](docs/guide.md).

## License

[Apache-2.0](LICENSE).

## Credits

Grew out of [ssloy/tinyraytracer](https://github.com/ssloy/tinyraytracer).
