# neraulGraphics

A raytracing rendering engine for games and simulation, usable as a library from Python or
C++. It runs on an NVIDIA GPU through CUDA when there is one, and on the CPU everywhere else.
Actively developed.

![Screenshot](https://raw.githubusercontent.com/aneangel/neraulGraphics/main/docs/screenshot.png)

Hand it a scene and a camera, get an image back. Every pixel is traced with real
rays, so mirrors, shadows and reflections work without tricks, and the picture
sharpens the longer the camera holds still.

## Features

- **Runs anywhere.** An NVIDIA GPU is used automatically when present; otherwise every CPU core is. Same picture either way, on Linux, macOS and Windows.
- **Library first.** `pip install` and `import neraulgraphics`, or `find_package(neraulgraphics)` in C++. No window or engine to adopt.
- **Meshes and instancing.** Share triangle meshes between any number of instances. Moving an instance only updates a small BVH, so thousands of moving bodies stay cheap.
- **Dynamic scenes.** `set_instance_transforms()` takes a pose per instance each frame, with poses as position plus `(x, y, z, w)` quaternion, the convention Warp, MuJoCo and Newton use.
- **Load models.** `ng.load("model.glb")` reads glTF, GLB and OBJ with materials, colour textures, smooth normals and node hierarchies.
- **Newton and MuJoCo adapters.** `NewtonRenderer(model).render(state)` and `MuJoCoRenderer(model).render(data)`: shapes become instances once, and each frame only poses are updated.
- **Reflections, soft shadows, anti-aliasing**, plus progressive refinement while the camera is still.
- **Interactive viewer** with free-fly camera: `make raytracer`.
- **Clear errors.** Bad scenes raise `ValueError` (Python) or `std::invalid_argument` (C++) saying what is wrong, instead of crashing.

## Install

```bash
pip install git+https://github.com/aneangel/neraulGraphics
```

That builds from source, so it needs CMake and a C++17 compiler. If the CUDA toolkit is
installed it also builds the GPU backend for your card; if not, it builds the CPU backend,
which needs nothing else. Prebuilt wheels are on the
[releases page](https://github.com/aneangel/neraulGraphics/releases): Linux x86_64 with the GPU
backend (needing only the NVIDIA driver, and using the CPU if there is no GPU), and CPU-only
for macOS (Apple Silicon), Windows and Linux aarch64.

`neraulgraphics.cuda_available()` says whether a GPU will be used, and
`ng.Renderer(..., device="cpu")` forces the CPU. At 1280x720 a 16-core CPU renders a
138,000-triangle model in about 17 to 41 ms; a GPU takes under 1 ms.

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

## Loading a model

```python
scene = ng.load("duck.glb")              # .glb, .gltf or .obj
image = scene.renderer(1280, 720).render()
```

The renderer picks a camera and lights from the model's size; `view.camera(position)`
takes over the camera. For more control, hand `scene.meshes`, `scene.instances` and
`scene.textures` to `ng.Renderer` yourself. Reading texture images needs Pillow
(`pip install neraulgraphics[images]`). Not supported: animation and skinning (a
skinned model draws in its bind pose), normal and roughness maps, transparency,
Draco-compressed glTF. See [examples/view_model.py](https://github.com/aneangel/neraulGraphics/blob/main/examples/view_model.py) for a turntable.

## Rendering a simulation

```python
from neraulgraphics.newton import NewtonRenderer          # or neraulgraphics.mujoco.MuJoCoRenderer

view = NewtonRenderer(model, 1280, 720)                   # shapes become mesh instances, once
for _ in range(steps):
    solver.step(state_0, state_1, control, contacts, dt)
    image = view.render(state_1)                          # (720, 1280, 3) uint8; only poses are read
```

It draws spheres, boxes, capsules, cylinders, cones, ellipsoids, planes and meshes, with the
colours the model gives them, and frames the scene and lights it for you. Pass your own
`camera=` (from `view.camera(position, target)`) or `lights=` to take control. A real
articulated humanoid renders in under a millisecond a frame. Runnable versions are in
[examples/](https://github.com/aneangel/neraulGraphics/blob/main/examples/).

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

Needs CMake 3.28+, a C++17 compiler and SDL2. If CUDA 12+ is installed the viewer uses your
GPU (built for RTX 50-series by default; `make raytracer CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native"`
for other cards), and otherwise it renders on the CPU. `-DNERAULGRAPHICS_CUDA=OFF` skips CUDA
even when it is installed.

## Status

Pre-1.0: the API may change between minor versions (see the [changelog](https://github.com/aneangel/neraulGraphics/blob/main/CHANGELOG.md)).
Planned next: GPU-side instance updates for 100k+ moving bodies, texture mip-mapping, and
backends for AMD, Intel and Apple GPUs (see [portability](https://github.com/aneangel/neraulGraphics/blob/main/docs/portability.md)).

More detail, benchmarks and viewer controls are in the [guide](https://github.com/aneangel/neraulGraphics/blob/main/docs/guide.md).

## License

[Apache-2.0](https://github.com/aneangel/neraulGraphics/blob/main/LICENSE).

## Credits

Grew out of [ssloy/tinyraytracer](https://github.com/ssloy/tinyraytracer).
