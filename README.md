# neraulGraphics

A real-time raytracer that runs on your GPU, as an interactive viewer or as a
library you call from Python or C++.

![Screenshot](docs/screenshot.png)

## What this is

Most 3D graphics you see in games is made by *rasterizing*. This does the opposite. For every pixel on
screen it fires a ray out into the world and follows it until it hits something,
then works out what color that surface should be by tracing more rays toward each
light.

That is a lot of rays. Doing that fast enough to move around in real time is the whole point of the project. It runs as a
CUDA program on the GPU, where thousands of pixels get traced at the same time.

## What it can do

- **Mirrors** — reflective surfaces that bounce rays up to five times, so mirrors
  show up inside other mirrors
- **Shadows** — from multiple lights at once, with optional soft edges
- **Anti-aliasing** — no jagged staircase edges on the spheres
- **A picture that improves while you sit still** — see [below](#the-image-gets-better-if-you-wait)
- **Save a screenshot** with `P`

## Requirements (at least what I used for this project)

- An NVIDIA GPU (RTX 50-series out of the box — see the note below for others)
- CUDA Toolkit 12 or newer
- SDL2
- CMake 3.28+

On Ubuntu or Debian:

```bash
sudo apt install libsdl2-dev cmake build-essential
```

CUDA comes from [NVIDIA's installer](https://developer.nvidia.com/cuda-downloads).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Using a different GPU?** The build targets RTX 50-series (`120`) by default. Pass
your card's number instead, or `native` to auto-detect:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
```

## Run

```bash
./build/neraulgraphics-viewer
```

### Controls

| Key | What it does |
|---|---|
| Drag left mouse | Look around |
| `W` `A` `S` `D` | Move around |
| `Q` / `E` | Move down / up |
| `Shift` | Move faster |
| `P` | Save a screenshot as `screenshot.ppm` |
| `Esc` | Quit |

The window title shows your framerate, how long the GPU spent on the last frame,
and how many samples have built up in the current image.

## The image gets better if you wait

This is the part worth knowing about.

A raytracer normally faces a trade-off: firing more rays per pixel makes the
picture cleaner but slows everything down. This one sidesteps that.

So:

- **While you're moving**, you get one quick sample. Maximum speed, and any
  roughness is hidden by the motion anyway.
- **The moment you stop**, samples start stacking up. After a second or so the
  image has averaged hundreds of them and looks genuinely smooth.

It's the same trick modern real-time raytracing in games uses.

It also makes normally-expensive effects cheap. Soft shadows, for example, would
usually mean firing dozens of rays at different points on a light.
## Changing the scene

Everything lives at the top of `src/main.cpp`, and it's plain data — no scene file
format to learn.

**Add a sphere.** Position, radius, material:

```cpp
{{0.0f, 1.0f, -20.0f}, 2.5f, ivory},
```

**Make a material.** Color, then how matte it is, how shiny, how mirror-like
(`0` = no reflection, `1` = perfect mirror), and how tight the highlight is:

```cpp
const Material jade = make_material({0.2f, 0.5f, 0.4f}, 0.7f, 0.2f, 0.15f, 80.0f);
```

Keep color values between 0 and 1, above that a surface would be reflecting more
light than lands on it, and it'll just clip to white. To make something brighter,
turn up a light instead.

**Turn on soft shadows.** The last number on each light is its physical size.
It's at `0`, which gives crisp shadows. Give it some size and the shadow edges
go soft:

```cpp
{{-20.0f, 20.0f, 20.0f}, 1.5f, 3.0f},   // position, brightness, radius
```

**Change the resolution.** Near the top of `main()`:

```cpp
constexpr int width  = 3840;   // 4K
constexpr int height = 2160;
```

## Speed

Time the GPU spends tracing one frame, on an RTX 5090:

| Resolution | Time per frame |
|---|---|
| 1280×960 | 0.07 ms |
| 1920×1080 | 0.10 ms |
| 3840×2160 (4K) | 0.28 ms |

For comparison, an earlier CPU version of this same renderer took about 23 ms at
640×480. 4K here has 27× as many pixels and still finishes roughly 80× quicker.

Tracing is no longer the slow part, copying the finished image back for display
now takes longer than drawing it.

## Use it from Python

```bash
pip install neraulgraphics-0.2.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Wheels are on the [releases page](https://github.com/aneangel/neraulGraphics/releases).
They need only an NVIDIA driver, not the CUDA toolkit, and work on any NVIDIA GPU
from the RTX 20-series on. Or build from source, which needs the CUDA toolkit and
compiles for the GPU in the machine you build on:

```bash
pip install git+https://github.com/aneangel/neraulGraphics@v0.2.0
```

Then:

```python
import neraulgraphics as ng

red = ng.Material(diffuse_color=(0.5, 0.1, 0.1))
mirror = ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0, specular_albedo=10,
                     reflectivity=0.8, specular_exponent=1425)

renderer = ng.Renderer(
    800, 450,
    spheres=[ng.Sphere((-2, 0, -12), 2, red), ng.Sphere((2.5, 0.5, -14), 2.5, mirror)],
    lights=[ng.Light((-10, 20, 10), intensity=1.6, radius=1.0)],
    floor=ng.Plane(y=-2, half_extent_x=12, z_min=-30, z_max=-2),
)
camera = ng.make_camera((0, 1, 0), (0, 0, -13), width=800, height=450)

for frame in range(64):              # more frames, smoother image
    image = renderer.render(camera, frame)

# image is a (450, 800, 3) uint8 numpy array
```

Save it with any image library, for example `PIL.Image.fromarray(image).save("out.png")`.
Vectors can be tuples, lists or numpy arrays. A mistake in the scene raises
`ValueError` saying what is wrong.

## Use it from C++

The renderer is a library, `neraulgraphics::core`, with no window or SDL attached.
Your project doesn't need to compile any CUDA, or even have the CUDA headers.

Install it once:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ~/.local
```

Then, in your project's `CMakeLists.txt`:

```cmake
find_package(neraulgraphics 0.2 REQUIRED)
target_link_libraries(my_app PRIVATE neraulgraphics::core)
```

Configure with `-DCMAKE_PREFIX_PATH=~/.local` if CMake can't find it. Or skip
installing and pull a release straight from GitHub:

```cmake
include(FetchContent)
FetchContent_Declare(neraulgraphics
    GIT_REPOSITORY https://github.com/aneangel/neraulGraphics.git
    GIT_TAG v0.2.0)
FetchContent_MakeAvailable(neraulgraphics)
```

Rendering a frame takes a scene, a camera, and a loop:

```cpp
#include <neraulgraphics/renderer.h>

neraulgraphics::Renderer renderer(width, height, spheres, lights, floor);
const auto camera = neraulgraphics::make_camera({0, 1, 0}, {0, 0, -13}, {0, 1, 0},
                                                1.57f, width, height);
const uint32_t *pixels = nullptr;
for (int frame = 0; frame < 64; ++frame)   // more frames, smoother image
    pixels = renderer.render(camera, frame, 1);
// pixels: width*height ARGB values, ready to save or display
```

If something is wrong with the scene, the renderer throws an exception that says
what, rather than closing your program.

## Project layout

| File | What's in it |
|---|---|
| `include/neraulgraphics/renderer.h` | The `Renderer` class you call to render a frame |
| `include/neraulgraphics/scene.h` | The scene building blocks: spheres, lights, materials, camera |
| `src/render_gpu.cu` | The actual raytracing, running on the GPU |
| `src/main.cpp` | The viewer: window, keyboard and mouse, camera, scene setup |
| `python/` | The Python package and its bindings |
| `tests/` | C++ tests (`ctest`) and Python tests (`pytest`); both need a GPU |
| `CHANGELOG.md`, `RELEASING.md` | What changed in each version, and how to publish one |

## Credits

The scene and the original CPU raytracer it grew out of were inspired by
[ssloy/tinyraytracer](https://github.com/ssloy/tinyraytracer).

And thank you to one of many [resources for raytracing](https://github.com/RayTracing/raytracing.github.io/tree/release)
