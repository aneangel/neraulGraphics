# Changelog

Each release has a section here. The release workflow refuses to publish a version
without one, and uses the section as the release notes.

Versions follow [semantic versioning](https://semver.org). Before 1.0, a minor
version (0.x.0) may change the API.

## [Unreleased]

### Added
- **A steadier picture while the camera moves**, following the principles of the original raytracer (explicit lights evaluated analytically, deterministic
  shading, accumulate when still and restart on change): (1) `sun_light()` and `physical_sky(include_sun=False)` make the sun an explicit light, so a highlight is
  computed from the light's direction with a shadow ray and not found by a random ray landing on a small disc in the sky map (the cause of glints that twinkle as
  a surface moves under a pixel); `environment_frame()` gives the frame the renderer uses for a map, so the light sits where the map's sun would (the map's x
  and z are the world's -z and +x for an up of +y). (2) `PathTracing.moving_blend` (default 0.2): while the camera differs from the previous frame the denoiser
  trusts new frames more, because reprojected history of a moving view carries lag and misalignment; a still camera uses `temporal_blend` and accumulates
  deeply. (3) The examples no longer reset the frame counter when the camera or bodies move, which had thrown away all denoiser history on every moving frame;
  they reset it only when the light changes. On an orbiting camera in the materials scene the error against a converged picture fell 62 % (0.0316 to 0.0120).
  Documentation fix: a sky map's `sun_strength` is irradiance (a white diffuse surface facing it shows `sun_strength / pi`), while a `Light`'s intensity is that
  radiance, so the same sun as a `Light` has intensity `sun_strength / pi`; the old docstrings said the units were the same.
- **Glass** (`Material.transmission`, `Material.ior`): a smooth dielectric with exact Fresnel reflection, refraction by Snell's law, total internal
  reflection, medium tracking for closed solids and Beer-Lambert tinting by `diffuse_color`. Glass casts a light, tinted shadow (shadow rays pass through
  it) and does not spend the path's bounce budget. A clear ball in a uniform environment returns exactly the environment, a ball lens inverts what is
  behind it, and an index of 1 is invisible (`tests/test_glass.py`).
- **`physical_sky`**: the Preetham daylight model with a sun that is attenuated by the atmosphere (a low sun is redder and dimmer), turbidity, a ground
  lit by sun and sky, and the sun-to-sky ratio set from physics (`tests/test_sky.py`).
- **White-furnace tests** (`tests/test_furnace.py`): a convex object in a uniform environment has an exactly known appearance. They found that rough metals
  lost energy (a white metal at roughness 0.6 returned 81 % of the light, 0.8 54 %, 1.0 29 %) and fixed it with multiple-scattering compensation fitted to
  this renderer's own BRDF (now 100 % to 98 % up to 0.8 and 92 % at 1.0; `tools/rendering/single_scatter_albedo.py`), and that output used a plain 2.2 power in
  place of the sRGB curve (shadows came out too dark; now exact sRGB).
- Moving, shaped lights. `Renderer.set_lights(lights)` replaces a scene's lights between frames (a torch that follows a camera). `Light` gains
  `falloff` (intensity / (1 + (d / falloff)^2), a lamp), `direction`, `spot_angle` and `spot_softness` (a spotlight with a smooth cone) and
  `color` (a warm torch or a flame). All default to what a plain light was, so existing scenes draw identically; classic and path-traced
  shading share one `light_factor`, CPU and CUDA agree, and every field is validated (`tests/test_lights.py`).
- Normal maps and occlusion/roughness/metal maps. `Instance(normal_texture=, orm_texture=)` and `Texture(image, linear=True)` for data that must not
  be sRGB-decoded. A tangent frame is built from the triangle's uvs inside `mesh_surface`, the one function both backends share, so the CPU and
  CUDA backends draw them identically (tested). Roughness and metal reach the path tracer, occlusion darkens the diffuse colour. The street
  scene's brick has relief and mortar occlusion, and its asphalt has wet, smoother patches that mirror the sky.
- Articulated people (`examples/scenery/characters.py`): humanoids of ellipsoid segments walked by forward kinematics (hip, knee, shoulder
  and elbow swing, the pelvis solved so the lower foot is on the ground), tested for constant bone lengths, joints that stay joined, feet on
  the ground, speed, arms against legs, and non-degenerate matrices.
- `neraulgraphics.physics`: a rigid-body world whose results are bit-identical on any CPU, operating system and thread count. Jolt
  Physics built for cross-platform determinism solves the contacts, and nothing else changes the simulation. `state_hash()` fingerprints
  every body's exact state. Built only with `-C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON` (it fetches Jolt). A golden hash in
  `tests/test_physics.py` enforces it; `docs/guide.md` says what is and is not guaranteed.
- The portable backend, milestone V0 (`tools/vulkan`, `docs/portable-backend.md`): acceleration structures and inline ray queries on any
  Vulkan device. On an NVIDIA RTX 5090, an AMD Radeon iGPU and the llvmpipe software device it agrees with the CPU backend on 99.9996 to
  99.9999 % of a million rays, and traces 5,400, 49 and 12.5 Mrays/s respectively. Opt-in with `-DNERAULGRAPHICS_BUILD_VULKAN=ON`.
- `neraulgraphics.Smoke`: dust as a volume. Particles are splatted as Gaussian blobs of optical depth, composited with Beer-Lambert absorption
  and hidden by a ray cast when they are behind a surface, so clouds are translucent where spheres were opaque.
- Procedural surface textures (`examples/scenery/textures.py`): brick with per-brick tone and mortar, speckled asphalt, stained concrete and
  stucco, seamless and made in 0.1 s with numpy; textured boxes carry uvs in real-world units. (Tests caught a stray factor of two that had
  drawn every texture at double size.)
- A street scene (`examples/scenery/street.py`): sidewalk, lane markings, parked cars, lamp posts, barriers, flanking buildings with lit
  and dark windows, a skyline, and an overcast atmosphere, laid out around any structure's footprint.
- Physics validation (`tests/test_physics_validation.py`): free fall, projectiles, friction on slopes, restitution, conservation of momentum
  and energy, torque-free spin and no energy gain. It found that Jolt's default body damping (5 % of velocity a second)
  acts as drag in a vacuum; the world now defaults it to zero.
- `Renderer.set_instance_matrices` (C++) and `Renderer.set_instance_transforms` taking a CUDA array (Python): a simulator's poses go to
  the trees over the instances straight from GPU memory, with no copy through the CPU. Handing over 9,000 bodies took 3.2 ms through the
  CPU and 0.4 ms this way.
- A web page with a showcase video and the measured stress tests (`site/`, published to GitHub Pages by `.github/workflows/pages.yml`).
  `tools/showcase/make_video.py` renders the video from the engine itself and writes the measurements it quotes.

### Changed
- `Renderer.set_instance_transforms` checks its matrices without building an error message for each one: 9,000 instances took
  3.0 ms and now take 0.5 ms.

## [0.5.0] - 2026-11-10

### Added
- Beauty mode: path tracing. `Renderer.set_render_mode(RenderMode.PathTraced, PathTracing(...))` draws
  light bouncing between surfaces with soft shadows, sky lighting and physically based materials
  (GGX specular with Lambert diffuse, sampled by visible normals), by unidirectional path tracing with direct
  light sampling and Russian roulette. A sample brighter than `max_radiance` is scaled down against fireflies.
  Both backends, with the same results (the CPU and the GPU agree to noise). Classic materials need no change:
  `Material` gains `roughness`, `metallic` and `emission`, and when roughness or metallic is left unset they
  are worked out from the specular exponent and reflectivity.
- `Environment` and `Renderer.set_environment`: a gradient or an equirectangular HDR map as the sky, which
  rays leaving the scene see in either renderer and path tracing also uses as light. A map is importance
  sampled, combined with material sampling by multiple importance sampling, so a small bright sun lights and
  shadows cleanly. `up` and `rotation` orient it. `load_hdr` reads Radiance `.hdr` files and `procedural_sky`
  makes a sky with a sun of a given strength.
- A denoiser for path-traced pictures: reuse of the previous frame along the camera's motion, with history
  rejected where the surface changed, then an edge-aware spatial filter steered by variance. One path per pixel
  looks clean, and the picture keeps improving while the camera holds still.
- `sim.beauty(sky=None, **settings)` and `sim.classic()` on the Newton and MuJoCo adapters (the sky follows
  the simulator's up axis). The glTF loader passes roughness, metallic and emission through.
- Viewer: `--beauty` and the `B` key.
- `examples/beauty_render.py`, a path-traced scene you can fly around, and `examples/batch_render.py` now opens a window with live worlds. Both fall back to writing a picture when there is no display.

### Fixed
- `examples/batch_render.py` crashed on a GPU machine without PyTorch: it asked for zero-copy output that nothing could read.
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
