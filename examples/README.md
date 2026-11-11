# Examples

Every example opens a **live window** (a free-fly or orbit camera, the mouse and the keyboard), shows the renderer's **fidelity** (path tracing,
a physical sky, glass, metals, textured and normal-mapped surfaces) and its **physics** (bodies you can throw, drop, knock over). With no display
they write a picture or a GIF instead, so they also run on a server.

```bash
python examples/run.py                  # the list
python examples/run.py materials        # run one
```

| | what it shows | physics |
|---|---|---|
| `materials.py` | glass that refracts and tints its shadow, gold, brushed steel, plastic, clay under a physical sky with a sun you can move | Jolt (deterministic): throw balls and boxes, they tumble and settle |
| `view_model.py` | a glTF, GLB or OBJ model (or a built-in torus knot) orbited in path-traced light | press D: the model drops as a rigid body |
| `sensor_images.py` | RGB, depth, normals, object IDs and a LiDAR sweep, six live panels | Jolt: bodies keep falling and the panels change |
| `newton_shapes.py` | Newton's shapes, path traced | Newton XPBD on the GPU |
| `mujoco_humanoid.py` | a MuJoCo humanoid ragdoll, path traced | MuJoCo: joint limits, motors, thrown balls |
| `batch_render.py` | thousands of worlds rendered at once | GPU physics for every world: gravity, bounce, friction |

Common keys: **click** the window to look around with the mouse (Esc lets go, a second Esc quits) or **drag** to orbit (view_model, newton,
mujoco), **W A S D / Q E** to move, **Space** to throw or drop something, **R** to reset, **[ ]** and **, .** to move the sun, **B** to switch
between path tracing and the fast classic picture, **P** to save a picture. The text in the corner of each window lists that example's keys.

Why the picture stays steady while you move: the sun is an explicit light, so highlights are computed and not found by chance; the denoiser keeps and
reprojects its history through camera motion (and trusts new frames more while the camera moves); and the examples only restart the picture when the light changes,
never when something moves.

Hold still and the picture keeps getting cleaner: each frame adds paths for every pixel, and the denoiser reuses earlier frames while nothing moves.

## What you need

`pip install neraulgraphics pillow pyglet` is enough for the renderer. The examples that show physics through Jolt (`materials`, `view_model`'s drop
and `sensors`) also need the package built with the deterministic physics module, which fetches Jolt:

```bash
pip install -C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON .        # from a checkout
```

`newton` needs [Newton](https://github.com/newton-physics/newton), `mujoco` needs `pip install mujoco`, and `batch` needs PyTorch for its GPU path.
An NVIDIA GPU makes everything much faster; without one the CPU is used.

## How they are put together

- `_live.py`: the window (pyglet, frames uploaded as a GL texture), the fly and orbit cameras, and the headless fallback.
- `_stage.py`: a physics stage (Jolt bodies drawn as renderer instances, with the Y-up/Z-up conversion done once), and `Scene`: a ground, a sky,
  a stage and the loop that steps the physics and renders.
- `scenery/`: procedural textures (`textures.py`), a street laid out around a structure (`street.py`) and walking people (`characters.py`).
