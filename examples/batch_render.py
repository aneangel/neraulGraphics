"""Render thousands of worlds at once, entirely on the GPU: poses in, images out, no copy through the CPU.

    python examples/batch_render.py                      # 1024 worlds: a speed test, then a window with 16 of them live
    python examples/batch_render.py --worlds 4096 --size 128
    python examples/batch_render.py --save batch.png     # also (or, with no display, instead) write a picture

The window shows the first 16 worlds as a mosaic while all of them are rendered every frame. Close it to quit.

Needs: pip install neraulgraphics pillow torch  (an NVIDIA GPU; without one it falls back to NumPy on the CPU)
"""

import argparse
import time

import numpy as np

from _viewer import Viewer

import neraulgraphics as ng
from neraulgraphics.shapes import look_at

parser = argparse.ArgumentParser()
parser.add_argument("--worlds", type=int, default=1024)
parser.add_argument("--size", type=int, default=192, help="each view is size x size pixels (64 is the size policies usually take, and the fastest)")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--save", metavar="FILE", help="write the mosaic of the first 16 worlds to FILE (a PNG)")
parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
args = parser.parse_args()

try:
    import torch
    on_gpu = torch.cuda.is_available() and ng.cuda_available()
except ImportError:
    on_gpu = False

colors = [(0.8, 0.25, 0.25), (0.25, 0.7, 0.35), (0.3, 0.4, 0.9), (0.9, 0.75, 0.2)]
meshes = [ng.make_box((0.5, 0.5, 0.5)), ng.make_icosphere(0.5, 2)]
per_world = 4  # a few objects per world, the same in every one
renderer = ng.BatchRenderer(
    args.size, args.size, args.worlds, lights=[ng.Light((-6, 9, 4), 1.4), ng.Light((8, 6, 2), 0.5)],
    floor=ng.Plane(y=-1.5, half_extent_x=20, z_min=-40, z_max=5), meshes=meshes,
    instances=[ng.Instance(i % 2, material=ng.Material(diffuse_color=colors[i])) for i in range(per_world)],
    device="auto")
renderer.ambient = 0.2
print(f"{args.worlds} worlds of {per_world} objects, {args.size}x{args.size} each, on the {renderer.device}")

camera = look_at((0.0, 1.2, 2.0), (0.0, 0.0, -5.0), up=(0, 1, 0))      # one camera per world, all alike
if on_gpu:
    cameras = torch.from_numpy(np.tile(camera, (args.worlds, 1, 1))).cuda()
    gen = torch.Generator(device="cuda").manual_seed(0)

    def new_poses():
        """Random poses, made on the GPU where the simulator would keep them: (worlds, objects, 3, 4)."""
        m = torch.zeros(args.worlds, per_world, 3, 4, device="cuda")
        angle = torch.rand(args.worlds, per_world, device="cuda", generator=gen) * 6.28
        c, s = torch.cos(angle), torch.sin(angle)
        m[..., 0, 0], m[..., 0, 2], m[..., 1, 1], m[..., 2, 0], m[..., 2, 2] = c, s, 1.0, -s, c
        m[..., 0, 3] = (torch.rand(args.worlds, per_world, device="cuda", generator=gen) - 0.5) * 5
        m[..., 1, 3] = -1.0 + torch.rand(args.worlds, per_world, device="cuda", generator=gen)
        m[..., 2, 3] = -3.0 - torch.rand(args.worlds, per_world, device="cuda", generator=gen) * 5
        return m

    base = new_poses()
    phase = torch.rand(args.worlds, per_world, device="cuda", generator=gen) * 6.28

    def moving(t):
        """The same objects spinning and bobbing, as of time t."""
        m = base.clone()
        c, s = torch.cos(t * 0.8 + phase), torch.sin(t * 0.8 + phase)
        m[..., 0, 0], m[..., 0, 2], m[..., 2, 0], m[..., 2, 2] = c, s, -s, c
        m[..., 1, 3] = base[..., 1, 3] + 0.4 * torch.sin(t * 1.5 + phase)
        return m
else:
    cameras = np.tile(camera, (args.worlds, 1, 1))
    rng = np.random.default_rng(0)

    def new_poses():
        m = np.zeros((args.worlds, per_world, 3, 4), np.float32)
        m[..., 0, 0] = m[..., 1, 1] = m[..., 2, 2] = 1.0
        m[..., 0, 3] = (rng.random((args.worlds, per_world)) - 0.5) * 5
        m[..., 1, 3] = -1.0 + rng.random((args.worlds, per_world))
        m[..., 2, 3] = -3.0 - rng.random((args.worlds, per_world)) * 5
        return m

    base = new_poses()
    phase = rng.random((args.worlds, per_world)).astype(np.float32) * 6.28

    def moving(t):
        """The same objects spinning and bobbing, as of time t."""
        m = base.copy()
        c, s = np.cos(t * 0.8 + phase), np.sin(t * 0.8 + phase)
        m[..., 0, 0], m[..., 0, 2], m[..., 2, 0], m[..., 2, 2] = c, s, -s, c
        m[..., 1, 3] = base[..., 1, 3] + 0.4 * np.sin(t * 1.5 + phase)
        return m


def step(poses=None):
    renderer.set_instance_transforms(new_poses() if poses is None else poses)
    # Views of the GPU's own buffers when PyTorch is there to take them; ordinary NumPy copies otherwise.
    return renderer.render(cameras, fov=1.0, rgb=True, depth=True, object_ids=True, zero_copy=on_gpu)


step()  # warm up
start = time.perf_counter()
for _ in range(args.steps):
    out = step()
if on_gpu:
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"{args.steps} steps of (new poses + {args.worlds} RGB, depth and ID images): "
      f"{1000 * elapsed / args.steps:.2f} ms each, {args.worlds * args.steps / elapsed:,.0f} images per second")

if on_gpu:
    depth, ids = torch.from_dlpack(out["depth"]), torch.from_dlpack(out["object_ids"])
    nearest = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e9)).flatten(1).min(dim=1).values
    print(f"computed on the GPU, with no copy: the nearest surface in each world is {float(nearest.mean()):.2f} away on average;"
          f" {float((ids >= ng.OBJECT_INSTANCE).float().mean()) * 100:.0f}% of pixels see an object")


def mosaic(frame):
    """The first 16 worlds' pictures as a 4 x 4 grid, on the CPU (only these few are copied back)."""
    rgb = torch.from_dlpack(frame["rgb"])[:16].cpu().numpy() if on_gpu else frame["rgb"][:16]
    return np.concatenate([np.concatenate(list(rgb[r * 4:(r + 1) * 4]), axis=1) for r in range(4)], axis=0)


def save(path, image):
    from PIL import Image
    Image.fromarray(image).save(path)
    print("wrote", path, "(the first 16 worlds)")


try:
    viewer = Viewer(f"neraulGraphics: {args.worlds} worlds", 4 * args.size, 4 * args.size, scale=max(1, 768 // (4 * args.size)))
except RuntimeError as problem:
    print(problem, "- writing a picture instead")
    save(args.save or "batch.png", mosaic(out))
    raise SystemExit

started = time.perf_counter()
picture = mosaic(out)
while viewer.open and not (args.seconds and time.perf_counter() - started > args.seconds):
    t = time.perf_counter() - started
    picture = mosaic(step(moving(torch.tensor(t, device="cuda") if on_gpu else t)))
    viewer.show(picture, f"all {args.worlds} worlds rendered every frame, on the {renderer.device}")
viewer.close()
if args.save:
    save(args.save, picture)
