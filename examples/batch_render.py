"""Thousands of worlds, simulated and rendered at once, entirely on the GPU: poses in, images out, no copy through the CPU.

    python examples/batch_render.py                      # 1024 worlds: a speed test, then a window with 16 of them live
    python examples/batch_render.py --worlds 4096 --size 128
    python examples/batch_render.py --save batch.png     # no window: write the mosaic of the first 16 worlds

Every world holds a few bodies that are thrown, fall under gravity, bounce with energy loss and friction, come to rest and are thrown again,
simulated for all worlds in one array operation on the GPU (PyTorch) where the renderer reads the poses in place. Each frame all worlds are
rendered, with colour, depth and object IDs: the window shows the first 16 as a mosaic, and the speed test reports how many images a second that is.
This is the loop of a reinforcement-learning environment, minus the learning.

Space throws everything again, P saves the mosaic, Esc quits.

Needs: pip install pillow pyglet torch  (an NVIDIA GPU; without one it falls back to NumPy on the CPU, much more slowly)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import neraulgraphics as ng
from neraulgraphics.shapes import look_at

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import open_window, save_picture  # noqa: E402

FLOOR_Y, RADIUS, GRAVITY, RESTITUTION = -1.5, 0.5, 9.81, 0.55
PER_WORLD = 4
COLOURS = [(0.8, 0.25, 0.25), (0.25, 0.7, 0.35), (0.3, 0.4, 0.9), (0.9, 0.75, 0.2)]


class Bouncers:
    """PER_WORLD bodies in each world, as arrays over (worlds, bodies): position, velocity and spin, stepped with one array operation.

    A body falls under gravity; when it reaches the floor its vertical velocity is reversed and scaled by RESTITUTION and it loses a tenth of its
    sideways speed to friction; when it stops bouncing it rests, and after a while it is thrown again from a new place. xp is torch or numpy."""

    def __init__(self, worlds, xp, device=None, seed=0):
        self.xp, self.device, self.worlds = xp, device, worlds
        self.generator = xp.Generator(device=device).manual_seed(seed) if xp.__name__ == "torch" else xp.random.default_rng(seed)
        shape = (worlds, PER_WORLD)
        self.pos = self._zeros(shape + (3,))
        self.vel = self._zeros(shape + (3,))
        self.angle = self._zeros(shape)
        self.spin = self._zeros(shape)
        self.rest = self._zeros(shape)               # seconds spent at rest
        self.throw(self._ones_bool(shape))

    # array helpers that differ between torch and numpy
    def _zeros(self, shape):
        return self.xp.zeros(shape, device=self.device) if self.device else self.xp.zeros(shape, np.float32)

    def _ones_bool(self, shape):
        return self.xp.ones(shape, dtype=self.xp.bool, device=self.device) if self.device else self.xp.ones(shape, bool)

    def _uniform(self, shape, lo, hi):
        if self.device:
            return lo + (hi - lo) * self.xp.rand(shape, device=self.device, generator=self.generator)
        return (lo + (hi - lo) * self.generator.random(shape)).astype(np.float32)

    def throw(self, which):
        """Throws the bodies selected by the boolean array which: from a random place above the floor, with a random velocity and spin."""
        xp = self.xp
        shape = (self.worlds, PER_WORLD)
        new_pos = xp.stack([self._uniform(shape, -2.4, 2.4), self._uniform(shape, 0.0, 3.0), self._uniform(shape, -8.0, -3.0)], -1)
        new_vel = xp.stack([self._uniform(shape, -1.5, 1.5), self._uniform(shape, 0.0, 4.0), self._uniform(shape, -1.0, 1.0)], -1)
        mask = which[..., None]
        self.pos = xp.where(mask, new_pos, self.pos)
        self.vel = xp.where(mask, new_vel, self.vel)
        self.spin = xp.where(which, self._uniform(shape, -4.0, 4.0), self.spin)
        self.rest = xp.where(which, self.rest * 0, self.rest)

    def step(self, dt):
        xp = self.xp
        self.vel[..., 1] -= GRAVITY * dt
        self.pos = self.pos + self.vel * dt
        self.angle = self.angle + self.spin * dt
        landed = self.pos[..., 1] < FLOOR_Y + RADIUS
        if bool(landed.any()):
            self.pos[..., 1] = xp.where(landed, FLOOR_Y + RADIUS, self.pos[..., 1])
            bounce = landed & (self.vel[..., 1] < 0)
            self.vel[..., 1] = xp.where(bounce, -self.vel[..., 1] * RESTITUTION, self.vel[..., 1])
            for axis in (0, 2):
                self.vel[..., axis] = xp.where(bounce, self.vel[..., axis] * 0.9, self.vel[..., axis])
            slow = landed & (xp.abs(self.vel[..., 1]) < 0.4)             # no more bounces: it rests
            self.vel[..., 1] = xp.where(slow, self.vel[..., 1] * 0, self.vel[..., 1])
            for axis in (0, 2):
                self.vel[..., axis] = xp.where(slow, self.vel[..., axis] * (1 - 3 * dt), self.vel[..., axis])
            self.spin = xp.where(slow, self.spin * (1 - 2 * dt), self.spin)
            self.rest = xp.where(slow, self.rest + dt, self.rest * 0)
        self.throw(self.rest > 1.5)

    def matrices(self):
        """(worlds, bodies, 3, 4): each body's pose, turned about the vertical by its angle."""
        xp = self.xp
        c, s = xp.cos(self.angle), xp.sin(self.angle)
        m = self._zeros(self.pos.shape[:2] + (3, 4))
        m[..., 0, 0], m[..., 0, 2], m[..., 1, 1], m[..., 2, 0], m[..., 2, 2] = c, s, 1.0, -s, c
        m[..., :3, 3] = self.pos
        return m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--size", type=int, default=192, help="each view is size x size pixels (64 is the size policies usually take, and the fastest)")
    parser.add_argument("--steps", type=int, default=50, help="steps of the speed test")
    parser.add_argument("--save", metavar="FILE", help="write the mosaic of the first 16 worlds to FILE (a PNG)")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    args = parser.parse_args()

    try:
        import torch

        on_gpu = torch.cuda.is_available() and ng.cuda_available()
    except ImportError:
        torch, on_gpu = None, False

    meshes = [ng.make_box((0.5, 0.5, 0.5)), ng.make_icosphere(0.5, 2)]
    renderer = ng.BatchRenderer(
        args.size, args.size, args.worlds, lights=[ng.Light((-6, 9, 4), 1.4), ng.Light((8, 6, 2), 0.5)],
        floor=ng.Plane(y=FLOOR_Y, half_extent_x=20, z_min=-40, z_max=5), meshes=meshes,
        instances=[ng.Instance(i % 2, material=ng.Material(diffuse_color=COLOURS[i])) for i in range(PER_WORLD)], device="auto")
    renderer.ambient = 0.2
    print(f"{args.worlds} worlds of {PER_WORLD} bodies, {args.size}x{args.size} each, on the {renderer.device}")

    camera = look_at((0.0, 1.2, 2.0), (0.0, -0.5, -5.0), up=(0, 1, 0))          # one camera per world, all alike
    if on_gpu:
        cameras = torch.from_numpy(np.tile(camera, (args.worlds, 1, 1))).cuda()
        bodies = Bouncers(args.worlds, torch, "cuda")
    else:
        cameras = np.tile(camera, (args.worlds, 1, 1))
        bodies = Bouncers(args.worlds, np)

    def step(dt=1 / 60):
        """Physics for every world, then every world's pictures: poses never leave the device they were made on."""
        bodies.step(dt)
        renderer.set_instance_transforms(bodies.matrices())
        # Views of the GPU's own buffers when PyTorch is there to take them; ordinary NumPy copies otherwise.
        return renderer.render(cameras, fov=1.0, rgb=True, depth=True, object_ids=True, zero_copy=on_gpu)

    step()                                                                         # warm up
    start = time.perf_counter()
    for _ in range(args.steps):
        out = step()
    if on_gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{args.steps} steps of (physics for {args.worlds * PER_WORLD:,} bodies + {args.worlds} RGB, depth and ID images): "
          f"{1000 * elapsed / args.steps:.2f} ms each, {args.worlds * args.steps / elapsed:,.0f} images per second")
    if on_gpu:
        depth, ids = torch.from_dlpack(out["depth"]), torch.from_dlpack(out["object_ids"])
        nearest = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e9)).flatten(1).min(dim=1).values
        print(f"computed on the GPU, with no copy: the nearest surface in each world is {float(nearest.mean()):.2f} away on average;"
              f" {float((ids >= ng.OBJECT_INSTANCE).float().mean()) * 100:.0f}% of pixels see a body")

    def mosaic(frame):
        """The first 16 worlds' pictures as a 4 x 4 grid, on the CPU (only these few are copied back)."""
        rgb = torch.from_dlpack(frame["rgb"])[:16].cpu().numpy() if on_gpu else frame["rgb"][:16]
        return np.concatenate([np.concatenate(list(rgb[r * 4:(r + 1) * 4]), axis=1) for r in range(4)], axis=0)

    window = open_window(f"neraulGraphics: {args.worlds} worlds", 4 * args.size, 4 * args.size, scale=max(1, 768 // (4 * args.size)))
    if window is None:
        save_picture(args.save or "batch.png", mosaic(out))
        return
    started, last, fps = time.perf_counter(), time.perf_counter(), 0.0
    picture = mosaic(out)
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            window.tick()
            for key in window.take_keys():
                if key == "space":
                    bodies.throw(bodies._ones_bool((args.worlds, PER_WORLD)))
                elif key == "p":
                    window.screenshot("batch.png")
            picture = mosaic(step())
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-4)
            last = now
            window.show(picture, f"all {args.worlds} worlds simulated and rendered every frame, on the {renderer.device}: {fps * args.worlds:,.0f} images a second\n"
                                 "Space throw everything   P picture   Esc quit")
    if args.save:
        save_picture(args.save, picture)


if __name__ == "__main__":
    main()
