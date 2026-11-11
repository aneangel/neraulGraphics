"""Newton's rigid-body solver, path traced live: spheres, ellipsoids, cylinders, capsules, boxes and cones tumbling onto the ground.

    python examples/newton_shapes.py                       # opens the viewer
    python examples/newton_shapes.py --save shapes.png     # no window: let them fall, write a picture
    python examples/newton_shapes.py --gif shapes.gif      # no window: write the fall as an animated GIF

Newton (XPBD) simulates the bodies on the GPU; neraulgraphics turns each step's poses into instance matrices and draws them with the path
tracer under a physical sky, so glass-like shading, soft shadows and interreflections come with the physics. Drag with the left button to orbit
the scene, the wheel zooms. Space drops another shape (they are used in turn), R puts everything back, [ and ] lower and raise the sun,
B switches between path tracing and the fast classic picture, P saves a picture, Esc quits. Hold still and the picture keeps getting cleaner.

Needs: pip install pillow pyglet, and Newton (https://github.com/newton-physics/newton).
"""

import argparse
import sys
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

import neraulgraphics as ng
from neraulgraphics.newton import NewtonRenderer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import OrbitCamera, open_window, save_picture  # noqa: E402

COPIES = 4                       # of each shape: 6 kinds x 4 = 24 bodies, parked until dropped
KINDS = ["sphere", "ellipsoid", "cylinder", "capsule", "box", "cone"]


def build_model():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    adders = {
        "sphere": lambda b: builder.add_shape_sphere(b, radius=0.5),
        "ellipsoid": lambda b: builder.add_shape_ellipsoid(b, rx=0.5, ry=0.5, rz=0.25),
        "cylinder": lambda b: builder.add_shape_cylinder(b, radius=0.4, half_height=0.6),
        "capsule": lambda b: builder.add_shape_capsule(b, radius=0.3, half_height=0.7),
        "box": lambda b: builder.add_shape_box(b, hx=0.5, hy=0.35, hz=0.25),
        "cone": lambda b: builder.add_shape_cone(b, radius=0.45, half_height=0.6),
    }
    for copy in range(COPIES):
        for i, kind in enumerate(KINDS):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(60.0 + 3.0 * i, 3.0 * copy, 1.0), q=wp.quat_identity()))
            adders[kind](body)
    return builder.finalize()


class Shapes:
    def __init__(self, width, height):
        self.model = build_model()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=6)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.view = NewtonRenderer(self.model, width, height)
        self.sun_elevation, self.sun_azimuth = 0.6, 2.2
        self.beauty = True
        self.frame = 0
        self.next_body = 0
        self.dropped = np.zeros(self.model.body_count, bool)
        self.initial_q, self.initial_qd = self.state_0.body_q.numpy().copy(), self.state_0.body_qd.numpy().copy()
        self.rng = np.random.default_rng(1)
        self.set_mode()

    def set_mode(self):
        if self.beauty:
            sky = ng.physical_sky(sun_elevation=self.sun_elevation, sun_azimuth=self.sun_azimuth, turbidity=2.6, sun_strength=4.0, sun_radius=0.02,
                                  sky_ratio=0.22, width=768, height=384)
            self.view.beauty(sky, bounces=6, exposure=0.8, denoise=True, denoise_iterations=3, max_radiance=60.0)
        else:
            self.view.classic()
        self.frame = 0

    def step(self):
        """One 1/120 s frame: four solver substeps."""
        for _ in range(4):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, 1 / 120 / 4)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def drop(self):
        """Drops the next body from above the middle, tumbling."""
        i = self.next_body % self.model.body_count
        self.next_body += 1
        q, qd = self.state_0.body_q.numpy().copy(), self.state_0.body_qd.numpy().copy()
        q[i] = (*(self.rng.uniform(-1.2, 1.2, 2)), 3.5 + self.rng.uniform(0, 2.0), *self._random_rotation())
        qd[i] = (*(self.rng.uniform(-1, 1, 2)), 0.0, *(self.rng.normal(size=3) * 2.5))
        self.state_0.body_q.assign(q)
        self.state_0.body_qd.assign(qd)
        self.dropped[i] = True

    def _random_rotation(self):
        v = self.rng.normal(size=4)
        v /= np.linalg.norm(v)
        return v

    def reset(self):
        self.state_0.body_q.assign(self.initial_q)
        self.state_0.body_qd.assign(self.initial_qd)
        self.next_body = 0
        self.dropped[:] = False

    def moving(self):
        """Whether any body that has been dropped is still moving, so the picture should start refining again."""
        if not self.dropped.any():
            return False
        return float(np.abs(self.state_0.body_qd.numpy()[self.dropped]).max()) > 0.05

    def render(self, camera):
        image = self.view.render(self.state_0, camera, frame=self.frame, samples_per_frame=4 if (self.beauty and self.view.renderer.device == "cuda") else 1)
        self.frame += 1
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--save", metavar="FILE", help="write a picture to FILE (a PNG); with no window, instead of showing one")
    parser.add_argument("--gif", metavar="FILE", help="with no window, write the fall as an animated GIF")
    parser.add_argument("--frames", type=int, default=96, help="frames to refine for when saving without a window")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    shapes = Shapes(width, height)
    orbit = OrbitCamera((0.0, 0.0, 0.8), 9.0, yaw=0.9, pitch=0.3, up=(0.0, 0.0, 1.0))
    window = open_window("neraulGraphics: Newton shapes", width, height)

    if window is None:
        frames = []
        for step in range(480):
            if step % 24 == 0 and shapes.next_body < 10:
                shapes.drop()
            shapes.step()
            if args.gif and step % 6 == 0:
                shapes.frame = 0
                frames.append(shapes.render(orbit.camera(width, height)))
        shapes.frame = 0
        for _ in range(args.frames):
            picture = shapes.render(orbit.camera(width, height))
        if args.gif:
            from PIL import Image

            images = [Image.fromarray(f) for f in frames]
            images[0].save(args.gif, save_all=True, append_images=images[1:], duration=40, loop=0)
            print("wrote", args.gif)
        save_picture(args.save or "newton_shapes.png", picture)
        return

    window.mouse_mode = "orbit"
    started, last, fps, owed, next_drop = time.perf_counter(), time.perf_counter(), 0.0, 0.0, 0.3
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            dt = window.tick()
            t = time.perf_counter() - started
            orbit.update(window, dt)
            for key in window.take_keys():
                if key == "space":
                    shapes.drop()
                elif key == "r":
                    shapes.reset()
                elif key in ("bracketleft", "bracketright"):
                    shapes.sun_elevation = float(np.clip(shapes.sun_elevation + (0.06 if key == "bracketright" else -0.06), 0.03, 1.5))
                    shapes.set_mode()
                elif key in ("comma", "period"):
                    shapes.sun_azimuth += 0.2 if key == "period" else -0.2
                    shapes.set_mode()
                elif key == "b":
                    shapes.beauty = not shapes.beauty
                    shapes.set_mode()
                elif key == "p":
                    window.screenshot("newton_shapes.png")
            if t >= next_drop and shapes.next_body < 6:
                shapes.drop()
                next_drop = t + 0.5
            owed += dt
            steps = 0
            while owed >= 1 / 120 and steps < 4:
                shapes.step()
                owed -= 1 / 120
                steps += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-4)
            last = now
            hud = (f"{fps:4.0f} fps   {shapes.frame} frames refining   {'path traced' if shapes.beauty else 'classic'}   Newton XPBD on {wp.get_device()}\n"
                   "drag orbit   wheel zoom   Space drop   R reset   [ ] , . sun   B beauty/classic   P picture   Esc quit")
            window.show(shapes.render(orbit.camera(width, height)), hud)
    if args.save:
        save_picture(args.save, shapes.render(orbit.camera(width, height)))


if __name__ == "__main__":
    main()
