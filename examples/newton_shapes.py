"""Drop one of each shape in Newton and render it with neraulgraphics.

    python examples/newton_shapes.py            # writes newton_shapes.gif
    python examples/newton_shapes.py --frames 90 --size 960x540

Needs: pip install neraulgraphics pillow, and Newton.
"""

import argparse

import newton
import warp as wp
from PIL import Image

from neraulgraphics.newton import NewtonRenderer

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=90)
parser.add_argument("--size", default="640x360")
parser.add_argument("--out", default="newton_shapes.gif")
args = parser.parse_args()
width, height = map(int, args.size.split("x"))

builder = newton.ModelBuilder()
builder.add_ground_plane()
shapes = {
    "sphere": lambda b: builder.add_shape_sphere(b, radius=0.5),
    "ellipsoid": lambda b: builder.add_shape_ellipsoid(b, rx=0.5, ry=0.5, rz=0.25),
    "cylinder": lambda b: builder.add_shape_cylinder(b, radius=0.4, half_height=0.6),
    "capsule": lambda b: builder.add_shape_capsule(b, radius=0.3, half_height=0.7),
    "box": lambda b: builder.add_shape_box(b, hx=0.5, hy=0.35, hz=0.25),
    "cone": lambda b: builder.add_shape_cone(b, radius=0.45, half_height=0.6),
}
for i, add in enumerate(shapes.values()):
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -3.0 + 1.2 * i, 2.0 + 0.3 * i), q=wp.quat_identity()))
    add(body)
model = builder.finalize()

solver = newton.solvers.SolverXPBD(model, iterations=5)
state_0, state_1 = model.state(), model.state()
control = model.control()
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()

view = NewtonRenderer(model, width, height)   # shapes become mesh instances once, here
camera = view.camera((8.0, 0.0, 2.5), (0.0, 0.0, 0.8))

frames = []
for _ in range(args.frames):
    for _ in range(5):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1 / 100 / 5)
        state_0, state_1 = state_1, state_0
    frames.append(Image.fromarray(view.render(state_0, camera)))   # only poses are updated per frame

frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=20, loop=0)
print(f"wrote {args.out} ({len(frames)} frames)")
