"""Render a glTF, GLB or OBJ model: a still image, or a turntable GIF.

    python examples/view_model.py model.glb                  # writes ./model.png
    python examples/view_model.py model.glb --turntable      # writes ./model.gif, one slow orbit

Needs: pip install neraulgraphics pillow
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import neraulgraphics as ng

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--size", default="800x600")
parser.add_argument("--turntable", action="store_true")
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--samples", type=int, default=8, help="samples per pixel; more is smoother")
args = parser.parse_args()
width, height = map(int, args.size.split("x"))

scene = ng.load(args.model)
view = scene.renderer(width, height)
out = Path(Path(args.model).stem + (".gif" if args.turntable else ".png"))  # in the current directory
print(f"{len(scene.meshes)} meshes, {len(scene.instances)} instances, {len(scene.textures)} textures")

up = np.asarray(scene.up)
side = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0])
side /= np.linalg.norm(side)
across = np.cross(up, side)


def camera_at(angle):
    """On a circle around the model, a little above its middle, looking at it."""
    offset = 2.4 * view.extent * (np.cos(angle) * side + np.sin(angle) * across) + 0.6 * view.extent * up
    return view.camera(view.center + offset)


if args.turntable:
    frames = [Image.fromarray(view.render(camera_at(2 * np.pi * i / args.frames), 0, args.samples))
              for i in range(args.frames)]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0)
else:
    Image.fromarray(view.render(camera_at(0.6), 0, args.samples)).save(out)
print(f"wrote {out}")
