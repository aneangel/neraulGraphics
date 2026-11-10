"""Render a MuJoCo model with neraulgraphics.

    python examples/mujoco_humanoid.py path/to/model.xml          # writes mujoco.gif

Needs: pip install neraulgraphics mujoco pillow
"""

import argparse

import mujoco
from PIL import Image

from neraulgraphics.mujoco import MuJoCoRenderer

parser = argparse.ArgumentParser()
parser.add_argument("model", help="a MuJoCo XML model")
parser.add_argument("--frames", type=int, default=90)
parser.add_argument("--size", default="640x360")
parser.add_argument("--out", default="mujoco.gif")
args = parser.parse_args()
width, height = map(int, args.size.split("x"))

model = mujoco.MjModel.from_xml_path(args.model)
data = mujoco.MjData(model)
view = MuJoCoRenderer(model, width, height)    # geoms become mesh instances once, here

frames = []
for _ in range(args.frames):
    for _ in range(int(1 / 30 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    frames.append(Image.fromarray(view.render(data)))   # only poses are updated per frame

frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=33, loop=0)
print(f"wrote {args.out} ({len(frames)} frames)")
