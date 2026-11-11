"""A path-traced scene in a window you can fly around: soft light, bounces, and a sky with a sun.

    python examples/beauty_render.py                    # opens the viewer
    python examples/beauty_render.py --hdri sky.hdr     # light it with your own HDR sky
    python examples/beauty_render.py --save beauty.png  # also (or, with no display, instead) write a picture

Drag with the left mouse button to look, W A S D to move, Q and E down and up, shift to go faster,
B to switch between beauty and the fast classic picture, P to save a picture, Esc to quit.
Hold still and the picture keeps getting cleaner.

Needs: pip install neraulgraphics pillow  (an NVIDIA GPU makes it fast; without one it uses the CPU)
"""

import argparse
import math
import time

import numpy as np

import neraulgraphics as ng
from _viewer import Viewer

parser = argparse.ArgumentParser()
parser.add_argument("--hdri", help="an equirectangular Radiance .hdr sky; the default is a procedural sky with a sun")
parser.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--frames", type=int, default=32, help="frames to refine for when saving without a window")
parser.add_argument("--save", metavar="FILE", help="write a picture to FILE (a PNG)")
parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
args = parser.parse_args()
width, height = args.size

# Materials: classic ones work as they are; these use the physically based parameters.
clay = ng.Material(diffuse_color=(0.8, 0.35, 0.25), roughness=0.6, metallic=0.0)
gold = ng.Material(diffuse_color=(1.0, 0.76, 0.34), roughness=0.15, metallic=1.0)
glass_ish = ng.Material(diffuse_color=(0.9, 0.95, 1.0), roughness=0.02, metallic=1.0)
lamp = ng.Material(diffuse_color=(0, 0, 0), emission=(6.0, 4.5, 3.0))

renderer = ng.Renderer(
    width, height,
    spheres=[ng.Sphere((-2.2, -0.5, -7), 1.5, clay), ng.Sphere((1.0, -0.8, -5), 1.2, gold),
             ng.Sphere((3.2, -1.2, -6.5), 0.8, glass_ish), ng.Sphere((-0.3, 3.0, -9), 0.4, lamp)],
    lights=[],  # the sun in the sky is the light
    floor=ng.Plane(y=-2, half_extent_x=40, z_min=-80, z_max=8, checker_scale=1.0),
)
sky = ng.load_hdr(args.hdri) if args.hdri else ng.procedural_sky(sun_elevation=0.7, sun_azimuth=2.2, sun_strength=3.0)
renderer.set_environment(ng.Environment(map=sky))
beauty = ng.PathTracing(bounces=5)
renderer.set_render_mode(ng.RenderMode.PathTraced, beauty)

position = np.array([0.0, 0.8, 2.0])
yaw, pitch = 0.0, -0.1  # yaw 0 looks down -z, like the scene's layout


def camera():
    forward = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])
    return ng.make_camera(tuple(position), tuple(position + forward), width=width, height=height, fov=1.0), forward


def save(path, image):
    from PIL import Image
    Image.fromarray(np.asarray(image)).save(path)
    print("wrote", path)


try:
    viewer = Viewer("neraulGraphics beauty", width, height)
except RuntimeError as problem:
    # No window available: refine for a while and write a picture instead.
    print(problem, "- writing a picture instead")
    cam, _ = camera()
    start = time.perf_counter()
    for frame in range(args.frames):  # the denoiser reuses earlier frames, so each is one path per pixel
        image = renderer.render(cam, frame)
    print(f"{args.frames} frames on {renderer.device} in {time.perf_counter() - start:.2f} s")
    save(args.save or "beauty.png", image)
    raise SystemExit

print("drag left mouse to look, WASD to move, Q/E down/up, shift to sprint, B beauty/classic, P to save, Esc to quit")
state = {"beauty": True, "frame": 0, "save": None}


def on_key(key):
    if key == "escape":
        viewer.close()
    elif key == "b":
        state["beauty"] = not state["beauty"]
        if state["beauty"]:
            renderer.set_render_mode(ng.RenderMode.PathTraced, beauty)
        else:
            renderer.set_render_mode(ng.RenderMode.Classic)
        state["frame"] = 0  # a new mode starts a fresh picture
    elif key == "p":
        state["save"] = args.save or "screenshot.png"


viewer.on_key = on_key
last, started = time.perf_counter(), time.perf_counter()
image = None
while viewer.open and not (args.seconds and time.perf_counter() - started > args.seconds):
    now = time.perf_counter()
    dt, last = min(now - last, 0.1), now
    dx, dy = viewer.take_look()
    yaw += dx * 0.004
    pitch = float(np.clip(pitch - dy * 0.004, -1.5, 1.5))
    cam, forward = camera()
    right = np.cross(forward, (0, 1, 0))
    right /= np.linalg.norm(right)
    speed = (24.0 if viewer.held("shift_l") or viewer.held("shift_r") else 8.0) * dt
    for key, direction in (("w", forward), ("s", -forward), ("d", right), ("a", -right),
                           ("e", np.array([0, 1.0, 0])), ("q", np.array([0, -1.0, 0]))):
        if viewer.held(key):
            position = position + direction * speed
    cam, _ = camera()

    # A path-traced frame carries its history along the camera's motion, so the frame number only restarts with a new
    # mode. The classic picture averages while still, so any movement restarts it.
    moved = bool(dx or dy or any(viewer.held(k) for k in "wasdqe"))
    if not state["beauty"] and moved:
        state["frame"] = 0
    image = renderer.render(cam, state["frame"])
    state["frame"] += 1
    viewer.show(image, f"{'beauty' if state['beauty'] else 'classic'}  {state['frame']} frames  on {renderer.device}")
    if state["save"]:
        save(state["save"], image)
        state["save"] = None

viewer.close()
if args.save and image is not None:
    save(args.save, image)
