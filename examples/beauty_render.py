"""A path-traced picture: soft light, bounces, and a sky with a sun.

    python examples/beauty_render.py                    # writes beauty.png
    python examples/beauty_render.py --hdri sky.hdr     # light it with your own HDR sky

Needs: pip install neraulgraphics pillow  (an NVIDIA GPU makes it fast; without one it uses the CPU)
"""

import argparse
import time

import numpy as np
from PIL import Image

import neraulgraphics as ng

parser = argparse.ArgumentParser()
parser.add_argument("--hdri", help="an equirectangular Radiance .hdr sky; the default is a procedural sky with a sun")
parser.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--frames", type=int, default=32, help="more frames, a cleaner picture")
parser.add_argument("--out", default="beauty.png")
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
renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=5))

camera = ng.make_camera((0, 0.8, 2), (0, -0.6, -7), width=width, height=height, fov=1.0)
start = time.perf_counter()
for frame in range(args.frames):  # the denoiser reuses earlier frames, so each is one path per pixel
    image = renderer.render(camera, frame)
print(f"{args.frames} frames on {renderer.device} in {time.perf_counter() - start:.2f} s")
Image.fromarray(np.asarray(image)).save(args.out)
print("wrote", args.out)
