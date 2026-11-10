"""The sensor outputs of a robot or RL pipeline: RGB, depth, surface normals, object IDs, and a LiDAR scan.

    python examples/sensor_images.py                     # writes sensors.png
    python examples/sensor_images.py --model duck.glb    # the same for any glTF or OBJ model

Needs: pip install neraulgraphics pillow
"""

import argparse

import numpy as np
from PIL import Image

import neraulgraphics as ng

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="a glTF/GLB/OBJ file to look at instead of the built-in scene")
parser.add_argument("--size", default="480x360")
parser.add_argument("--out", default="sensors.png")
args = parser.parse_args()
width, height = map(int, args.size.split("x"))

if args.model:
    scene = ng.load(args.model)
    view = scene.renderer(width, height)
    renderer, camera = view.renderer, view.camera(view.center + view.extent * np.array([1.4, 0.8, 2.0]))
    origin = view.center + np.array([0.0, 0.0, 2.5 * view.extent])
    reach = 6 * view.extent
else:
    red, green, blue = (ng.Material(diffuse_color=c) for c in [(0.8, 0.2, 0.2), (0.2, 0.7, 0.3), (0.3, 0.4, 0.9)])
    renderer = ng.Renderer(
        width, height, [ng.Sphere((-2.2, 0, -9), 1.5, red), ng.Sphere((2.4, -0.4, -7), 1.1, green)],
        [ng.Light((-6, 9, 3), 1.5), ng.Light((8, 6, 2), 0.6)], ng.Plane(y=-1.5, half_extent_x=30, z_min=-60, z_max=0),
        meshes=[ng.make_box((1, 1, 1)), ng.make_icosphere(1, 3)],
        instances=[ng.Instance(0, ng.make_transform((0, -0.5, -6), (0, 0.4, 0, 0.92), 1.0), blue),
                   ng.Instance(1, ng.make_transform((0.3, 1.2, -11), scale=1.4), green)])
    renderer.ambient = 0.2
    camera = ng.make_camera((0, 1.5, 3), (0, 0, -8), width=width, height=height)
    origin, reach = np.array([0.0, 0.3, -3.0]), 30.0

out = renderer.render_frame(camera, samples_per_frame=4, depth=True, normals=True, object_ids=True)
rgb, depth, normals, ids = out["rgb"], out["depth"], out["normals"], out["object_ids"]

# Depth as grey (near is bright); nothing hit is black.
hit = np.isfinite(depth)
near, far = depth[hit].min(), np.percentile(depth[hit], 98)
depth_img = np.zeros((height, width, 3), np.uint8)
depth_img[hit] = (255 * (1 - np.clip((depth[hit] - near) / (far - near), 0, 1)))[:, None]
# Normals as the usual colour code: x, y, z become red, green, blue.
normal_img = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
# Each object ID gets its own colour.
palette = np.random.default_rng(3).integers(60, 255, size=(ids.max() + 1, 3)).astype(np.uint8)
palette[ng.OBJECT_NONE] = 0
id_img = palette[ids]

# A 360-degree LiDAR sweep: rays around a circle in the horizontal plane, one query for all of them.
angles = np.linspace(0, 2 * np.pi, 720, endpoint=False, dtype=np.float32)
directions = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=1)
scan = renderer.cast_rays(np.tile(origin.astype(np.float32), (len(angles), 1)), directions, max_distance=reach)
lidar = np.full((height, width, 3), 20, np.uint8)
ok = np.isfinite(scan["distance"])
scale = 0.42 * min(width, height) / (min(scan["distance"][ok].max(), reach) if ok.any() else 1)
px = (width / 2 + scale * scan["distance"][ok] * directions[ok, 0]).astype(int)
py = (height / 2 + scale * scan["distance"][ok] * directions[ok, 2]).astype(int)
colours = palette[scan["object_id"][ok]]
for dx in (-1, 0, 1):  # each return as a small square, so the scan is visible
    for dy in (-1, 0, 1):
        x, y = px + dx, py + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        lidar[y[inside], x[inside]] = colours[inside]
lidar[height // 2 - 2:height // 2 + 3, width // 2 - 2:width // 2 + 3] = 255

sheet = np.concatenate([np.concatenate([rgb, depth_img, normal_img], axis=1),
                        np.concatenate([id_img, lidar, np.zeros_like(rgb)], axis=1)], axis=0)
Image.fromarray(sheet).save(args.out)
print(f"wrote {args.out}: rgb, depth, normals / object ids, LiDAR top-down scan ({ok.sum()} of {len(angles)} rays hit)")
