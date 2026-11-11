"""What a robot sees: RGB, depth, surface normals, object IDs, a LiDAR scan and the bodies' speeds, live, of a scene that physics keeps rearranging.

    python examples/sensor_images.py                     # opens the viewer
    python examples/sensor_images.py --save sensors.png  # no window: drop a few things, write the six panels as one picture

Six panels, the first four produced by the renderer in one pass each frame: the path-traced picture, depth (near is bright), normals (x, y, z as
red, green, blue) and object IDs (a colour each); then a top-down view of a 360-degree LiDAR sweep (one batch of 720 rays at ankle height), and a
bar for the speed of each body. Bodies fall, bounce and settle under Jolt's deterministic physics, so every panel changes as they do.

Click the window to look around with the mouse (Esc lets go, a second Esc quits). W A S D move, Q and E go down and up. Space drops another body in
front of you, R clears the floor, P saves the six panels as a picture.

Needs: pip install pillow pyglet, and neraulgraphics built with the physics module (pip install -C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON .).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import neraulgraphics as ng

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import FlyCamera, open_window, save_picture  # noqa: E402
from _stage import Body, Scene  # noqa: E402

COLOURS = [(0.85, 0.2, 0.15), (0.2, 0.65, 0.3), (0.25, 0.4, 0.9), (0.95, 0.75, 0.2), (0.7, 0.7, 0.75), (0.8, 0.4, 0.7)]
BALLS, BOXES = 10, 6


class SensorScene(Scene):
    def __init__(self, panel_width, panel_height, device="auto"):
        bodies = [Body("sphere", 0.35 + 0.06 * (i % 4), ng.Material(diffuse_color=COLOURS[i % len(COLOURS)], roughness=0.25 + 0.1 * (i % 3), metallic=0.0),
                       density=1200.0, restitution=0.5) for i in range(BALLS)]
        bodies += [Body("box", (0.3 + 0.05 * (i % 2), 0.3, 0.3 + 0.04 * (i % 3)), ng.Material(diffuse_color=COLOURS[(i + 2) % len(COLOURS)], roughness=0.5, metallic=0.0),
                        density=700.0) for i in range(BOXES)]
        super().__init__(panel_width, panel_height, bodies, device=device)
        self.rng = np.random.default_rng(4)
        self.next_body = 0

    def drop(self, centre=(0.0, 0.0, -5.0), spread=1.6):
        """Drops the next body from above a point near centre (renderer axes), with a little sideways speed."""
        i = self.next_body % len(self.stage.bodies)
        self.next_body += 1
        x, _, z = centre
        px, pz = x + self.rng.uniform(-spread, spread), z + self.rng.uniform(-spread, spread)
        self.throw(i, (px, 3.0 + self.rng.uniform(0, 1.5), pz), (self.rng.uniform(-0.3, 0.3), -1.0, self.rng.uniform(-0.3, 0.3)), speed=self.rng.uniform(1.0, 3.5))

    def reset(self):
        self.stage.park_all()
        self.push()
        self.frame = 0


def depth_panel(depth):
    """Depth as grey, near bright; nothing hit is black."""
    hit = np.isfinite(depth)
    panel = np.zeros(depth.shape + (3,), np.uint8)
    if hit.any():
        near, far = float(depth[hit].min()), float(np.percentile(depth[hit], 97))
        panel[hit] = (255 * (1 - np.clip((depth[hit] - near) / max(far - near, 1e-6), 0, 1)))[:, None]
    return panel


def id_panel(ids, palette):
    return palette[np.minimum(ids, len(palette) - 1)]


def lidar_panel(scene, camera_position, panel_width, panel_height, reach=18.0, rays=720):
    """A top-down plot of a 360-degree sweep in the horizontal plane through the camera: one query casts all the rays."""
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False, dtype=np.float32)
    directions = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=1)
    origins = np.tile(np.asarray([camera_position[0], 0.45, camera_position[2]], np.float32), (rays, 1))      # at ankle height, where the bodies are
    scan = scene.renderer.cast_rays(origins, directions, max_distance=reach)
    distance = np.asarray(scan["distance"], np.float32)
    panel = np.full((panel_height, panel_width, 3), 18, np.uint8)
    scale = 0.45 * min(panel_width, panel_height) / reach
    cx, cy = panel_width // 2, panel_height // 2
    for radius in (reach / 3, 2 * reach / 3, reach):                       # range rings
        theta = np.linspace(0, 2 * np.pi, 180)
        panel[np.clip((cy + scale * radius * np.cos(theta)).astype(int), 0, panel_height - 1), np.clip((cx + scale * radius * np.sin(theta)).astype(int), 0, panel_width - 1)] = (45, 45, 55)
    hit = np.isfinite(distance)
    px = np.clip((cx + scale * distance[hit] * directions[hit, 0]).astype(int), 1, panel_width - 3)
    py = np.clip((cy + scale * distance[hit] * directions[hit, 2]).astype(int), 1, panel_height - 3)
    for dx in (0, 1):
        for dy in (0, 1):
            panel[py + dy, px + dx] = (80, 255, 120)
    panel[cy - 2:cy + 3, cx - 2:cx + 3] = (255, 80, 80)                    # the sensor
    return panel


def physics_panel(scene, palette, panel_width, panel_height):
    """One bar per body, as tall as its speed: bodies fall (the bars grow), bounce and settle (they shrink to nothing). Colours are the ID panel's."""
    panel = np.full((panel_height, panel_width, 3), 24, np.uint8)
    n = len(scene.stage.bodies)
    speed = np.linalg.norm(scene.stage.world.velocities()[:, :3], axis=1)
    speed = np.where(scene.stage.active, speed, 0.0)
    width = panel_width // n
    top = 0.12 * panel_height
    for i in range(n):
        height = int(min(speed[i] / 9.0, 1.0) * (panel_height - top))
        colour = palette[ng.OBJECT_INSTANCE + 1 + i] if ng.OBJECT_INSTANCE + 1 + i < len(palette) else (200, 200, 200)
        panel[panel_height - max(height, 2):, i * width + 2:(i + 1) * width - 2] = colour if scene.stage.active[i] else (45, 45, 50)
    return panel


def mosaic(scene, camera, palette, panel_width, panel_height):
    out = scene.renderer.render_frame(camera, scene.frame, 4 if scene.renderer.device == "cuda" else 1, rgb=True, depth=True, normals=True, object_ids=True)
    scene.frame += 1
    rgb, depth, normals, ids = (np.asarray(out[k]) for k in ("rgb", "depth", "normals", "object_ids"))
    normal_panel = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    lidar = lidar_panel(scene, camera_position(camera), panel_width, panel_height)
    legend = physics_panel(scene, palette, panel_width, panel_height)
    top = np.concatenate([rgb, depth_panel(depth), normal_panel], axis=1)
    bottom = np.concatenate([id_panel(ids, palette), lidar, legend], axis=1)
    return np.concatenate([top, bottom], axis=0), int((ids >= ng.OBJECT_INSTANCE + 1).sum()), int(np.isfinite(depth).sum())


def camera_position(camera):
    return np.array(camera.position, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="480x270", help="each of the six panels, width x height")
    parser.add_argument("--save", metavar="FILE", help="write the six panels to FILE (a PNG); with no window, instead of showing a window")
    parser.add_argument("--frames", type=int, default=48, help="frames to refine for when saving without a window")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    args = parser.parse_args()
    pw, ph = (int(v) for v in args.size.lower().split("x"))

    try:
        scene = SensorScene(pw, ph)
    except ImportError as problem:
        sys.exit(f"{problem}")
    ids = ng.OBJECT_INSTANCE + 1 + len(scene.stage.bodies) + 2
    palette = np.random.default_rng(3).integers(70, 255, size=(ids, 3)).astype(np.uint8)          # object ids: the ground, then each body
    palette[ng.OBJECT_NONE] = 0
    camera_rig = FlyCamera((0.0, 2.2, 1.0), yaw=0.0, pitch=-0.32, speed=4.0)
    window = open_window("neraulGraphics: sensors", 3 * pw, 2 * ph)
    clock = [0.0]

    if window is None:
        for step in range(300):
            if step % 40 == 0:
                scene.drop()
            scene.advance(1 / 60, clock)
        scene.frame = 0
        for _ in range(args.frames):
            picture, _, _ = mosaic(scene, camera_rig.camera(pw, ph, 0.9), palette, pw, ph)
        save_picture(args.save or "sensors.png", picture)
        return

    started, last, fps, next_drop = time.perf_counter(), time.perf_counter(), 0.0, 0.5
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            dt = window.tick()
            t = time.perf_counter() - started
            camera_rig.update(window, dt)
            for key in window.take_keys():
                if key == "space":
                    scene.drop()
                elif key == "r":
                    scene.reset()
                elif key == "p":
                    window.screenshot("sensors.png")
            if t >= next_drop:
                scene.drop()
                next_drop = t + 1.1
            scene.advance(dt, clock)
            picture, on_objects, in_view = mosaic(scene, camera_rig.camera(pw, ph, 0.9), palette, pw, ph)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-4)
            last = now
            window.show(picture, f"{fps:4.0f} fps    RGB | depth | normals     object ids | LiDAR (top view, 720 rays) | body speeds (physics)"
                                 "\nSpace drop    R clear    P picture    Esc quit")
    if args.save:
        save_picture(args.save, picture)


if __name__ == "__main__":
    main()
