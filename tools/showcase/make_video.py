"""Renders the showcase video: what neraulGraphics can do, and how it behaves under load.

    python tools/showcase/make_video.py                    # 1080p, about 90 s, writes showcase.mp4
    python tools/showcase/make_video.py --quick            # 960x540 and a third of the length, to check changes
    python tools/showcase/make_video.py --only stress      # just the chapters whose name contains "stress"

Every frame is drawn by the engine itself. The numbers in the corner of the stress tests are measured
while that frame was being made (instance update time, render time, images per second), not typed in.
Frames are piped to ffmpeg, which uses the GPU's encoder when it has one.

Needs: neraulgraphics, numpy, pillow, torch (an NVIDIA GPU), ffmpeg on the PATH; Newton and MuJoCo for the last chapters.
"""

import argparse
import collections
import json
import math
import statistics
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import neraulgraphics as ng

FPS = 30
FONTS = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono-Bold.ttf" if mono and bold else "DejaVuSansMono.ttf" if mono else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------------------------------------------- drawing --


class Canvas:
    """Everything that is drawn on top of a rendered picture: captions, the stats panel, fades."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.s = height / 1080.0  # sizes below are for 1080 lines
        self.title_font, self.sub_font = font(int(46 * self.s), bold=True), font(int(26 * self.s))
        self.stat_font, self.panel_font = font(int(24 * self.s), mono=True), font(int(30 * self.s), bold=True)

    def px(self, v):
        return int(v * self.s)

    def compose(self, image, title=None, sub=None, stats=None, labels=None, fade=1.0):
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img, "RGBA")
        if title:
            bar = self.px(34)
            lines = 2 if sub else 1
            top = self.h - self.px(70 + 44 * lines) - bar
            draw.rectangle([0, top, self.w, self.h], fill=(8, 10, 16, 170))
            draw.text((self.px(56), top + self.px(26)), title, font=self.title_font, fill=(255, 255, 255, 255))
            if sub:
                draw.text((self.px(58), top + self.px(26) + self.px(66)), sub, font=self.sub_font, fill=(190, 205, 225, 255))
        if stats:
            width = max(draw.textlength(line, font=self.stat_font) for line in stats) + self.px(40)
            height = self.px(24 + 36 * len(stats))
            x0, y0 = self.w - width - self.px(36), self.px(36)
            draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=self.px(14), fill=(8, 10, 16, 190))
            for i, line in enumerate(stats):
                draw.text((x0 + self.px(20), y0 + self.px(14) + self.px(36) * i), line, font=self.stat_font, fill=(150, 255, 190, 255))
        for text, (x, y) in labels or []:
            x, y = self.px(x), self.px(y)
            if x < 0:  # a negative x is measured from the right edge
                x = self.w + x - draw.textlength(text, font=self.panel_font)
            draw.text((x, y), text, font=self.panel_font, fill=(255, 255, 255, 255), stroke_width=self.px(3), stroke_fill=(0, 0, 0, 255))
        out = np.asarray(img)
        if fade < 1.0:
            out = (out.astype(np.float32) * fade).astype(np.uint8)
        return out


def title_card(canvas, heading, lines, progress):
    """A dark card with centred text; progress 0..1 fades the text in."""
    h, w = canvas.h, canvas.w
    y = np.linspace(0.0, 1.0, h)[:, None, None]
    bg = (np.array([14, 20, 34]) * (1 - y) + np.array([4, 6, 12]) * y) * np.ones((1, w, 1))
    img = Image.fromarray(bg.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    alpha = min(1.0, progress * 4.0)
    big, small = font(canvas.px(110), bold=True), font(canvas.px(38))
    draw.text((w / 2, h * 0.38), heading, font=big, fill=tuple(int(255 * alpha) for _ in range(3)), anchor="mm")
    for i, line in enumerate(lines):
        shade = int(185 * alpha)
        draw.text((w / 2, h * 0.38 + canvas.px(120 + 56 * i)), line, font=small, fill=(shade, shade + 15, min(255, shade + 40)), anchor="mm")
    return np.asarray(img)


def turbo(t):
    """A colour ramp for depth, from near (warm) to far (cool); t in 0..1."""
    stops = np.array([[255, 240, 140], [250, 150, 50], [190, 50, 80], [70, 30, 120], [10, 10, 40]], np.float32)
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ helpers --


def orbit(t, radius, height, target, speed=0.25, phase=0.0):
    a = phase + t * speed
    return (target[0] + radius * math.sin(a), height, target[2] + radius * math.cos(a))


def spinning_field(count, extent, depth, seed, scale=(0.5, 1.6), floor_y=-3.0, height=18.0, near=-6.0):
    """Instances for a block of space: meshes alternate between a box and a sphere, a few colours."""
    rng = np.random.default_rng(seed)
    palette = [(0.9, 0.3, 0.25), (0.25, 0.75, 0.4), (0.3, 0.45, 0.95), (0.95, 0.8, 0.2), (0.8, 0.8, 0.85), (0.7, 0.35, 0.9)]
    materials = [ng.Material(diffuse_color=c, diffuse_albedo=0.85, specular_albedo=0.25, specular_exponent=60.0) for c in palette]
    instances = [ng.Instance(i % 2, material=materials[i % len(materials)]) for i in range(count)]
    base = np.stack([rng.uniform(-extent, extent, count), floor_y + 1.0 + rng.uniform(0.0, height, count),
                     near - rng.uniform(0.0, depth, count)], axis=1).astype(np.float32)
    return instances, base, rng.uniform(0, 6.28, count).astype(np.float32), rng.uniform(*scale, count).astype(np.float32)


def field_poses(base, phase, scale, t):
    """(N, 3, 4) matrices: every instance spins about its own up axis and bobs, as of time t."""
    c, s = np.cos(t * 0.9 + phase) * scale, np.sin(t * 0.9 + phase) * scale
    m = np.zeros((len(base), 3, 4), np.float32)
    m[:, 0, 0], m[:, 0, 2], m[:, 1, 1], m[:, 2, 0], m[:, 2, 2] = c, s, scale, -s, c
    m[:, :, 3] = base
    m[:, 1, 3] += 0.5 * np.sin(t * 1.4 + phase)
    return m


SHARED_MESHES = None


def meshes():
    global SHARED_MESHES
    if SHARED_MESHES is None:
        SHARED_MESHES = [ng.make_box((0.5, 0.5, 0.5)), ng.make_icosphere(0.5, 2)]
    return SHARED_MESHES


def median(values):
    return round(statistics.median(values), 3)


def floor_plane(half=400.0, z_min=-900.0, scale=1.0):
    return ng.Plane(y=-3.0, half_extent_x=half, z_min=z_min, z_max=60.0, checker_scale=scale)


# ------------------------------------------------------------------------------------------------ chapters --
# A chapter is an object with: name, seconds, setup(size), frame(i, n) -> (image, title, sub, stats), close().


class Classic:
    name, seconds = "classic", 8

    def setup(self, size):
        self.size = size
        self.instances, self.base, self.phase, self.scale = spinning_field(300, 22.0, 30.0, 1, floor_y=-3.0, height=5.0, near=-8.0)
        mirror = ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0.0, specular_albedo=10.0, reflectivity=0.8, specular_exponent=1425.0)
        ivory = ng.Material(diffuse_color=(0.4, 0.4, 0.3), diffuse_albedo=0.6, specular_albedo=0.3, reflectivity=0.1, specular_exponent=50.0)
        self.r = ng.Renderer(*size, [ng.Sphere((-3, 0, -16), 2.5, ivory), ng.Sphere((2.5, -0.5, -13), 2.5, mirror),
                                     ng.Sphere((8, 3, -20), 3.5, mirror)],
                             [ng.Light((-20, 25, 15), 1.5, 1.5), ng.Light((25, 30, -10), 1.0, 1.0)],
                             floor=floor_plane(60, -80, 1.0), meshes=meshes(), instances=self.instances)
        self.r.ambient = 0.15

    def frame(self, i, n):
        t = i / FPS
        self.r.set_instance_transforms(field_poses(self.base, self.phase, self.scale, t))
        cam = ng.make_camera(orbit(t, 16.0, 4.0 + 2.0 * math.sin(t * 0.5), (0, -0.5, -16), 0.3), (0, -0.5, -16), width=self.size[0], height=self.size[1], fov=1.0)
        image = self.r.render(cam, 0, 4)
        return image, "A fast classic renderer", "Reflections, soft shadows, anti-aliasing, 300 spinning meshes sharing two shapes", \
            ["classic renderer", f"render  {self.r.last_kernel_ms:5.2f} ms"]

    def close(self):
        del self.r


class Beauty:
    name, seconds = "beauty", 13

    def setup(self, size):
        self.size = size
        clay = ng.Material(diffuse_color=(0.8, 0.35, 0.25), roughness=0.6, metallic=0.0)
        gold = ng.Material(diffuse_color=(1.0, 0.76, 0.34), roughness=0.15, metallic=1.0)
        steel = ng.Material(diffuse_color=(0.9, 0.95, 1.0), roughness=0.03, metallic=1.0)
        blue = ng.Material(diffuse_color=(0.2, 0.4, 0.85), roughness=0.25, metallic=0.0)
        lamp = ng.Material(diffuse_color=(0, 0, 0), emission=(7.0, 5.0, 3.2))
        spheres = [ng.Sphere((-2.6, -0.3, -7), 1.6, clay), ng.Sphere((0.6, -0.9, -5.2), 1.1, gold), ng.Sphere((3.0, -0.8, -6.5), 1.2, steel),
                   ng.Sphere((-0.5, -1.6, -3.4), 0.5, blue), ng.Sphere((-4.6, 2.4, -9.5), 0.45, lamp)]
        floor = ng.Plane(y=-2, half_extent_x=60, z_min=-120, z_max=10, checker_scale=1.0)
        sky = ng.Environment(map=ng.procedural_sky(sun_elevation=0.6, sun_azimuth=2.4, sun_strength=3.2, sun_radius=0.04))

        def make(mode):
            r = ng.Renderer(*size, spheres[:-1] if mode == "classic" else spheres, [ng.Light((-30, 40, 20), 1.6, 1.0)] if mode == "classic" else [], floor=floor)
            r.set_environment(sky if mode != "classic" else ng.Environment())
            if mode == "classic":
                r.ambient = 0.2
            else:
                r.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=5, denoise=(mode == "denoised")))
            return r

        self.classic, self.raw, self.clean = make("classic"), make("raw"), make("denoised")

    def wipe(self, left, right, x):
        """left up to column x, right after it, with a thin line between."""
        col = int(x * self.size[0])
        image = np.concatenate([left[:, :col], right[:, col:]], axis=1)
        image[:, max(col - 1, 0):col + 1] = 255
        return image

    def frame(self, i, n):
        t = i / FPS
        w, h = self.size
        cam = ng.make_camera(orbit(t, 7.0, 0.9 + 0.4 * math.sin(t * 0.6), (0, -0.6, -6.0), 0.22, 0.3), (0, -0.7, -6.0), width=w, height=h, fov=1.0)
        a, labels = i / n, []
        # classic; wipe to path traced; path traced; wipe from raw 1 sample per pixel to denoised; denoised
        if a < 0.2:
            image = self.classic.render(cam, 0, 4)
            title, sub = "Beauty mode", "Same scene, same engine: first the classic picture"
            stats = ["classic renderer", f"render  {self.classic.last_kernel_ms:5.2f} ms"]
        else:
            clean = self.clean.render(cam, i)  # every frame: its history is what makes it clean
            image = clean
            stats = [f"path traced  {self.clean.last_kernel_ms:5.2f} ms", "5 bounces, 1 sample/pixel"]
            if a < 0.4:
                image = self.wipe(self.classic.render(cam, 0, 4), clean, (a - 0.2) / 0.2)
                labels = [("classic", (40, 150)), ("path traced", (-40, 150))]
                title, sub = "Beauty mode: path tracing", "Light bounces between surfaces; the sky and a sun light the scene"
            elif a < 0.65:
                title, sub = "Physically based materials", "Clay, gold, polished steel, painted plastic and an emissive lamp, with soft shadows and reflections"
            elif a < 0.9:
                image = self.wipe(self.raw.render(cam, i), clean, (a - 0.65) / 0.25)
                labels = [("1 sample per pixel, raw", (40, 150)), ("denoised", (-40, 150))]
                title, sub = "Denoised in real time", "One path per pixel, reusing the last frame along the camera's motion, then an edge-aware filter"
            else:
                title, sub = "Beauty mode", "Fully path traced and denoised, orbiting live"
        return image, title, sub, stats, labels

    def close(self):
        del self.classic, self.raw, self.clean


class Sensors:
    name, seconds = "sensors", 8

    def setup(self, size):
        self.size = size
        self.half = (size[0] // 2, size[1] // 2)
        self.instances, self.base, self.phase, self.scale = spinning_field(60, 7.0, 14.0, 5, scale=(0.5, 1.2), floor_y=-3.0, height=3.5, near=-4.0)
        sphere = ng.Material(diffuse_color=(0.85, 0.85, 0.9), specular_albedo=0.3)
        self.r = ng.Renderer(*self.half, [ng.Sphere((0, -1.0, -13), 2.0, sphere)], [ng.Light((-15, 20, 10), 1.4, 1.0)],
                             floor=floor_plane(40, -60, 1.0), meshes=meshes(), instances=self.instances)
        self.r.ambient = 0.2

    def frame(self, i, n):
        t = i / FPS
        self.r.set_instance_transforms(field_poses(self.base, self.phase, self.scale, t))
        w, h = self.half
        cam = ng.make_camera(orbit(t, 11.0, 3.0, (0, -1.0, -10), 0.3, 0.4), (0, -1.0, -10), width=w, height=h, fov=1.1)
        out = self.r.render_frame(cam, 0, 4, rgb=True, depth=True, normals=True, object_ids=True)
        depth = out["depth"]
        finite = np.isfinite(depth)
        shown = np.zeros((h, w, 3), np.uint8)
        if finite.any():
            lo, hi = float(depth[finite].min()), float(np.percentile(depth[finite], 98))
            shown[finite] = turbo((depth[finite] - lo) / max(hi - lo, 1e-3))
        normals = ((out["normals"] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        ids = out["object_ids"].astype(np.uint32)
        hashed = ids * np.uint32(2654435761)
        colors = np.stack([(hashed >> 24) & 255, (hashed >> 16) & 255, (hashed >> 8) & 255], -1).astype(np.uint8) // 2 + 64
        colors[ids == ng.OBJECT_NONE] = 0
        grid = np.concatenate([np.concatenate([out["rgb"], shown], axis=1), np.concatenate([normals, colors], axis=1)], axis=0)
        labels = [("RGB", (24, 16)), ("Depth", (984, 16)), ("Normals", (24, 556)), ("Object IDs", (984, 556))]
        return grid, "Sensors for robots and RL", "Depth, surface normals and segmentation come from the same rays, with no extra pass over the scene", \
            ["every pixel: one exact ray", f"{self.r.last_kernel_ms:5.2f} ms for all four"], labels

    def close(self):
        del self.r


class InstanceStress:
    """A ramp of moving instances: each stage is a new renderer with more of them, flown through."""

    name = "stress-instances"
    stages = [1_000, 10_000, 50_000, 100_000, 200_000]
    seconds = 3.2 * len(stages)

    def setup(self, size):
        self.size, self.stage, self.r = size, -1, None
        self.samples = collections.defaultdict(list)

    def build(self, count):
        self.r = None
        self.instances, self.base, self.phase, self.scale = spinning_field(count, 70.0, 260.0, 11, scale=(0.5, 1.8), height=22.0)
        self.r = ng.Renderer(*self.size, [], [ng.Light((-60, 90, 30), 1.3, 0.0), ng.Light((70, 60, -80), 0.6, 0.0)],
                             floor=floor_plane(200, -400, 2.0), meshes=meshes(), instances=self.instances)
        self.r.ambient = 0.25
        self.count = count

    def frame(self, i, n):
        per = n // len(self.stages)
        stage = min(i // per, len(self.stages) - 1)
        if stage != self.stage:
            self.stage = stage
            self.build(self.stages[stage])
        t = i / FPS
        start = time.perf_counter()
        self.r.set_instance_transforms(field_poses(self.base, self.phase, self.scale, t))
        update_ms = (time.perf_counter() - start) * 1000.0
        z = -8.0 - 120.0 * (0.5 + 0.5 * math.sin(t * 0.35))
        # above the field, looking down at the swarm: a camera inside it would spend the frame inside one box
        cam = ng.make_camera((14.0 * math.sin(t * 0.4), 34.0 + 3.0 * math.sin(t * 0.7), z), (0.0, 4.0, z - 55.0), width=self.size[0], height=self.size[1], fov=1.05)
        start = time.perf_counter()
        image = self.r.render(cam, 0, 1)
        render_ms = (time.perf_counter() - start) * 1000.0
        self.samples[self.count].append((update_ms, render_ms))
        stats = [f"instances   {self.count:>9,}", f"update      {update_ms:7.2f} ms", f"render      {render_ms:7.2f} ms",
                 f"{self.size[0]}x{self.size[1]}  {1000.0 / max(update_ms + render_ms, 1e-3):6.0f} fps"]
        return image, f"Stress test: {self.count:,} moving instances", \
            "Every body moves every frame. Two shared meshes; only a pose per instance is updated", stats

    def summary(self):
        out = {}
        for count, rows in self.samples.items():
            update, render = median([r[0] for r in rows]), median([r[1] for r in rows])
            out[str(count)] = {"update_ms": update, "render_ms": render, "frames": len(rows), "fps": round(1000.0 / (update + render), 1)}
        return out

    def close(self):
        self.r = None


class BatchStress:
    name = "stress-batch"
    seconds = 12

    def setup(self, size):
        import torch

        self.torch, self.size = torch, size
        self.stage = -1
        self.stages = [(1024, 240), (4096, 120)]
        self.samples = collections.defaultdict(list)

    def build(self, worlds, tile):
        torch = self.torch
        self.r = None
        colors = [(0.8, 0.25, 0.25), (0.25, 0.7, 0.35), (0.3, 0.4, 0.9), (0.9, 0.75, 0.2)]
        self.r = ng.BatchRenderer(tile, tile, worlds, 1, [], [ng.Light((-6, 9, 4), 1.4, 0.0), ng.Light((8, 6, 2), 0.5, 0.0)],
                                  ng.Plane(y=-1.5, half_extent_x=20, z_min=-40, z_max=5), [ng.make_box((0.5, 0.5, 0.5)), ng.make_icosphere(0.5, 2)],
                                  [ng.Instance(k % 2, material=ng.Material(diffuse_color=colors[k])) for k in range(4)])
        self.r.ambient = 0.2
        from neraulgraphics.shapes import look_at

        self.cameras = torch.from_numpy(np.tile(look_at((0.0, 1.2, 2.0), (0.0, 0.0, -5.0), up=(0, 1, 0)), (worlds, 1, 1))).cuda()
        gen = torch.Generator(device="cuda").manual_seed(3)
        self.phase = torch.rand(worlds, 4, device="cuda", generator=gen) * 6.28
        self.base = torch.zeros(worlds, 4, 3, 4, device="cuda")
        self.base[..., 0, 3] = (torch.rand(worlds, 4, device="cuda", generator=gen) - 0.5) * 5
        self.base[..., 1, 3] = -1.0 + torch.rand(worlds, 4, device="cuda", generator=gen)
        self.base[..., 2, 3] = -3.0 - torch.rand(worlds, 4, device="cuda", generator=gen) * 5
        self.worlds, self.tile = worlds, tile
        self.cols = self.size[0] // tile
        self.rows = min(self.size[1] // tile, 32)

    def frame(self, i, n):
        torch = self.torch
        stage = 0 if i < n * 0.55 else 1
        if stage != self.stage:
            self.stage = stage
            self.build(*self.stages[stage])
        t = i / FPS
        m = self.base.clone()
        c, s = torch.cos(t * 0.9 + self.phase), torch.sin(t * 0.9 + self.phase)
        m[..., 0, 0], m[..., 0, 2], m[..., 1, 1], m[..., 2, 0], m[..., 2, 2] = c, s, 1.0, -s, c
        m[..., 1, 3] += 0.4 * torch.sin(t * 1.5 + self.phase)
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.r.set_instance_transforms(m)
        out = self.r.render(self.cameras, fov=1.0, rgb=True, zero_copy=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000.0
        self.samples[(self.worlds, self.tile)].append(ms)
        count = self.cols * self.rows
        rgb = torch.from_dlpack(out["rgb"])[:count].cpu().numpy()
        grid = rgb.reshape(self.rows, self.cols, self.tile, self.tile, 3).transpose(0, 2, 1, 3, 4).reshape(self.rows * self.tile, self.cols * self.tile, 3)
        canvas = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        canvas[:grid.shape[0], :grid.shape[1]] = grid[:self.size[1], :self.size[0]]
        stats = [f"worlds     {self.worlds:>7,}", f"views      {self.tile}x{self.tile}", f"one call   {ms:7.2f} ms", f"images/s   {self.worlds / (ms / 1000.0):9,.0f}"]
        return canvas, f"Stress test: {self.worlds:,} worlds at once", \
            f"{count} of them shown. All {self.worlds:,} are rendered every frame, from poses and cameras that never leave the GPU", stats

    def summary(self):
        out = {}
        for (worlds, tile), values in self.samples.items():
            ms = median(values)
            out[str(worlds)] = {"view": f"{tile}x{tile}", "call_ms": ms, "images_per_second": round(worlds / (ms / 1000.0)), "frames": len(values)}
        return out

    def close(self):
        self.r = None


class PathStress:
    name, seconds = "stress-path", 9

    def setup(self, size):
        self.size = size
        self.count = 12_000
        self.samples = []
        self.instances, self.base, self.phase, self.scale = spinning_field(self.count, 45.0, 120.0, 21, scale=(0.5, 1.5), height=12.0, near=-6.0)
        self.r = ng.Renderer(*size, [], [], floor=floor_plane(150, -300, 2.0), meshes=meshes(), instances=self.instances)
        self.r.set_environment(ng.Environment(map=ng.procedural_sky(sun_elevation=0.55, sun_azimuth=1.0, sun_strength=3.5, sun_radius=0.04)))
        self.r.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=3))

    def frame(self, i, n):
        t = i / FPS
        start = time.perf_counter()
        self.r.set_instance_transforms(field_poses(self.base, self.phase, self.scale, t))
        update_ms = (time.perf_counter() - start) * 1000.0
        z = -10.0 - 55.0 * (0.5 + 0.5 * math.sin(t * 0.45))
        cam = ng.make_camera((8.0 * math.sin(t * 0.5), 21.0 + 2.0 * math.sin(t * 0.8), z), (0.0, 2.0, z - 35.0), width=self.size[0], height=self.size[1], fov=1.0)
        start = time.perf_counter()
        image = self.r.render(cam, i, 2)
        render_ms = (time.perf_counter() - start) * 1000.0
        self.samples.append((update_ms, render_ms))
        stats = [f"instances   {self.count:>7,}", f"update      {update_ms:6.2f} ms", f"path trace  {render_ms:6.2f} ms", f"{self.size[0]}x{self.size[1]}  3 bounces, 2 spp"]
        return image, f"Stress test: path tracing {self.count:,} moving instances", "Global illumination, sun shadows and the denoiser, while every body moves", stats

    def summary(self):
        update, render = median([r[0] for r in self.samples]), median([r[1] for r in self.samples])
        return {str(self.count): {"update_ms": update, "render_ms": render, "frames": len(self.samples), "fps": round(1000.0 / (update + render), 1),
                                  "settings": "3 bounces, 2 samples per pixel, denoised"}}

    def close(self):
        del self.r


class SimNewton:
    name, seconds = "newton", 9

    def setup(self, size):
        import newton
        import warp as wp
        from neraulgraphics.newton import NewtonRenderer

        self.wp, self.newton = wp, newton
        rng = np.random.default_rng(4)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        adders = [lambda b: builder.add_shape_sphere(b, radius=0.45), lambda b: builder.add_shape_box(b, hx=0.4, hy=0.4, hz=0.4),
                  lambda b: builder.add_shape_capsule(b, radius=0.28, half_height=0.5), lambda b: builder.add_shape_cylinder(b, radius=0.35, half_height=0.45)]
        for k in range(70):
            pos = wp.vec3(float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)), 1.5 + 0.55 * k)
            body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))
            adders[k % len(adders)](body)
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=6)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.view = NewtonRenderer(self.model, *size)
        self.view.beauty(ng.Environment(up=(0, 0, 1), map=ng.procedural_sky(sun_elevation=0.7, sun_azimuth=0.5, sun_strength=3.2, sun_radius=0.04)), bounces=4)
        self.size = size

    def frame(self, i, n):
        for _ in range(5):
            self.s0.clear_forces()
            self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, 1 / 150)
            self.s0, self.s1 = self.s1, self.s0
        t = i / FPS
        a = 0.35 + 0.25 * math.sin(t * 0.3)
        cam = self.view.camera((11.0 * math.cos(a), 11.0 * math.sin(a) - 4.0, 5.0), (0.0, 0.0, 2.0))
        image = self.view.render(self.s0, cam, frame=i)
        return image, "Newton simulation, beauty mode", "70 rigid bodies from Newton, drawn as instances: after the first frame only poses are updated", \
            ["Newton XPBD", f"path trace {self.view.renderer.last_kernel_ms:5.2f} ms", "Z-up sky"]

    def close(self):
        del self.view


MJCF = """
<mujoco><option timestep="0.004"/><worldbody>
<geom type="plane" size="20 20 0.1" rgba="0.7 0.7 0.72 1"/>
{bodies}
</worldbody></mujoco>
"""


class SimMuJoCo:
    name, seconds = "mujoco", 6

    def setup(self, size):
        import mujoco
        from neraulgraphics.mujoco import MuJoCoRenderer

        self.mujoco = mujoco
        rng = np.random.default_rng(8)
        kinds = ['type="sphere" size="0.35"', 'type="box" size="0.3 0.3 0.3"', 'type="capsule" size="0.22 0.35"', 'type="cylinder" size="0.3 0.3"']
        bodies = []
        for k in range(60):
            color = rng.uniform(0.25, 0.95, 3)
            bodies.append(f'<body pos="{rng.uniform(-2.5, 2.5):.2f} {rng.uniform(-2.5, 2.5):.2f} {1.0 + 0.5 * k:.2f}"><freejoint/>'
                          f'<geom {kinds[k % 4]} rgba="{color[0]:.2f} {color[1]:.2f} {color[2]:.2f} 1" mass="1"/></body>')
        self.model = mujoco.MjModel.from_xml_string(MJCF.format(bodies="\n".join(bodies)))
        self.data = mujoco.MjData(self.model)
        self.view = MuJoCoRenderer(self.model, *size)
        self.view.beauty(ng.Environment(up=(0, 0, 1), map=ng.procedural_sky(sun_elevation=0.5, sun_azimuth=2.2, sun_strength=3.0, sun_radius=0.04)), bounces=4)
        self.size = size
        self.steps = int(1 / FPS / self.model.opt.timestep)

    def frame(self, i, n):
        for _ in range(self.steps):
            self.mujoco.mj_step(self.model, self.data)
        t = i / FPS
        a = 2.0 + 0.2 * t
        cam = self.view.camera((10.0 * math.cos(a), 10.0 * math.sin(a), 5.0), (0.0, 0.0, 1.8))
        image = self.view.render(self.data, cam, frame=i)
        return image, "MuJoCo simulation, beauty mode", "The same engine draws MuJoCo: its geoms become instances once, then only poses move", \
            ["MuJoCo", f"path trace {self.view.renderer.last_kernel_ms:5.2f} ms"]

    def close(self):
        del self.view


CHAPTERS = [Classic, Beauty, Sensors, InstanceStress, BatchStress, PathStress, SimNewton, SimMuJoCo]


# ---------------------------------------------------------------------------------------------------- main --


class Encoder:
    def __init__(self, path, size, fps):
        self.size = size
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-"]
        for codec in (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "30", "-b:v", "0"], ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]):
            self.proc = subprocess.Popen(cmd + codec + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            time.sleep(0.4)
            if self.proc.poll() is None:
                self.codec = codec[1]
                return
        raise RuntimeError("ffmpeg could not start")

    def write(self, frame):
        assert frame.shape == (self.size[1], self.size[0], 3) and frame.dtype == np.uint8, (frame.shape, frame.dtype)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        return self.proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="showcase.mp4")
    parser.add_argument("--quick", action="store_true", help="960x540 and a third of the length")
    parser.add_argument("--only", help="only chapters whose name contains this")
    parser.add_argument("--stats", help="write the measurements of the stress tests here (default: next to the video)")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is needed on the PATH")
    size = (960, 540) if args.quick else (1920, 1080)
    canvas = Canvas(*size)
    scale = 1 / 3 if args.quick else 1.0
    encoder = Encoder(args.out, size, FPS)
    print(f"{size[0]}x{size[1]} at {FPS} fps, encoding with {encoder.codec}")
    began = time.perf_counter()
    total = 0
    measured = {}

    def card(heading, lines, seconds):
        nonlocal total
        n = int(seconds * FPS * scale) or 1
        for i in range(n):
            frame = title_card(canvas, heading, lines, i / max(n - 1, 1))
            fade = min(1.0, (n - i) / (0.4 * FPS))
            encoder.write((frame * fade).astype(np.uint8))
            total += 1

    if not args.only:
        card("neraulGraphics", ["GPU raytracing for robots, simulators and anyone who needs a picture", "Python and C++  ·  NVIDIA GPU or any CPU  ·  Apache-2.0"], 4)
    for cls in CHAPTERS:
        if args.only and args.only not in cls.name:
            continue
        chapter = cls()
        n = max(int(chapter.seconds * FPS * scale), 12)
        print(f"  {chapter.name}: {n} frames", flush=True)
        chapter.setup(size)
        start = time.perf_counter()
        for i in range(n):
            result = chapter.frame(i, n)
            image, title, sub, stats = result[:4]
            labels = result[4] if len(result) > 4 else None
            fade = min(1.0, (i + 1) / (0.3 * FPS), (n - i) / (0.3 * FPS))
            encoder.write(canvas.compose(image, title, sub, stats, labels, fade))
            total += 1
        print(f"    {time.perf_counter() - start:.1f} s to render", flush=True)
        if hasattr(chapter, "summary"):
            measured[chapter.name] = chapter.summary()
        chapter.close()
    if not args.only:
        card("pip install neraulgraphics", ["github.com/aneangel/neraulGraphics", "Beauty mode  ·  thousands of worlds  ·  Newton and MuJoCo adapters"], 5)
    status = encoder.close()
    if measured:
        import torch

        machine = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "resolution": f"{size[0]}x{size[1]}",
                   "neraulgraphics": ng.__version__, "note": "medians over the frames of each stage, measured while rendering the video"}
        stats_path = Path(args.stats) if args.stats else Path(args.out).with_suffix(".stats.json")
        stats_path.write_text(json.dumps({"machine": machine, **measured}, indent=2))
        print("wrote", stats_path)
    print(f"wrote {args.out}: {total} frames, {total / FPS:.0f} s, in {time.perf_counter() - began:.0f} s (ffmpeg exit {status})")


if __name__ == "__main__":
    main()
