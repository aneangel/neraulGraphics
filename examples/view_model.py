"""A path-traced model viewer: orbit a glTF, GLB or OBJ model, or the built-in torus knot, then drop it and watch it fall.

    python examples/view_model.py                         # the built-in gold torus knot
    python examples/view_model.py model.glb               # any glTF, GLB or OBJ model, with its own materials and textures
    python examples/view_model.py model.glb --save a.png  # no window: write a picture
    python examples/view_model.py --turntable             # write a slow orbit as a GIF

Drag with the left button to orbit, the wheel zooms. D drops the model from above as a rigid body (its bounding box is the collider) and it
tumbles to rest under Jolt's deterministic physics; R puts it back. M cycles the knot's material (gold, glass, steel, clay). [ and ] lower and
raise the sun, , and . swing it round, B switches between the fast classic picture and path tracing, P saves a picture, Esc quits.
Hold still and the picture keeps getting cleaner.

Needs: pip install neraulgraphics pillow pyglet (and NERAULGRAPHICS_BUILD_PHYSICS=ON for the drop). An NVIDIA GPU makes it fast.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import neraulgraphics as ng
from neraulgraphics import physics

sys.path.insert(0, str(Path(__file__).resolve().parent / "scenery"))

from _live import OrbitCamera, open_window, save_picture, up_frame  # noqa: E402

MATERIALS = {
    "gold": ng.Material(diffuse_color=(1.0, 0.78, 0.34), roughness=0.1, metallic=1.0),
    "glass": ng.Material(diffuse_color=(0.85, 0.95, 1.0), roughness=0.0, metallic=0.0, transmission=1.0, ior=1.5),
    "steel": ng.Material(diffuse_color=(0.78, 0.80, 0.84), roughness=0.35, metallic=1.0),
    "clay": ng.Material(diffuse_color=(0.7, 0.45, 0.35), roughness=0.85, metallic=0.0),
}


def torus_knot(p=2, q=3, radius=1.0, tube=0.28, segments=360, sides=36):
    """A (p, q) torus knot as a smooth tube, lying flat (y is up): a closed curve with a ring of vertices at each step and a normal at every
    vertex. Returns the mesh and its vertices."""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    r = radius * (2.0 + np.cos(q * t)) / 3.0
    curve = np.stack([r * np.cos(p * t), -radius * np.sin(q * t) / 3.0 * 2.0, r * np.sin(p * t)], axis=1)
    tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    binormal = np.cross(tangent, reference)
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    normals = np.cos(angle)[None, :, None] * reference[:, None, :] + np.sin(angle)[None, :, None] * binormal[:, None, :]
    vertices = curve[:, None, :] + tube * normals
    index = np.arange(segments * sides).reshape(segments, sides)
    a, b = index, np.roll(index, -1, axis=0)
    c, d = np.roll(index, -1, axis=1), np.roll(np.roll(index, -1, axis=0), -1, axis=1)
    triangles = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    flat = vertices.reshape(-1, 3).astype(np.float32)
    return ng.Mesh(flat, triangles.astype(np.uint32), normals.reshape(-1, 3).astype(np.float32)), flat


class Viewer:
    def __init__(self, scene, width, height, knot_material=None):
        self.scene, self.width, self.height = scene, width, height
        self.up = np.asarray(scene.up, np.float64)
        self.frame_axes = up_frame(self.up)                           # rows: the physics x, y and z (z along up)
        self.center = np.asarray(scene.center, np.float64)
        self.extent = float(scene.extent)
        self.base = np.stack([i.transform.matrix.astype(np.float64) for i in scene.instances])      # (N, 3, 4) as loaded
        # the model's half extents along the physics axes, from the box round its geometry
        lo, hi = np.asarray(scene.bounds_min, np.float64), np.asarray(scene.bounds_max, np.float64)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        local = (corners - self.center) @ self.frame_axes.T
        self.half = np.maximum((local.max(axis=0) - local.min(axis=0)) / 2.0, 1e-3)
        self.bottom = float((corners @ self.up).min())
        self.origin = self.center - self.up * (self.half[2])         # the ground point below the resting centre, so z = 0 is the ground
        self.drop_height = self.half[2] + 1.2 * self.extent
        self.sun_elevation, self.sun_azimuth = 0.7, 2.2
        self.beauty = True
        self.frame = 0
        self.world = None
        self._build_renderer()
        if physics.available():
            self._build_world()

    def _build_renderer(self):
        from textures import concrete, textured_box

        colour, normal, orm = concrete(seed=9, stains=0.05, streaks=0.03)
        textures = list(self.scene.textures) + [ng.Texture(colour), ng.Texture(normal, linear=True), ng.Texture(orm, linear=True)]
        t0 = len(self.scene.textures)
        v, tri, nrm, uv = textured_box((600.0 * self.extent, 0.5, 600.0 * self.extent), tile=max(0.4 * self.extent, 0.2))
        meshes = list(self.scene.meshes) + [ng.Mesh(v, tri, nrm, uv)]
        slab = len(self.scene.meshes)
        ground = ng.Instance(slab, ng.Transform(np.eye(4, dtype=np.float32)), ng.Material(diffuse_color=(0.85, 0.84, 0.82), roughness=1.0, metallic=0.0),
                             texture=t0, normal_texture=t0 + 1, orm_texture=t0 + 2)
        self.renderer = ng.Renderer(self.width, self.height, [], [ng.Light((0, 0, 0), intensity=0.0)], meshes=meshes,
                                    instances=list(self.scene.instances) + [ground], textures=textures, device="auto")
        self.renderer.ambient = 0.15
        self.ground_matrix = self._ground_matrix()
        self.set_sky()
        self.set_mode()
        self.place(self.rest_pose())

    def _ground_matrix(self):
        """The slab's matrix: its local x, y and z along the physics x, z (up) and y axes, its top face at the ground level of the model."""
        x, y, z = self.frame_axes
        m = np.zeros((3, 4))
        m[:, 0], m[:, 1], m[:, 2] = x, z, y            # the slab's thin axis (its local y) along up; a mirrored matrix is fine for a symmetric slab
        m[:, 3] = self.origin - self.up * 0.5
        return m.astype(np.float32)

    def set_sky(self):
        # the sky without its sun, and the sun as an explicit light, so highlights are computed and do not twinkle (see _stage.Scene.set_sky);
        # for a model whose up is not +y the sun's direction is turned with the sky
        sun = dict(sun_elevation=self.sun_elevation, sun_azimuth=self.sun_azimuth, turbidity=2.8, sun_strength=4.0, sun_radius=0.02)
        sky = ng.physical_sky(sky_ratio=0.22, ground_albedo=0.25, width=768, height=384, include_sun=False, **sun)
        up = tuple(float(v) for v in self.up)
        self.renderer.set_environment(ng.Environment(up=up, map=sky))
        self.renderer.set_lights([ng.sun_light(up=up, **sun)])
        self.frame = 0

    def set_mode(self):
        if self.beauty:
            self.renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=10, exposure=0.8, denoise=True, denoise_iterations=3, max_radiance=60.0))
        else:
            self.renderer.set_render_mode(ng.RenderMode.Classic)
        self.frame = 0

    # --------------------------------------------------------------------------------------------------- physics --

    def _build_world(self):
        self.world = physics.World(max_bodies=4)
        self.world.add_ground(friction=0.7)
        h = self.half
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * h.astype(np.float32)
        self.world.add_convex_hull(np.ascontiguousarray(corners), (0.0, 0.0, float(h[2])), 600.0, 0.6, 0.35, 0.0, False)
        self.world.finish_setup()
        self.falling = False

    def rest_pose(self):
        """(centre in the renderer's world, rotation in the renderer's world): the model as loaded."""
        return self.center, np.eye(3)

    def place(self, pose):
        """Moves the whole model rigidly so that its centre is at pose[0] and it is turned by pose[1] about that centre."""
        centre, rotation = pose
        a = self.base
        new = np.empty_like(a)
        new[:, :, :3] = np.einsum("ij,njk->nik", rotation, a[:, :, :3])
        new[:, :, 3] = np.einsum("ij,nj->ni", rotation, a[:, :, 3] - self.center) + centre
        matrices = np.concatenate([new, self.ground_matrix[None].astype(np.float64)]).astype(np.float32)
        self.renderer.set_instance_transforms(matrices)

    def drop(self):
        if not self.world:
            return
        angle = np.random.default_rng(int(time.time())).normal(size=3)
        q = np.concatenate([np.sin(0.4) * angle / np.linalg.norm(angle), [np.cos(0.4)]]).astype(np.float32)
        poses = self.world.poses().copy()
        poses[0, :3] = (0.0, 0.0, self.drop_height)
        poses[0, 3:7] = q
        self.world.reset(poses, np.zeros((1, 6), np.float32))
        self.falling = True

    def reset(self):
        if self.world:
            poses = self.world.poses().copy()
            poses[0, :3] = (0.0, 0.0, float(self.half[2]))
            poses[0, 3:7] = (0, 0, 0, 1)
            self.world.reset(poses, np.zeros((1, 6), np.float32))
        self.falling = False
        self.place(self.rest_pose())

    def advance(self, dt, clock):
        """Steps the physics (when dropped) and moves the model to match. Returns whether it moved."""
        if not (self.world and self.falling):
            return False
        clock[0] += dt
        steps = 0
        while clock[0] >= 1 / 60 and steps < 4:
            self.world.step(1 / 60, 4)
            clock[0] -= 1 / 60
            steps += 1
        if not steps:
            return False
        from neraulgraphics.shapes import quat_to_matrix

        pose = self.world.poses()[0]
        R_phys = quat_to_matrix(pose[3:7])[0]
        B = self.frame_axes.T                                           # columns: the physics axes in the renderer's world
        rotation = B @ R_phys @ B.T
        centre = self.origin + B @ pose[:3]
        self.place((centre, rotation))
        return True

    # ----------------------------------------------------------------------------------------------- the picture --

    def render(self, camera):
        image = self.renderer.render(camera, self.frame, 4 if (self.beauty and self.renderer.device == "cuda") else 1)
        self.frame += 1
        return image


def build_scene(args):
    if args.model:
        return ng.load(args.model), None
    material = MATERIALS[args.material]
    mesh, vertices = torus_knot()
    lift = -float(vertices[:, 1].min())                           # stand it on the ground
    lo, hi = vertices.min(axis=0) + (0.0, lift, 0.0), vertices.max(axis=0) + (0.0, lift, 0.0)
    scene = ng.LoadedScene(meshes=[mesh], instances=[ng.Instance(0, ng.make_transform((0.0, lift, 0.0)), material)], textures=[], up=(0.0, 1.0, 0.0),
                           center=(lo + hi) / 2.0, extent=float(np.linalg.norm(hi - lo)) / 2.0, bounds_min=lo, bounds_max=hi)
    return scene, material


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="a glTF, GLB or OBJ file; with none, the built-in torus knot")
    parser.add_argument("--material", default="gold", choices=sorted(MATERIALS), help="the knot's material (a loaded model keeps its own)")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--save", metavar="FILE", help="write a picture to FILE (a PNG); with no window, instead of showing one")
    parser.add_argument("--turntable", action="store_true", help="write a slow orbit as a GIF")
    parser.add_argument("--frames", type=int, default=120, help="frames to refine for when saving a picture, or in a turntable")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    scene, knot_material = build_scene(args)
    view = Viewer(scene, width, height)
    orbit = OrbitCamera(view.center, 2.6 * view.extent, yaw=0.6, pitch=0.28, up=view.up)

    if args.turntable:
        from PIL import Image

        out = Path((Path(args.model).stem if args.model else "knot") + ".gif")
        frames = []
        for i in range(args.frames):
            orbit.yaw = 2 * np.pi * i / args.frames
            for k in range(6):
                image = view.render(orbit.camera(width, height))
            frames.append(Image.fromarray(image))
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print("wrote", out)
        return

    window = open_window("neraulGraphics: model viewer", width, height)
    if window is None:
        view.frame = 0
        for _ in range(args.frames):
            image = view.render(orbit.camera(width, height))
        save_picture(args.save or (Path(args.model).stem if args.model else "knot") + ".png", image)
        return

    window.mouse_mode = "orbit"
    clock, started, fps, last = [0.0], time.perf_counter(), 0.0, time.perf_counter()
    materials = list(MATERIALS)
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            dt = window.tick()
            orbit.update(window, dt)
            for key in window.take_keys():
                if key == "d":
                    view.drop()
                elif key == "r":
                    view.reset()
                elif key in ("bracketleft", "bracketright"):
                    view.sun_elevation = float(np.clip(view.sun_elevation + (0.06 if key == "bracketright" else -0.06), 0.03, 1.5))
                    view.set_sky()
                elif key in ("comma", "period"):
                    view.sun_azimuth += 0.2 if key == "period" else -0.2
                    view.set_sky()
                elif key == "b":
                    view.beauty = not view.beauty
                    view.set_mode()
                elif key == "m" and knot_material is not None and not args.model:
                    args.material = materials[(materials.index(args.material) + 1) % len(materials)]
                    scene, knot_material = build_scene(args)
                    view = Viewer(scene, width, height)
                elif key == "p":
                    window.screenshot("model.png")
            view.advance(dt, clock)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-4)
            last = now
            hud = (f"{fps:4.0f} fps   {view.frame} frames refining   {'path traced' if view.beauty else 'classic'}\n"
                   f"{'drop physics: ' + ('falling' if view.falling else 'ready (D)') if view.world else 'no physics module'}   sun {np.degrees(view.sun_elevation):.0f} deg\n"
                   "drag orbit   wheel zoom   D drop   R reset   M material   [ ] , . sun   B beauty/classic   P picture   Esc quit")
            window.show(view.render(orbit.camera(width, height)), hud)
    if args.save:
        save_picture(args.save, view.render(orbit.camera(width, height)))


if __name__ == "__main__":
    main()
