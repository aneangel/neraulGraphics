"""A physics stage for the examples: rigid bodies simulated by neraulgraphics.physics.World (Jolt, deterministic) and drawn by the renderer.

    stage = Stage([Body("sphere", 0.5, glass), Body("box", (0.4, 0.4, 0.4), plastic)])
    renderer = ng.Renderer(..., meshes=stage.meshes() + [...], instances=stage.instances() + [...])
    stage.place(0, (0, 0, 3), velocity=(2, 0, 0))              # simulation axes: Z up
    stage.step(1 / 60)
    renderer.set_instance_transforms(stage.matrices())         # renderer axes: Y up

The bodies are made once, up front (the world cannot grow), and parked on the ground out of sight until place() puts one in the scene.
The ground is the plane z = 0 in simulation axes, which is y = 0 in the renderer's.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import neraulgraphics as ng
from neraulgraphics import physics

sys.path.insert(0, str(Path(__file__).resolve().parent / "scenery"))

# simulation (x, y, z), Z up, to renderer (x, z, -y), Y up
SIM_TO_RENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def smooth_sphere(subdivisions: int = 4) -> ng.Mesh:
    """A unit sphere with a normal at every vertex, so it shades smooth however few triangles it has (make_icosphere has none)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    points = [np.array(v, np.float64) / np.linalg.norm(v) for v in vertices]
    for _ in range(subdivisions):
        midpoint, refined = {}, []

        def middle(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                m = points[a] + points[b]
                points.append(m / np.linalg.norm(m))
                midpoint[key] = len(points) - 1
            return midpoint[key]

        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    p = np.array(points, np.float32)
    return ng.Mesh(p, np.array(faces, np.uint32), p.copy())


@dataclass
class Body:
    kind: str                       # "sphere" or "box"
    size: object                    # a sphere's radius, or a box's half extents (x, y, z)
    material: ng.Material
    density: float = 1000.0
    friction: float = 0.6
    restitution: float = 0.15


class Stage:
    def __init__(self, bodies: list[Body], gravity=(0.0, 0.0, -9.81), ground_friction: float = 0.7, substeps: int = 4, threads: int = 0) -> None:
        self.bodies, self.substeps = bodies, substeps
        self.world = physics.require()(gravity=gravity, threads=threads, max_bodies=len(bodies) + 8)
        self.world.add_ground(friction=ground_friction, restitution=0.0)
        self.half = np.ones((len(bodies), 3), np.float32)
        for i, b in enumerate(bodies):
            rest = self._parking(i, b)
            if b.kind == "sphere":
                self.world.add_sphere(float(b.size), rest, b.density, b.friction, b.restitution)
                self.half[i] = float(b.size)
            elif b.kind == "box":
                half = np.broadcast_to(np.asarray(b.size, np.float32), (3,))
                corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * half
                self.world.add_convex_hull(np.ascontiguousarray(corners), rest, b.density, b.friction, b.restitution, 0.0, False)
                self.half[i] = half
            else:
                raise ValueError(f"unknown body kind {b.kind!r}: use 'sphere' or 'box'")
        self.world.finish_setup()
        self.active = np.zeros(len(bodies), bool)

    @staticmethod
    def _parking(i: int, body: Body) -> tuple:
        """Where an unused body waits: on the ground, far from the scene."""
        height = float(body.size) if body.kind == "sphere" else float(np.max(body.size))
        return (500.0 + 3.0 * i, 0.0, height + 0.01)

    # ----------------------------------------------------------------------------------------------- the scene --

    def meshes(self) -> list[ng.Mesh]:
        """The two shared meshes, a smooth unit sphere (index 0) and a unit cube (index 1): bodies scale them by their matrices."""
        return [smooth_sphere(4), ng.make_box((1.0, 1.0, 1.0))]

    def instances(self, first_mesh: int = 0, textures: dict | None = None) -> list[ng.Instance]:
        """One instance per body, in order, of the stage's meshes (which must sit at first_mesh and first_mesh + 1)."""
        return [ng.Instance(first_mesh + (0 if b.kind == "sphere" else 1), material=b.material) for b in self.bodies]

    def place(self, index: int, position, velocity=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)) -> None:
        """Puts body index in the scene at position (simulation axes) with a velocity and a spin."""
        self.world.fire(index, tuple(float(v) for v in position), tuple(float(v) for v in velocity), tuple(float(v) for v in angular))
        self.active[index] = True

    def park(self, index: int) -> None:
        self.world.fire(index, self._parking(index, self.bodies[index]), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.active[index] = False

    def park_all(self) -> None:
        for i in range(len(self.bodies)):
            self.park(i)

    def step(self, dt: float) -> None:
        self.world.step(dt, self.substeps)

    def state_hash(self) -> int:
        return self.world.state_hash()

    # --------------------------------------------------------------------------------------------- the picture --

    def poses(self) -> np.ndarray:
        """(bodies, 7): position then quaternion x y z w, in simulation axes."""
        return self.world.poses()

    def matrices(self) -> np.ndarray:
        """(bodies, 3, 4) instance matrices in renderer axes, scale included.

        The meshes are drawn in the renderer's axes and a body's rotation R is in simulation axes, so the matrix is P R P^T S: the mesh's own
        axes taken into simulation axes, turned by R, and brought back by P (P being the axis change). S scales along the mesh's axes, so a
        box's half extents (x, y, z in simulation) become (x, z, y) there."""
        from neraulgraphics.shapes import quat_to_matrix

        poses = self.poses()
        rotation = quat_to_matrix(poses[:, 3:7])
        scale = self.half[:, [0, 2, 1]]
        out = np.empty((len(poses), 3, 4))
        out[:, :, :3] = np.einsum("ij,njk,kl->nil", SIM_TO_RENDER, rotation, SIM_TO_RENDER.T) * scale[:, None, :]
        out[:, :, 3] = poses[:, :3] @ SIM_TO_RENDER.T
        return out.astype(np.float32)

    def positions(self) -> np.ndarray:
        """Body positions in renderer axes (Y up)."""
        return self.poses()[:, :3] @ SIM_TO_RENDER.T


class Scene:
    """A path-traced scene with a textured ground, a physical sky and a Stage of bodies: what the interactive examples share.

    scene = Scene(1280, 720, bodies)         # the bodies are parked until place()d
    scene.advance(dt, clock)                 # real seconds in: fixed 1/60 s physics steps, the result handed to the renderer
    scene.throw(index, origin, direction)    # renderer axes: sends a body flying, spinning
    image = scene.render(camera)             # refines while nothing moves
    """

    def __init__(self, width: int, height: int, bodies: list[Body], *, hdri: str | None = None, sun: tuple = (0.62, 2.4), device: str = "auto",
                 extra_meshes: list | None = None, extra_instances: list | None = None, textures: list | None = None) -> None:
        from textures import concrete, textured_box

        self.width, self.height, self.hdri = width, height, hdri
        colour, normal, orm = concrete(seed=7, stains=0.05, streaks=0.03)         # calm: on a ground this size, slow stains would show the repeat
        tex = [ng.Texture(colour), ng.Texture(normal, linear=True), ng.Texture(orm, linear=True)] + list(textures or [])
        vertices, triangles, normals, uvs = textured_box((400.0, 0.5, 400.0), tile=1.6)
        ground = ng.Mesh(vertices, triangles, normals, uvs)
        material = ng.Material(diffuse_color=(0.85, 0.84, 0.82), roughness=1.0, metallic=0.0)
        self.stage = Stage(bodies)
        meshes = [ground] + self.stage.meshes() + list(extra_meshes or [])
        instances = ([ng.Instance(0, ng.make_transform((0.0, -0.5, 0.0)), material, texture=0, normal_texture=1, orm_texture=2)]
                     + self.stage.instances(first_mesh=1) + list(extra_instances or []))
        self.ground_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, -0.5], [0, 0, 1, 0]], np.float32)
        self.extra_matrices = np.array([i.transform.matrix for i in (extra_instances or [])], np.float32).reshape(-1, 3, 4)
        self.renderer = ng.Renderer(width, height, [], [], meshes=meshes, instances=instances, textures=tex, device=device)
        self.sun_elevation, self.sun_azimuth, self.exposure, self.turbidity = sun[0], sun[1], 0.8, 2.6
        self.frame, self.slow, self.next_slot = 0, False, 0
        self.set_sky()
        self.set_exposure()

    # ------------------------------------------------------------------------------------------------- lighting --

    def set_sky(self) -> None:
        if self.hdri:
            sky = ng.load_hdr(self.hdri)
            lights = []
        else:
            # the sky without its sun, and the sun as an explicit light: a highlight is then computed from the light's direction with a shadow ray,
            # not found by a random ray landing on a small disc, so it does not twinkle as the view moves
            sun = dict(sun_elevation=self.sun_elevation, sun_azimuth=self.sun_azimuth, turbidity=self.turbidity, sun_strength=4.0, sun_radius=0.02)
            sky = ng.physical_sky(sky_ratio=0.22, ground_albedo=0.25, width=768, height=384, include_sun=False, **sun)
            lights = [ng.sun_light(**sun)]
        self.renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
        self.renderer.set_lights(lights)
        self.frame = 0

    def set_exposure(self) -> None:
        self.renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=12, exposure=self.exposure, denoise=True, denoise_iterations=3, max_radiance=60.0))
        self.frame = 0

    # ------------------------------------------------------------------------------------------------- physics --

    def push(self) -> None:
        """Hands the bodies' poses to the renderer."""
        self.renderer.set_instance_transforms(np.concatenate([self.ground_matrix[None], self.stage.matrices(), self.extra_matrices]))

    def advance(self, dt: float, clock: list) -> bool:
        """Steps the physics by real time dt in fixed 1/60 s steps and hands the result to the renderer. Returns whether anything is moving."""
        clock[0] += dt * (0.25 if self.slow else 1.0)
        steps = 0
        while clock[0] >= 1 / 60 and steps < 4:
            self.stage.step(1 / 60)
            clock[0] -= 1 / 60
            steps += 1
        if steps:
            self.push()
        if not self.stage.active.any():
            return False
        return steps > 0 and float(np.abs(self.stage.world.velocities()[self.stage.active, :3]).max()) > 0.05

    def throw(self, index: int, origin, direction, speed: float = 14.0) -> None:
        """Sends body index flying from origin (renderer axes) along direction, spinning."""
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        to_sim = SIM_TO_RENDER.T
        self.stage.place(index, to_sim @ np.asarray(origin, float), velocity=to_sim @ (d * speed), angular=np.random.default_rng(index).normal(size=3) * 4.0)

    def render(self, camera, samples: int | None = None) -> np.ndarray:
        image = self.renderer.render(camera, self.frame, samples or (4 if self.renderer.device == "cuda" else 1))     # a few paths a pixel keeps motion clean
        self.frame += 1
        return image
