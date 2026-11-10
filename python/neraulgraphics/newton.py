"""Render a `Newton <https://github.com/newton-physics/newton>`_ simulation.

::

    import newton
    from neraulgraphics.newton import NewtonRenderer

    model = builder.finalize()
    view = NewtonRenderer(model, 1280, 720)
    ...
    solver.step(state_0, state_1, control, contacts, dt)
    image = view.render(state_1)          # (720, 1280, 3) uint8

The model's shapes become mesh instances once, when the renderer is made. Each
``render`` call then only reads the state's body poses and moves those instances, which
is the cheap update the two-level BVH is built for. Nothing here imports Newton at
module level except ``newton`` itself, so importing it needs Newton installed.

What is drawn: sphere, box, capsule, cylinder, cone, ellipsoid, plane and mesh shapes
that carry the ``VISIBLE`` flag, with colours from ``model.shape_color``. Heightfields and
Gaussians are skipped, and opacity is ignored.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

import newton

from ._adapter import SceneBuilder, SimRenderer, scene_bounds
from ._core import Light, Mesh
from .shapes import capsule, checker_texture, compose_poses, cone, cylinder, plane, pose_matrices

__all__ = ["NewtonBatchRenderer", "NewtonRenderer"]

_GEO = newton.GeoType
_FALLBACK_COLOR = (0.62, 0.64, 0.70)
# The ground is a checkerboard of these two, one repeat per _GROUND_TILE metres (so cells are half that).
_GROUND_COLORS = ((0.30, 0.33, 0.38), (0.40, 0.43, 0.48))
_GROUND_TILE = 2.0
# What the Newton viewer uses for a plane with no size ("infinite").
_INFINITE_PLANE = 1000.0
_UP = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}


def _shape_mesh(builder: SceneBuilder, geo: int, scale: np.ndarray, source: object | None
                ) -> tuple[int, np.ndarray] | None:
    """The mesh index for one shape and the scale to give its instance, or None if it is not drawn.

    Primitives made at unit size are shared and scaled by the instance; only those whose
    shape changes with size (capsules, planes) get a mesh of their own.
    """
    s = [float(v) for v in scale]
    one = np.ones(3)
    if geo == _GEO.SPHERE:
        return builder.unit_sphere(), one * s[0]
    if geo == _GEO.BOX:
        return builder.unit_box(), np.array(s[:3]) if len(s) >= 3 else one * s[0]
    if geo == _GEO.ELLIPSOID:
        return builder.unit_sphere(), np.array(s[:3])
    if geo == _GEO.CYLINDER:
        return builder.mesh("unit_cylinder", cylinder, np.sqrt(2.0)), np.array([s[0], s[0], s[1]])
    if geo == _GEO.CONE:
        return builder.mesh("unit_cone", cone, np.sqrt(2.0)), np.array([s[0], s[0], s[1]])
    if geo == _GEO.CAPSULE:
        return builder.mesh(("capsule", s[0], s[1]), lambda: capsule(s[0], s[1]), s[0] + s[1]), one
    if geo == _GEO.PLANE:
        width = s[0] if s[0] > 0.0 else _INFINITE_PLANE
        length = s[1] if len(s) > 1 and s[1] > 0.0 else _INFINITE_PLANE
        # Radius 0: a plane is the world, not the subject, so it does not count when framing the scene.
        return builder.mesh(("plane", width, length), lambda: plane(width, length, _GROUND_TILE), 0.0), one
    if geo in (_GEO.MESH, _GEO.CONVEX_MESH) and source is not None:
        # A source mesh is shared by every shape using it, whatever its scale; the scale
        # (possibly negative, which mirrors it) goes on the instance.
        vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(source.indices, dtype=np.uint32).reshape(-1, 3)
        radius = float(np.linalg.norm(vertices, axis=1).max())
        return builder.mesh(("mesh", id(source)), lambda: Mesh(vertices, triangles), radius), np.array(s[:3])
    return None


class NewtonRenderer(SimRenderer):
    """Draws a :class:`newton.Model` and its :class:`newton.State` with the raytracer.

    Args:
        model: A finalized Newton model.
        width, height: Image size in pixels.
        lights: Lights to use. By default a key and a fill light are placed from the scene's size.
        ambient: Light that reaches every surface regardless of the lights, as a fraction of its colour.
        device: "auto" (a GPU if there is one, else the CPU), "cuda" or "cpu".
        world_offsets: Optional ``(world_count, 3)`` offsets, so the worlds of a batched
            simulation sit side by side instead of on top of each other.
        fov: Vertical field of view of the default camera, in radians.
    """

    def __init__(self, model: newton.Model, width: int = 1280, height: int = 720, *,
                 lights: Sequence[Light] | None = None, world_offsets: np.ndarray | None = None,
                 fov: float = 1.0, ambient: float = 0.2, device: str = "auto") -> None:
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        flags = model.shape_flags.numpy()
        colors = model.shape_color.numpy() if model.shape_color is not None else None

        builder = SceneBuilder()
        keep: list[int] = []
        scales: list[np.ndarray] = []
        meshes_used: list[int] = []
        skipped: set[str] = set()
        for s in range(len(shape_type)):
            if not flags[s] & int(newton.ShapeFlags.VISIBLE):
                continue
            geo = int(shape_type[s])
            made = _shape_mesh(builder, geo, shape_scale[s], model.shape_source[s])
            if made is None:
                skipped.add(_GEO(geo).name)
                continue
            mesh_index, scale = made
            if geo == _GEO.PLANE:  # a checkerboard whatever the model's colour, so the ground shows scale and motion
                builder.add(mesh_index, (1.0, 1.0, 1.0), builder.texture("ground", lambda: checker_texture(*_GROUND_COLORS)))
            else:
                builder.add(mesh_index, colors[s] if colors is not None else _FALLBACK_COLOR)
            keep.append(s)
            scales.append(scale)
            meshes_used.append(mesh_index)
        if skipped:
            warnings.warn(f"neraulgraphics: shapes of type {sorted(skipped)} are not drawn", stacklevel=2)

        self._shapes = np.array(keep, dtype=np.int64)
        self._scale = np.array(scales, dtype=np.float64).reshape(-1, 3)
        transform = model.shape_transform.numpy()[self._shapes]
        self._local_p, self._local_q = transform[:, :3].astype(np.float64), transform[:, 3:7].astype(np.float64)
        self._parent = model.shape_body.numpy()[self._shapes]
        self._world = model.shape_world.numpy()[self._shapes] if model.shape_world is not None else None
        self._offsets = None if world_offsets is None else np.asarray(world_offsets, dtype=np.float64).reshape(-1, 3)
        self._initial_body_q = model.body_q.numpy() if model.body_q is not None and model.body_q.shape[0] else None

        matrices = self._matrices(self._initial_body_q)
        # Framing comes from the placed shapes' bounding spheres. Planes have radius 0 so
        # they do not count: they are as big as the world, not what the camera is for.
        reach = np.array([builder.radius[m] for m in meshes_used]) * np.abs(self._scale).max(axis=1, initial=0.0)
        center, extent = scene_bounds(matrices[:, :, 3][reach > 0], reach[reach > 0])
        super().__init__(width, height, builder, up=_UP[int(model.up_axis)], center=center, extent=extent,
                         lights=lights, fov=fov, ambient=ambient, device=device)
        self.renderer.set_instance_transforms(matrices)

    def _matrices(self, body_q: np.ndarray | None) -> np.ndarray:
        """World matrices of every drawn shape: body pose composed with the shape's own, then scaled."""
        n = len(self._shapes)
        if body_q is None or not (self._parent >= 0).any():
            position, quat = self._local_p, self._local_q
        else:
            has_body = self._parent >= 0
            body = np.asarray(body_q, dtype=np.float64)[np.maximum(self._parent, 0)]
            body_p = np.where(has_body[:, None], body[:, :3], 0.0)
            body_q_ = np.where(has_body[:, None], body[:, 3:7], np.array([0.0, 0.0, 0.0, 1.0]))
            position, quat = compose_poses(body_p, body_q_, self._local_p, self._local_q)
        if self._offsets is not None and self._world is not None:
            in_world = (self._world >= 0) & (self._world < len(self._offsets))
            position = position + np.where(in_world[:, None], self._offsets[np.clip(self._world, 0, None)], 0.0)
        return pose_matrices(position, quat, self._scale) if n else np.zeros((0, 3, 4), dtype=np.float32)

    def render(self, state: newton.State | None = None, camera=None, *, frame: int = 0,
               samples_per_frame: int = 1) -> np.ndarray:
        """Draws ``state`` (default: the model's initial pose) as a ``(height, width, 3)`` uint8 image.

        Pass ``frame=0`` for each new state, since the picture built up so far shows the old one.
        If the camera is fixed and the simulation paused, increasing ``frame`` sharpens the image.
        """
        return self._render(self._placed(state), camera, frame, samples_per_frame)

    def _placed(self, state: newton.State | None) -> np.ndarray:
        """The shapes' world matrices for ``state`` (default: the model's initial pose)."""
        if state is not None and state.body_q is not None and state.body_q.shape[0]:
            return self._matrices(state.body_q.numpy())
        return self._matrices(self._initial_body_q)

    def update(self, state: newton.State | None = None) -> None:
        """Moves the shapes to ``state`` without drawing anything, so :meth:`cast_rays` sees it."""
        matrices = self._placed(state)
        if len(matrices):
            self.renderer.set_instance_transforms(matrices)

    def render_frame(self, state: newton.State | None = None, camera=None, frame: int = 0, samples_per_frame: int = 1, *,
                     rgb: bool = True, depth: bool = False, normals: bool = False, object_ids: bool = False,
                     zero_copy: bool = False) -> dict:
        """Draws ``state`` into the images asked for, as a dict: ``rgb``, ``depth``, ``normals``, ``object_ids``.

        The sensor outputs of a robotics or RL pipeline. depth is along the view axis and ``inf`` where
        nothing is hit; object_ids number the shapes as :attr:`object_id_of_shape` says. With
        ``zero_copy=True`` the arrays are views of the renderer's own buffers (CUDA arrays on a GPU, for
        ``torch.from_dlpack`` and the like), overwritten by the next render. See
        :meth:`neraulgraphics.Renderer.render_frame`.
        """
        return self._render_frame(self._placed(state), camera, frame, samples_per_frame,
                                  dict(rgb=rgb, depth=depth, normals=normals, object_ids=object_ids, zero_copy=zero_copy))

    @property
    def object_id_of_shape(self) -> dict[int, int]:
        """Which object ID each drawn Newton shape index has in an ``object_ids`` image or a ray query."""
        from ._core import OBJECT_INSTANCE
        return {int(shape): OBJECT_INSTANCE + i for i, shape in enumerate(self._shapes)}


class NewtonBatchRenderer:
    """Draws a batched Newton simulation: many worlds, each seen by its own camera(s), in one call.

    ::

        builder.replicate(robot, world_count=4096)         # identical worlds, as Newton batches them
        model = builder.finalize()
        view = NewtonBatchRenderer(model, 64, 64)
        ...
        solver.step(state_0, state_1, control, contacts, dt)
        images = view.render(state_1, depth=True, zero_copy=True)

    The shapes of one world become instances once; each ``render`` hands ``state.body_q`` to the renderer as
    it is, so on a GPU the simulator's state goes to the images and the images to a policy without touching
    the CPU. Every world must hold the same shapes on the same bodies, as ``replicate`` makes them (this is
    checked); shapes that belong to no world, such as a ground plane, appear in every world. A camera
    sees only its own world, so worlds that overlap in space (as Newton leaves them) do not show through one
    another.

    Args:
        model: A finalized Newton model with ``world_count`` identical worlds.
        width, height: Size of each view in pixels.
        cameras_per_world: Cameras looking at each world.
        lights, fov, ambient, device: As for :class:`NewtonRenderer`.
    """

    def __init__(self, model: newton.Model, width: int = 128, height: int = 128, cameras_per_world: int = 1, *,
                 lights: Sequence[Light] | None = None, fov: float = 1.0, ambient: float = 0.2,
                 device: str = "auto") -> None:
        from ._core import BatchRenderer, Instance, Transform
        from ._adapter import SceneBuilder, _material

        worlds = int(model.world_count)
        bodies = int(model.body_count)
        if bodies % worlds:
            raise ValueError(f"the model has {bodies} bodies in {worlds} worlds, which is not the same number in each")
        per_world_bodies = bodies // worlds
        if bodies and not np.array_equal(model.body_world.numpy(), np.repeat(np.arange(worlds), per_world_bodies)):
            raise ValueError("the bodies are not stored world by world (all of world 0's, then world 1's, and so on)")

        shape_type, shape_scale = model.shape_type.numpy(), model.shape_scale.numpy()
        flags, shape_world = model.shape_flags.numpy(), model.shape_world.numpy()
        shape_body, transform = model.shape_body.numpy(), model.shape_transform.numpy()
        colors = model.shape_color.numpy() if model.shape_color is not None else None

        builder = SceneBuilder()
        # What each shape is, in a form two worlds can be compared by.
        signatures: dict[int, list] = {}
        entries: dict[int, list] = {}
        for s in range(len(shape_type)):
            if not flags[s] & int(newton.ShapeFlags.VISIBLE):
                continue
            made = _shape_mesh(builder, int(shape_type[s]), shape_scale[s], model.shape_source[s])
            if made is None:
                continue
            w = int(shape_world[s])
            body = int(shape_body[s])
            local_body = body - w * per_world_bodies if (body >= 0 and w >= 0) else body
            mesh_index, scale = made
            color = colors[s] if colors is not None else _FALLBACK_COLOR
            entries.setdefault(w, []).append((s, mesh_index, scale, local_body))
            signatures.setdefault(w, []).append((mesh_index, tuple(np.round(scale, 5)), local_body,
                                                 tuple(np.round(transform[s], 5)), tuple(np.round(color, 4)),
                                                 int(shape_type[s])))
        template = signatures.get(0, [])
        for w in range(1, worlds):
            if signatures.get(w, []) != template:
                raise ValueError(f"world {w} differs from world 0 (its shapes, sizes, colours, bodies or offsets), "
                                 "but a batch needs identical worlds, as Newton's replicate() makes them")

        instances, body_of, local, scales = [], [], [], []
        for w in (-1, 0):  # shapes that belong to no world, then the shapes of one world
            for s, mesh_index, scale, local_body in entries.get(w, []):
                is_plane = int(shape_type[s]) == _GEO.PLANE
                texture = builder.texture("ground", lambda: checker_texture(*_GROUND_COLORS)) if is_plane else -1
                color = (1.0, 1.0, 1.0) if is_plane else (colors[s] if colors is not None else _FALLBACK_COLOR)
                instances.append(Instance(mesh_index, Transform(np.eye(4, dtype=np.float32)), _material(color), texture))
                body_of.append(local_body)
                local.append(transform[s])
                scales.append(scale)
        if not instances:
            raise ValueError("the model has no visible shapes to draw")

        self._worlds, self._cameras_per_world = worlds, cameras_per_world
        self.width, self.height, self.fov = width, height, fov
        self.up = np.asarray(_UP[int(model.up_axis)], dtype=np.float64)

        # Where to look: the middle and size of the first world's shapes in the model's initial pose.
        initial = model.body_q.numpy() if per_world_bodies else np.zeros((0, 7))
        first = initial[:per_world_bodies]
        positions, reach = [], []
        for (s, mesh_index, scale, local_body), p_local in zip(entries.get(0, []), [transform[e[0]] for e in entries.get(0, [])]):
            if local_body >= 0:
                p, _ = compose_poses(first[local_body:local_body + 1, :3], first[local_body:local_body + 1, 3:7],
                                     p_local[None, :3], p_local[None, 3:7])
            else:
                p = p_local[None, :3]
            positions.append(p[0])
            reach.append(builder.radius[mesh_index] * float(np.abs(scale).max()))
        center, extent = scene_bounds(np.array(positions).reshape(-1, 3), np.array(reach))
        self.center, self.extent = center, max(extent, 1e-3)

        self.renderer = BatchRenderer(width, height, worlds, cameras_per_world, [], list(lights) if lights is not None else
                                      self._default_lights(), meshes=builder.meshes, instances=instances,
                                      textures=builder.textures, device=device)
        self.renderer.ambient = ambient
        self.renderer.bind_bodies(np.asarray(body_of, np.int32), np.asarray(local, np.float32).reshape(-1, 7),
                                  np.asarray(scales, np.float32).reshape(-1, 3), per_world_bodies)
        self._model = model
        self._has_bodies = per_world_bodies > 0
        self._pose_static()

    def _default_lights(self) -> list[Light]:
        stub = SimRenderer.__new__(SimRenderer)
        stub.up, stub.center, stub.extent = self.up, self.center, self.extent
        return SimRenderer.default_lights(stub)

    def _pose_static(self) -> None:
        # Worlds with no bodies (everything fixed) are placed once, from an empty set of poses.
        if not self._has_bodies:
            self.renderer.set_body_poses(np.zeros((self._worlds, 0, 7), np.float32))

    @property
    def view_count(self) -> int:
        return self._worlds * self._cameras_per_world

    def default_cameras(self) -> np.ndarray:
        """One camera per view, framing the first world's shapes from the side, slightly above: ``(views, 3, 4)``."""
        from .shapes import look_at
        h = np.cross(self.up, np.cross([0.0, 1.0, 0.0] if abs(self.up[1]) < 0.9 else [1.0, 0.0, 0.0], self.up))
        h /= np.linalg.norm(h)
        eye = self.center - h * 2.6 * self.extent + self.up * 0.7 * self.extent
        return np.tile(look_at(eye, self.center, self.up), (self.view_count, 1, 1))

    def update(self, state: newton.State | None = None) -> None:
        """Places every instance of every world from ``state`` (default: the model's initial pose)."""
        if not self._has_bodies:
            return
        body_q = state.body_q if state is not None else self._initial_body_q()
        # A Warp array: handed over as it is, so on a GPU it is read in place with no copy. A renderer on the CPU
        # cannot read GPU memory, so it gets the copy it needs.
        self.renderer.set_body_poses(body_q.numpy() if self.renderer.device == "cpu" else body_q)

    def _initial_body_q(self):
        return self._model.body_q

    def render(self, state: newton.State, cameras=None, *, fov: float | None = None, rgb: bool = True, depth: bool = False,
               normals: bool = False, object_ids: bool = False, zero_copy: bool = False, frame: int = 0,
               samples_per_pixel: int = 1) -> dict:
        """Draws every world of ``state``: a dict of ``rgb``, ``depth``, ``normals`` and ``object_ids`` batches.

        ``cameras`` is ``(views, 3, 4)`` camera-to-world matrices (NumPy or a CUDA array), the OpenGL convention;
        the default frames each world. With ``zero_copy=True`` the images are CUDA arrays for
        ``torch.from_dlpack`` and the like, overwritten by the next render. See
        :meth:`neraulgraphics.BatchRenderer.render`.
        """
        self.update(state)
        return self.renderer.render(self.default_cameras() if cameras is None else cameras, self.fov if fov is None else fov,
                                    samples_per_pixel, rgb=rgb, depth=depth, normals=normals, object_ids=object_ids,
                                    frame=frame, zero_copy=zero_copy)

    def cast_rays(self, origins, directions, max_distance: float = float("inf")) -> dict:
        """Ray queries in every world: ``(world_count, rays_per_world, 3)`` arrays. See :meth:`neraulgraphics.BatchRenderer.cast_rays`."""
        return self.renderer.cast_rays(origins, directions, max_distance)
