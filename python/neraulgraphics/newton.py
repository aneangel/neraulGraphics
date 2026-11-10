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

__all__ = ["NewtonRenderer"]

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
