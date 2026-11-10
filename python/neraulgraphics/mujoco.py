"""Render a `MuJoCo <https://github.com/google-deepmind/mujoco>`_ simulation.

::

    import mujoco
    from neraulgraphics.mujoco import MuJoCoRenderer

    model = mujoco.MjModel.from_xml_path("humanoid.xml")
    data = mujoco.MjData(model)
    view = MuJoCoRenderer(model, 1280, 720)
    ...
    mujoco.mj_step(model, data)
    image = view.render(data)             # (720, 1280, 3) uint8

Geoms become mesh instances once, when the renderer is made; each ``render`` call only
reads ``data.geom_xpos`` and ``data.geom_xmat`` and moves them. The same adapter serves
`mujoco_warp` and anything else that fills an ``MjData``.

What is drawn: plane, sphere, capsule, ellipsoid, cylinder, box and mesh geoms in the
groups you ask for (MuJoCo's default is 0 to 2, so group 3 and up, often collision
shapes, stay hidden). Colours come from ``geom_rgba`` or the geom's material; textures,
heightfields and SDFs are not drawn.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence

import mujoco
import numpy as np

from ._adapter import SceneBuilder, SimRenderer, scene_bounds
from ._core import Light, Mesh
from .shapes import capsule, checker_texture, cylinder, matrices_from_rotations, plane

__all__ = ["MuJoCoRenderer"]

_GEOM = mujoco.mjtGeom
_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
# MuJoCo's own convention for a plane with size 0: it extends forever.
_INFINITE_PLANE = 1000.0
_GROUND_TILE = 2.0  # one checkerboard repeat per this many metres


class MuJoCoRenderer(SimRenderer):
    """Draws an :class:`mujoco.MjModel` and its :class:`mujoco.MjData` with the raytracer.

    Args:
        model: A compiled MuJoCo model.
        width, height: Image size in pixels.
        geom_groups: Which geom groups to draw. Defaults to 0, 1 and 2, as MuJoCo's viewer does.
        lights: Lights to use. By default a key and a fill light are placed from the scene's size.
        ambient: Light that reaches every surface regardless of the lights, as a fraction of its colour.
        fov: Vertical field of view of the default camera, in radians.
    """

    def __init__(self, model: mujoco.MjModel, width: int = 1280, height: int = 720, *,
                 geom_groups: Collection[int] = (0, 1, 2), lights: Sequence[Light] | None = None,
                 fov: float = 1.0, ambient: float = 0.2) -> None:
        builder = SceneBuilder()
        keep: list[int] = []
        scales: list[np.ndarray] = []
        radii: list[float] = []
        skipped: set[str] = set()

        for g in range(model.ngeom):
            kind = int(model.geom_type[g])
            rgba = model.geom_rgba[g]
            if model.geom_group[g] not in geom_groups or rgba[3] <= 0.0:
                continue
            made = self._geom_mesh(builder, model, g, kind)
            if made is None:
                skipped.add(_GEOM(kind).name)
                continue
            mesh_index, scale, radius = made
            if model.geom_matid[g] >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
                rgba = model.mat_rgba[model.geom_matid[g]]  # the geom takes its material's colour
            if kind == _GEOM.mjGEOM_PLANE:
                # A checkerboard in the geom's own colour, so the ground shows scale and motion.
                base = np.asarray(rgba[:3], dtype=np.float64)
                builder.add(mesh_index, (1.0, 1.0, 1.0), builder.texture(
                    ("ground", tuple(base)), lambda: checker_texture(base * 0.85, np.minimum(base * 1.15, 1.0))))
            else:
                builder.add(mesh_index, rgba[:3])
            keep.append(g)
            scales.append(scale)
            radii.append(radius)
        if skipped:
            warnings.warn(f"neraulgraphics: geoms of type {sorted(skipped)} are not drawn", stacklevel=2)

        self._geoms = np.array(keep, dtype=np.int64)
        self._scale = np.array(scales, dtype=np.float64).reshape(-1, 3)

        # Frame the scene from where the geoms start, ignoring planes (they are the world).
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        reach = np.array(radii) * (np.abs(self._scale).max(axis=1, initial=0.0) if len(radii) else 0.0)
        position = probe.geom_xpos[self._geoms]
        solid = reach > 0
        center, extent = scene_bounds(position[solid], reach[solid])
        super().__init__(width, height, builder, up=(0.0, 0.0, 1.0), center=center, extent=extent, lights=lights,
                         fov=fov, ambient=ambient)
        self._place(probe)

    @staticmethod
    def _geom_mesh(builder: SceneBuilder, model: mujoco.MjModel, g: int, kind: int
                   ) -> tuple[int, np.ndarray, float] | None:
        """(mesh index, instance scale, bounding radius of the unscaled mesh) for geom ``g``, or None."""
        size = [float(v) for v in model.geom_size[g]]
        one = np.ones(3)
        if kind == _GEOM.mjGEOM_SPHERE:
            return builder.unit_sphere(), one * size[0], 1.0
        if kind == _GEOM.mjGEOM_ELLIPSOID:
            return builder.unit_sphere(), np.array(size), 1.0
        if kind == _GEOM.mjGEOM_BOX:
            return builder.unit_box(), np.array(size), float(np.sqrt(3.0))
        if kind == _GEOM.mjGEOM_CYLINDER:
            return builder.mesh("unit_cylinder", cylinder, np.sqrt(2.0)), np.array([size[0], size[0], size[1]]), np.sqrt(2.0)
        if kind == _GEOM.mjGEOM_CAPSULE:
            radius = size[0] + size[1]
            return builder.mesh(("capsule", size[0], size[1]), lambda: capsule(size[0], size[1]), radius), one, radius
        if kind == _GEOM.mjGEOM_PLANE:
            width = 2.0 * size[0] if size[0] > 0.0 else _INFINITE_PLANE
            length = 2.0 * size[1] if size[1] > 0.0 else _INFINITE_PLANE
            return builder.mesh(("plane", width, length), lambda: plane(width, length, _GROUND_TILE), 0.0), one, 0.0
        if kind == _GEOM.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            vertices = np.asarray(model.mesh_vert[v0:v0 + nv], dtype=np.float32)
            faces = np.asarray(model.mesh_face[f0:f0 + nf], dtype=np.uint32)
            radius = float(np.linalg.norm(vertices, axis=1).max())
            return builder.mesh(("mesh", mesh), lambda: Mesh(vertices, faces), radius), one, radius
        return None

    def _place(self, data: mujoco.MjData) -> None:
        self.renderer.set_instance_transforms(self._matrices(data))

    def _matrices(self, data: mujoco.MjData) -> np.ndarray:
        rotation = np.asarray(data.geom_xmat[self._geoms], dtype=np.float64).reshape(-1, 3, 3)
        return matrices_from_rotations(rotation, data.geom_xpos[self._geoms], self._scale)

    def render(self, data: mujoco.MjData, camera=None, *, frame: int = 0, samples_per_frame: int = 1) -> np.ndarray:
        """Draws ``data`` as a ``(height, width, 3)`` uint8 image.

        ``data.geom_xpos`` must be current, so call ``mj_forward`` or ``mj_step`` first. Pass
        ``frame=0`` for each new state; with a fixed camera and a paused simulation,
        increasing ``frame`` sharpens the image.
        """
        return self._render(self._matrices(data), camera, frame, samples_per_frame)
