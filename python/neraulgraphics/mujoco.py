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

__all__ = ["MJWarpBatchRenderer", "MuJoCoRenderer"]

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
        device: "auto" (a GPU if there is one, else the CPU), "cuda" or "cpu".
        fov: Vertical field of view of the default camera, in radians.
    """

    def __init__(self, model: mujoco.MjModel, width: int = 1280, height: int = 720, *,
                 geom_groups: Collection[int] = (0, 1, 2), lights: Sequence[Light] | None = None,
                 fov: float = 1.0, ambient: float = 0.2, device: str = "auto") -> None:
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
                         fov=fov, ambient=ambient, device=device)
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

    def update(self, data: mujoco.MjData) -> None:
        """Moves the geoms to ``data`` without drawing anything, so :meth:`cast_rays` sees it."""
        self._place(data)

    def render_frame(self, data: mujoco.MjData, camera=None, frame: int = 0, samples_per_frame: int = 1, *, rgb: bool = True,
                     depth: bool = False, normals: bool = False, object_ids: bool = False, zero_copy: bool = False) -> dict:
        """Draws ``data`` into the images asked for, as a dict: ``rgb``, ``depth``, ``normals``, ``object_ids``.

        The sensor outputs of a robotics or RL pipeline. depth is along the view axis and ``inf`` where
        nothing is hit; :attr:`object_id_of_geom` says which geom each ID is. With ``zero_copy=True`` the
        arrays are views of the renderer's own buffers (CUDA arrays on a GPU, for ``torch.from_dlpack`` and
        the like), overwritten by the next render. See :meth:`neraulgraphics.Renderer.render_frame`.
        """
        return self._render_frame(self._matrices(data), camera, frame, samples_per_frame,
                                  dict(rgb=rgb, depth=depth, normals=normals, object_ids=object_ids, zero_copy=zero_copy))

    @property
    def object_id_of_geom(self) -> dict[int, int]:
        """Which object ID each drawn MuJoCo geom index has in an ``object_ids`` image or a ray query."""
        from ._core import OBJECT_INSTANCE
        return {int(geom): OBJECT_INSTANCE + i for i, geom in enumerate(self._geoms)}

    def render(self, data: mujoco.MjData, camera=None, *, frame: int = 0, samples_per_frame: int = 1) -> np.ndarray:
        """Draws ``data`` as a ``(height, width, 3)`` uint8 image.

        ``data.geom_xpos`` must be current, so call ``mj_forward`` or ``mj_step`` first. Pass
        ``frame=0`` for each new state; with a fixed camera and a paused simulation,
        increasing ``frame`` sharpens the image.
        """
        return self._render(self._matrices(data), camera, frame, samples_per_frame)


class MJWarpBatchRenderer:
    """Draws `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ worlds: thousands at once, each with its own camera.

    ::

        import mujoco_warp as mjw
        model, data = mjw.put_model(mjm), mjw.make_data(mjm, nworld=4096)
        view = MJWarpBatchRenderer(mjm, 4096, 64, 64)
        ...
        mjw.step(model, data)
        images = view.render(data, depth=True, zero_copy=True)   # CUDA arrays, ready for a policy

    The geoms of the model become instances once. Each ``render`` reads ``data.geom_xpos`` and
    ``data.geom_xmat``, which live on the GPU, composes them with each geom's size in one Warp kernel, and
    hands the result to the renderer: nothing passes through the CPU. A camera sees only its own world.

    Args:
        model: The compiled MuJoCo model the Warp model was made from.
        world_count: The ``nworld`` of the Warp data.
        width, height: Size of each view in pixels.
        cameras_per_world: Cameras looking at each world.
        geom_groups, lights, fov, ambient, device: As for :class:`MuJoCoRenderer`.
    """

    _kernel = None

    def __init__(self, model: mujoco.MjModel, world_count: int, width: int = 128, height: int = 128,
                 cameras_per_world: int = 1, *, geom_groups: Collection[int] = (0, 1, 2),
                 lights: Sequence[Light] | None = None, fov: float = 1.0, ambient: float = 0.2,
                 device: str = "auto") -> None:
        import warp as wp
        from ._adapter import SceneBuilder, _material
        from ._core import BatchRenderer, Instance, Transform
        from .shapes import look_at

        builder = SceneBuilder()
        geoms, instances, scales, radii = [], [], [], []
        for g in range(model.ngeom):
            rgba = model.geom_rgba[g]
            if model.geom_group[g] not in geom_groups or rgba[3] <= 0.0:
                continue
            made = MuJoCoRenderer._geom_mesh(builder, model, g, int(model.geom_type[g]))
            if made is None:
                continue
            mesh_index, scale, radius = made
            if model.geom_matid[g] >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
                rgba = model.mat_rgba[model.geom_matid[g]]
            texture = -1
            color = rgba[:3]
            if int(model.geom_type[g]) == _GEOM.mjGEOM_PLANE:
                base = np.asarray(rgba[:3], dtype=np.float64)
                texture = builder.texture(("ground", tuple(base)), lambda: checker_texture(base * 0.85, np.minimum(base * 1.15, 1.0)))
                color = (1.0, 1.0, 1.0)
            instances.append(Instance(mesh_index, Transform(np.eye(4, dtype=np.float32)), _material(color), texture))
            geoms.append(g)
            scales.append(scale)
            radii.append(radius)
        if not instances:
            raise ValueError("the model has no visible geoms to draw")

        self._worlds, self._cameras_per_world = int(world_count), cameras_per_world
        self.width, self.height, self.fov = width, height, fov
        self._geoms = np.asarray(geoms, np.int32)
        reach = np.array(radii) * np.abs(np.asarray(scales)).max(axis=1)
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        solid = reach > 0
        self.center, extent = scene_bounds(probe.geom_xpos[self._geoms][solid], reach[solid])
        self.extent = max(extent, 1e-3)
        self.up = np.array([0.0, 0.0, 1.0])

        stub = SimRenderer.__new__(SimRenderer)
        stub.up, stub.center, stub.extent = self.up, self.center, self.extent
        self.renderer = BatchRenderer(width, height, self._worlds, cameras_per_world, [],
                                      list(lights) if lights is not None else SimRenderer.default_lights(stub),
                                      meshes=builder.meshes, instances=instances, textures=builder.textures, device=device)
        self.renderer.ambient = ambient
        self._look_at = look_at

        self._warp = wp
        self._scales = np.asarray(scales, np.float32).reshape(-1, 3)
        self._wp_geoms = self._wp_scales = self._matrices = None

    @property
    def view_count(self) -> int:
        return self._worlds * self._cameras_per_world

    def default_cameras(self) -> np.ndarray:
        """One camera per view, framing the scene from the side, slightly above: ``(views, 3, 4)``."""
        eye = self.center - np.array([0.0, 1.0, 0.0]) * 2.6 * self.extent + self.up * 0.7 * self.extent
        return np.tile(self._look_at(eye, self.center, self.up), (self.view_count, 1, 1))

    @classmethod
    def _compose(cls):
        """The Warp kernel that turns geom poses into instance matrices, compiled the first time it is wanted."""
        if cls._kernel is None:
            import warp as wp

            @wp.kernel
            def compose(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), geoms: wp.array(dtype=wp.int32),
                        scales: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
                w, i = wp.tid()
                g = geoms[i]
                p, r, s = xpos[w, g], xmat[w, g], scales[i]
                base = (w * geoms.shape[0] + i) * 12
                for row in range(3):
                    out[base + 4 * row + 0] = r[row, 0] * s[0]
                    out[base + 4 * row + 1] = r[row, 1] * s[1]
                    out[base + 4 * row + 2] = r[row, 2] * s[2]
                    out[base + 4 * row + 3] = p[row]

            cls._kernel = compose
        return cls._kernel

    def update(self, data) -> None:
        """Places every geom of every world from ``data.geom_xpos`` and ``data.geom_xmat`` (call ``mjw.forward`` or ``step`` first)."""
        wp = self._warp
        device = data.geom_xpos.device
        if self._matrices is None or self._matrices.device != device:
            self._wp_geoms = wp.array(self._geoms, dtype=wp.int32, device=device)
            self._wp_scales = wp.array(self._scales, dtype=wp.vec3, device=device)
            self._matrices = wp.empty(self._worlds * len(self._geoms) * 12, dtype=wp.float32, device=device)
        wp.launch(self._compose(), dim=(self._worlds, len(self._geoms)),
                  inputs=[data.geom_xpos, data.geom_xmat, self._wp_geoms, self._wp_scales, self._matrices], device=device)
        # Handed over as the Warp array it is: read in place on a GPU. A renderer on the CPU gets the copy it needs.
        self.renderer.set_instance_transforms(self._matrices.numpy() if self.renderer.device == "cpu" else self._matrices)

    def render(self, data, cameras=None, *, fov: float | None = None, rgb: bool = True, depth: bool = False, normals: bool = False,
               object_ids: bool = False, zero_copy: bool = False, frame: int = 0, samples_per_pixel: int = 1) -> dict:
        """Draws every world of ``data``: a dict of ``rgb``, ``depth``, ``normals`` and ``object_ids`` batches.

        See :meth:`neraulgraphics.newton.NewtonBatchRenderer.render`; the arguments are the same.
        """
        self.update(data)
        return self.renderer.render(self.default_cameras() if cameras is None else cameras, self.fov if fov is None else fov,
                                    samples_per_pixel, rgb=rgb, depth=depth, normals=normals, object_ids=object_ids,
                                    frame=frame, zero_copy=zero_copy)

    @property
    def object_id_of_geom(self) -> dict[int, int]:
        """Which object ID each drawn MuJoCo geom index has in an ``object_ids`` image or a ray query."""
        from ._core import OBJECT_INSTANCE
        return {int(geom): OBJECT_INSTANCE + i for i, geom in enumerate(self._geoms)}

    def cast_rays(self, origins, directions, max_distance: float = float("inf")) -> dict:
        """Ray queries in every world: ``(world_count, rays_per_world, 3)`` arrays. See :meth:`neraulgraphics.BatchRenderer.cast_rays`."""
        return self.renderer.cast_rays(origins, directions, max_distance)
