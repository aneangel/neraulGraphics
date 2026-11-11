"""What the simulator adapters (:mod:`neraulgraphics.newton`, :mod:`neraulgraphics.mujoco`) share.

A simulator describes shapes once (kind, size, colour) and then reports new poses every
step. So an adapter builds the renderer once, from the shapes, and afterwards only turns
poses into ``(N, 3, 4)`` matrices for ``set_instance_transforms``: no geometry is rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

import numpy as np

from ._core import Camera, Environment, Instance, Light, Material, Mesh, PathTracing, RenderMode, Renderer, Texture, Transform
from ._core import make_box, make_camera
from ._core import make_icosphere

__all__ = ["SceneBuilder", "SimRenderer", "scene_bounds"]

# Plain matte surfaces: simulators describe colour, not shininess.
_SPECULAR = 0.2
_EXPONENT = 40.0


def _material(color: Sequence[float]) -> Material:
    rgb = tuple(float(np.clip(c, 0.0, 1.0)) for c in list(color)[:3])
    return Material(diffuse_color=rgb, diffuse_albedo=0.9, specular_albedo=_SPECULAR, reflectivity=0.0,
                    specular_exponent=_EXPONENT)


class SceneBuilder:
    """Collects the meshes and instances of a scene, sharing one mesh between equal shapes."""

    def __init__(self) -> None:
        self.meshes: list[Mesh] = []
        self.radius: list[float] = []  # per mesh: how far any vertex is from the origin
        self.instances: list[Instance] = []
        self.textures: list[Texture] = []
        self._by_key: dict[Hashable, int] = {}

    def mesh(self, key: Hashable, make: Callable[[], Mesh], radius: float) -> int:
        """The index of the mesh for ``key``, calling ``make`` only the first time.

        ``radius`` bounds the mesh around its origin; it is how the adapters size up a scene.
        """
        if key not in self._by_key:
            self._by_key[key] = len(self.meshes)
            self.meshes.append(make())
            self.radius.append(float(radius))
        return self._by_key[key]

    def texture(self, key: Hashable, make: Callable[[], Texture]) -> int:
        """The index of the texture for ``key``, calling ``make`` only the first time."""
        if ("texture", key) not in self._by_key:
            self._by_key[("texture", key)] = len(self.textures)
            self.textures.append(make())
        return self._by_key[("texture", key)]

    def unit_sphere(self) -> int:
        return self.mesh("unit_sphere", lambda: make_icosphere(1.0, 3), 1.0)

    def unit_box(self) -> int:
        return self.mesh("unit_box", lambda: make_box((1.0, 1.0, 1.0)), float(np.sqrt(3.0)))

    def add(self, mesh: int, color: Sequence[float], texture: int = -1) -> int:
        """Adds an instance at the origin. The first ``set_instance_transforms`` places it."""
        self.instances.append(Instance(mesh, Transform(np.eye(4, dtype=np.float32)), _material(color), texture))
        return len(self.instances) - 1


def scene_bounds(positions: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and half-size of spheres of radius ``reach`` placed at ``positions`` (empty: a unit scene)."""
    if len(positions) == 0:
        return np.zeros(3), 1.0
    lo, hi = (positions - reach[:, None]).min(axis=0), (positions + reach[:, None]).max(axis=0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0


def _horizontal(up: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``up``, the direction the default camera looks along."""
    probe = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    h = np.cross(up, np.cross(probe, up))
    return h / np.linalg.norm(h)


class SimRenderer:
    """Common to every adapter: owns the :class:`Renderer` and the camera conveniences."""

    def __init__(self, width: int, height: int, scene, *, up: Sequence[float],
                 center: Sequence[float], extent: float, lights: Sequence[Light] | None, fov: float,
                 ambient: float = 0.2, device: str = "auto") -> None:
        # scene: anything with .meshes, .instances and .textures (a SceneBuilder, or a LoadedScene).
        # center and extent: the middle and half-size of what is worth looking at.
        self.width, self.height, self.fov = width, height, fov
        self.up = np.asarray(up, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.extent = float(max(extent, 1e-3))
        self.renderer = Renderer(width, height, [], list(lights) if lights is not None else self.default_lights(),
                                 meshes=scene.meshes, instances=scene.instances, textures=scene.textures,
                                 device=device)
        # Surfaces turned from every light would be black; a little ambient light keeps them readable.
        self.renderer.ambient = ambient

    @property
    def instance_count(self) -> int:
        return self.renderer.instance_count

    def beauty(self, sky=None, *, bounces: int = 4, **settings) -> None:
        """Draws realistic pictures from now on: path tracing, with soft shadows and sky light.

        ``sky`` is what lights the scene from all sides: None for a plain blue gradient, an
        :class:`Environment`, or a ``(height, width, 3)`` float array of radiance (see
        ``procedural_sky`` and ``load_hdr``). A sky given as an array or left out is turned to match the
        simulator's up axis. The other ``settings`` are those of :class:`PathTracing`. Call ``render`` with
        increasing frame numbers while the camera holds still, and the picture refines itself.
        """
        if sky is None:
            sky = Environment(up=tuple(float(v) for v in self.up))
        elif not isinstance(sky, Environment):
            sky = Environment(up=tuple(float(v) for v in self.up), map=sky)
        self.renderer.set_environment(sky)
        self.renderer.set_render_mode(RenderMode.PathTraced, PathTracing(bounces=bounces, **settings))

    def classic(self) -> None:
        """Goes back to the fast, classic pictures."""
        self.renderer.set_render_mode(RenderMode.Classic)

    def default_lights(self) -> list[Light]:
        """A key light high and to one side, and a dimmer fill from the other, scaled to the scene."""
        side = _horizontal(self.up)
        across = np.cross(self.up, side)
        key = self.center + self.up * 3.0 * self.extent + side * 1.5 * self.extent + across * 1.0 * self.extent
        fill = self.center + self.up * 2.0 * self.extent - side * 2.0 * self.extent - across * 1.5 * self.extent
        return [Light(tuple(key), intensity=1.0, radius=0.0), Light(tuple(fill), intensity=0.4, radius=0.0)]

    def camera(self, position: Sequence[float] | None = None, target: Sequence[float] | None = None,
               fov: float | None = None) -> Camera:
        """A camera looking at ``target`` (default: the scene's centre) from ``position``.

        With no arguments it frames the whole scene from the side, slightly above.
        """
        target = self.center if target is None else np.asarray(target, dtype=np.float64)
        if position is None:
            position = self.center - _horizontal(self.up) * 2.6 * self.extent + self.up * 0.7 * self.extent
        return make_camera(tuple(float(v) for v in position), tuple(float(v) for v in target), width=self.width,
                           height=self.height, world_up=tuple(float(v) for v in self.up),
                           fov=self.fov if fov is None else fov)

    def _render(self, matrices: np.ndarray, camera: Camera | None, frame: int, samples_per_frame: int) -> np.ndarray:
        if len(matrices):
            self.renderer.set_instance_transforms(matrices)
        return self.renderer.render(self.camera() if camera is None else camera, frame, samples_per_frame)

    def _render_frame(self, matrices: np.ndarray, camera: Camera | None, frame: int, samples_per_frame: int,
                      outputs: dict) -> dict:
        if len(matrices):
            self.renderer.set_instance_transforms(matrices)
        return self.renderer.render_frame(self.camera() if camera is None else camera, frame, samples_per_frame,
                                          **outputs)

    def cast_rays(self, origins, directions, max_distance: float = float("inf")) -> dict:
        """Casts rays into the scene as it was last placed (by ``render``, ``render_frame`` or ``update``).

        origins and directions are ``(N, 3)`` float arrays, as NumPy or, on a GPU renderer, as CUDA arrays
        such as PyTorch tensors. Returns ``distance``, ``normal`` and ``object_id``: see
        :meth:`neraulgraphics.Renderer.cast_rays`.
        """
        return self.renderer.cast_rays(origins, directions, max_distance)
