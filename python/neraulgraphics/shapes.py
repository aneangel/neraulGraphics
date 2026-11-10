"""Shape generators and pose math for feeding simulators to the renderer.

Every generator returns a :class:`neraulgraphics.Mesh` along the **Z axis**, the
convention MuJoCo and Newton share. They are unit-friendly: a cylinder or cone made
with radius 1 and half-height 1 becomes any size by scaling its instance, so one mesh
serves every body of that kind. A capsule's end caps are round, so it cannot be
rescaled and needs its own mesh per size.
"""

from __future__ import annotations

import numpy as np

from ._core import Mesh, Texture

__all__ = ["capsule", "checker_texture", "compose_poses", "cone", "cylinder", "plane", "pose_matrices", "quat_to_matrix",
           "uv_sphere"]


def _revolve(profile: np.ndarray, segments: int) -> Mesh:
    """Spin a (radius, z) profile around the Z axis into a closed surface.

    Rows with radius 0 are single points (a pole or a cap centre); other rows are
    rings of ``segments`` vertices. Consecutive rows are joined with quads, or with
    a fan where one of them is a point.
    """
    if segments < 3:
        raise ValueError(f"segments must be at least 3, got {segments}")
    angle = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angle), np.sin(angle)

    vertices: list[np.ndarray] = []
    start: list[int] = []  # index of each row's first vertex
    for radius, z in profile:
        start.append(sum(len(v) for v in vertices))
        if radius == 0.0:
            vertices.append(np.array([[0.0, 0.0, z]]))
        else:
            vertices.append(np.stack([radius * cos, radius * sin, np.full(segments, z)], axis=1))

    triangles: list[tuple[int, int, int]] = []
    for row in range(len(profile) - 1):
        a_point, b_point = profile[row][0] == 0.0, profile[row + 1][0] == 0.0
        a, b = start[row], start[row + 1]
        for i in range(segments):
            j = (i + 1) % segments
            if a_point and b_point:
                continue
            if a_point:
                triangles.append((a, b + i, b + j))
            elif b_point:
                triangles.append((a + i, a + j, b))
            else:
                triangles.append((a + i, a + j, b + j))
                triangles.append((a + i, b + j, b + i))

    return Mesh(np.concatenate(vertices).astype(np.float32), np.array(triangles, dtype=np.uint32))


def uv_sphere(radius: float = 1.0, segments: int = 32, rings: int = 16) -> Mesh:
    """A latitude/longitude sphere centred on the origin, poles on the Z axis."""
    lat = np.linspace(np.pi / 2, -np.pi / 2, rings + 1)
    profile = np.stack([radius * np.cos(lat), radius * np.sin(lat)], axis=1)
    profile[0, 0] = profile[-1, 0] = 0.0  # cos(pi/2) is not exactly 0
    return _revolve(profile, segments)


def capsule(radius: float, half_height: float, segments: int = 32, rings: int = 8) -> Mesh:
    """A cylinder of half-height ``half_height`` capped with hemispheres of ``radius``.

    The total length along Z is ``2 * (half_height + radius)``, as in MuJoCo and Newton.
    """
    top = np.linspace(np.pi / 2, 0.0, rings + 1)  # latitude on the upper cap
    bottom = np.linspace(0.0, -np.pi / 2, rings + 1)
    profile = np.concatenate([
        np.stack([radius * np.cos(top), half_height + radius * np.sin(top)], axis=1),
        np.stack([radius * np.cos(bottom), -half_height + radius * np.sin(bottom)], axis=1),
    ])
    profile[0, 0] = profile[-1, 0] = 0.0
    return _revolve(profile, segments)


def cylinder(radius: float = 1.0, half_height: float = 1.0, segments: int = 32) -> Mesh:
    """A capped cylinder along Z. Made at 1 x 1, scale an instance to any size."""
    return _revolve(np.array([[0, half_height], [radius, half_height], [radius, -half_height], [0, -half_height]],
                             dtype=np.float64), segments)


def cone(radius: float = 1.0, half_height: float = 1.0, segments: int = 32) -> Mesh:
    """A cone along Z with its apex at +Z and a flat base at -Z."""
    return _revolve(np.array([[0, half_height], [radius, -half_height], [0, -half_height]], dtype=np.float64),
                    segments)


def plane(width: float, length: float, tile: float | None = None) -> Mesh:
    """A flat rectangle in the XY plane, ``width`` along X and ``length`` along Y (full sizes).

    With ``tile`` it has texture coordinates that repeat a texture once every ``tile`` units.
    """
    x, y = width / 2.0, length / 2.0
    uvs = None
    if tile is not None:
        u, v = width / tile, length / tile
        uvs = np.array([[0, v], [u, v], [u, 0], [0, 0]], dtype=np.float32)
    return Mesh(np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]], dtype=np.float32),
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32), uvs=uvs)


def checker_texture(color_a=(0.30, 0.33, 0.38), color_b=(0.40, 0.43, 0.48), size: int = 64) -> Texture:
    """A 2 x 2 checkerboard (one repeat of the pattern) of two colours given as 0..1 RGB.

    ``size`` is the image's side in pixels, a multiple of 2. Colours are what you see on screen
    (sRGB), not linear light.
    """
    if size < 2 or size % 2:
        raise ValueError(f"size must be an even number of at least 2, got {size}")
    half = size // 2
    a = np.round(np.clip(color_a, 0, 1) * 255).astype(np.uint8)
    b = np.round(np.clip(color_b, 0, 1) * 255).astype(np.uint8)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:half, :half] = image[half:, half:] = a
    image[:half, half:] = image[half:, :half] = b
    return Texture(image)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternions ``(N, 4)`` as (x, y, z, w) to rotation matrices ``(N, 3, 3)``. Normalizes first."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    m = np.empty((len(q), 3, 3))
    m[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1)
    m[:, 1] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1)
    m[:, 2] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)
    return m


def compose_poses(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pose ``(p1, q1)`` applied after ``(p2, q2)``: ``p = p1 + R1 p2`` and ``q = q1 * q2``.

    All arguments are ``(N, 3)`` positions and ``(N, 4)`` (x, y, z, w) quaternions.
    """
    p1, q1, p2, q2 = (np.asarray(a, dtype=np.float64) for a in (p1, q1, p2, q2))
    position = p1 + np.einsum("nij,nj->ni", quat_to_matrix(q1), p2)
    x1, y1, z1, w1 = q1.T
    x2, y2, z2, w2 = q2.T
    quat = np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=1)
    return position, quat


def pose_matrices(position: np.ndarray, quat: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Poses as the ``(N, 3, 4)`` float32 matrices ``Renderer.set_instance_transforms`` takes.

    ``scale`` is ``(N, 3)`` and is applied first, along the mesh's own axes.
    """
    return matrices_from_rotations(quat_to_matrix(quat), position, scale)


def matrices_from_rotations(rotation: np.ndarray, position: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Like :func:`pose_matrices`, from rotation matrices ``(N, 3, 3)`` (as MuJoCo provides)."""
    out = np.empty((len(rotation), 3, 4), dtype=np.float32)
    out[:, :, :3] = rotation if scale is None else rotation * np.asarray(scale)[:, None, :]
    out[:, :, 3] = position
    return out
