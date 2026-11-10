"""Tests for the shape generators and pose math. No GPU needed."""

import numpy as np
import pytest

import neraulgraphics as ng
from neraulgraphics import shapes


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return rng.normal(size=(n, 3)), q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quat_to_matrix_is_a_rotation():
    _, q = random_poses(50)
    r = shapes.quat_to_matrix(q)
    assert np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(r), 1.0)


def test_quat_to_matrix_known_rotation():
    # 90 degrees about z, as (x, y, z, w): the x axis goes to the y axis. XYZW, not WXYZ.
    q = np.array([[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    assert np.allclose(shapes.quat_to_matrix(q)[0] @ [1, 0, 0], [0, 1, 0])


def test_quat_to_matrix_normalizes():
    q = np.array([[0, 0, 2 * np.sin(np.pi / 4), 2 * np.cos(np.pi / 4)]])  # twice unit length
    assert np.allclose(shapes.quat_to_matrix(q)[0] @ [1, 0, 0], [0, 1, 0])


def test_compose_poses_matches_matrix_multiplication():
    p1, q1 = random_poses(40, 1)
    p2, q2 = random_poses(40, 2)
    p, q = shapes.compose_poses(p1, q1, p2, q2)
    r1, r2 = shapes.quat_to_matrix(q1), shapes.quat_to_matrix(q2)
    assert np.allclose(shapes.quat_to_matrix(q), r1 @ r2, atol=1e-12)
    assert np.allclose(p, p1 + np.einsum("nij,nj->ni", r1, p2), atol=1e-12)


def test_pose_matrices_scale_applies_along_the_meshs_own_axes():
    p, q = random_poses(5, 3)
    scale = np.array([[2.0, 3.0, 4.0]] * 5)
    m = shapes.pose_matrices(p, q, scale)
    assert m.shape == (5, 3, 4) and m.dtype == np.float32
    unscaled = shapes.pose_matrices(p, q)
    assert np.allclose(m[:, :, :3], unscaled[:, :, :3] * scale[:, None, :], atol=1e-6)
    assert np.allclose(m[:, :, 3], p, atol=1e-6)


@pytest.mark.parametrize("make, triangles", [
    (lambda: shapes.cylinder(1, 1, 16), 4 * 16),   # two caps of 16, a side of 32
    (lambda: shapes.cone(1, 1, 16), 2 * 16),
    (lambda: shapes.plane(2, 3), 2),
])
def test_triangle_counts(make, triangles):
    assert make().triangle_count == triangles


def test_capsule_and_sphere_are_closed_surfaces():
    # Euler's formula for a closed sphere-like surface: V - E + F = 2, with E = 3F / 2.
    for mesh in (shapes.capsule(0.3, 0.7, 12, 4), shapes.uv_sphere(1.0, 12, 6)):
        v, f = mesh.vertex_count, mesh.triangle_count
        assert v - 3 * f // 2 + f == 2


def test_generators_reject_too_few_segments():
    with pytest.raises(ValueError, match="segments"):
        shapes.cylinder(1, 1, 2)
