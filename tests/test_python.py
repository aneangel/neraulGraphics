"""Tests for the neraulgraphics Python package.

Rendering tests need an NVIDIA GPU and skip without one; the import and version
tests run anywhere the package installs.
"""

import re
import threading

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 160, 90


def scene():
    ivory = ng.Material(diffuse_color=(0.4, 0.4, 0.3), diffuse_albedo=0.6, specular_albedo=0.3,
                        reflectivity=0.1, specular_exponent=50)
    mirror = ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0.0, specular_albedo=10.0,
                         reflectivity=0.8, specular_exponent=1425)
    spheres = [ng.Sphere((-3, 0, -16), 2, ivory), ng.Sphere((-1, -1.5, -12), 2, mirror)]
    lights = [ng.Light((-20, 20, 20), intensity=1.5), ng.Light((30, 50, 30), 1.7, radius=2.0)]
    floor = ng.Plane(y=-4, half_extent_x=10, z_min=-30, z_max=-10)
    return spheres, lights, floor


def camera():
    return ng.make_camera((0, 0, 0), (0, 0, -16), width=W, height=H)


@pytest.fixture(scope="module")
def gpu():
    try:
        ng.Renderer(8, 8, [], [])
    except ng.CudaError as e:
        pytest.skip(f"no usable CUDA GPU: {e}")


def accumulate(renderer, frames):
    image = None
    for frame in range(frames):
        image = renderer.render(camera(), frame)
    return image


# ------------------------------------------------------------------ no GPU --

def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ng.__version__)


def test_vectors_accept_tuples_lists_and_arrays():
    for v in [(1, 2, 3), [1, 2, 3], np.array([1, 2, 3], dtype=np.float32)]:
        assert ng.Sphere(v, 1.0).center == (1.0, 2.0, 3.0)


def test_vectors_reject_wrong_shapes():
    for bad in [(1, 2), (1, 2, 3, 4), "abc", (1, "x", 3), 5]:
        with pytest.raises(TypeError):
            ng.Sphere(bad, 1.0)


def test_fields_round_trip():
    s = ng.Sphere((0, 0, -5), 2)
    s.center = (1, 2, 3)
    m = s.material
    m.reflectivity = 0.5
    s.material = m
    assert s.center == (1.0, 2.0, 3.0)
    assert s.material.reflectivity == 0.5


# --------------------------------------------------------------------- GPU --

def test_render_returns_rgb_image(gpu):
    r = ng.Renderer(W, H, *scene())
    image = r.render(camera())
    assert image.shape == (H, W, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert image.std() > 0  # not a blank frame
    assert r.width == W and r.height == H
    assert r.last_kernel_ms > 0


def test_images_are_copies(gpu):
    r = ng.Renderer(W, H, *scene())
    first = r.render(camera(), 0)
    kept = first.copy()
    r.render(ng.make_camera((5, 5, 0), (0, 0, -16), width=W, height=H), 0)
    assert np.array_equal(first, kept)


def test_deterministic(gpu):
    a = accumulate(ng.Renderer(W, H, *scene()), 4)
    b = accumulate(ng.Renderer(W, H, *scene()), 4)
    assert np.array_equal(a, b)


def test_independent_renderers(gpu):
    spheres, lights, floor = scene()
    other = [ng.Sphere((0, 0, -10), 3)]
    ref_a = accumulate(ng.Renderer(W, H, spheres, lights, floor), 4)
    ref_b = accumulate(ng.Renderer(W, H, other, lights, floor), 4)
    a = ng.Renderer(W, H, spheres, lights, floor)
    b = ng.Renderer(W, H, other, lights, floor)
    for frame in range(4):
        out_a = a.render(camera(), frame)
        out_b = b.render(camera(), frame)
    assert not np.array_equal(ref_a, ref_b)
    assert np.array_equal(out_a, ref_a) and np.array_equal(out_b, ref_b)


def test_default_floor_is_no_floor(gpu):
    spheres, lights, floor = scene()
    with_floor = accumulate(ng.Renderer(W, H, spheres, lights, floor), 2)
    without = accumulate(ng.Renderer(W, H, spheres, lights), 2)
    assert not np.array_equal(with_floor, without)


def test_threads_share_a_renderer(gpu):
    r = ng.Renderer(W, H, *scene())
    expected = r.render(camera(), 0)
    results, errors = [], []

    def worker():
        try:
            for _ in range(10):
                results.append(r.render(camera(), 0))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 40
    assert all(np.array_equal(img, expected) for img in results)


@pytest.mark.parametrize("bad", [
    dict(width=0),
    dict(height=-1),
    dict(spheres=[ng.Sphere((0, 0, -5), 1)] * (ng.MAX_SPHERES + 1)),
    dict(lights=[ng.Light((0, 5, 0))] * (ng.MAX_LIGHTS + 1)),
    dict(spheres=[ng.Sphere((0, 0, -5), 0)]),
    dict(spheres=[ng.Sphere((0, float("nan"), -5), 1)]),
    dict(lights=[ng.Light((0, 5, 0), radius=-1)]),
    dict(floor=ng.Plane(y=float("inf"))),
])
def test_constructor_rejects_bad_scenes(gpu, bad):
    spheres, lights, floor = scene()
    args = dict(width=W, height=H, spheres=spheres, lights=lights, floor=floor) | bad
    with pytest.raises(ValueError):
        ng.Renderer(**args)


def test_render_rejects_bad_arguments_and_recovers(gpu):
    r = ng.Renderer(W, H, *scene())
    for frame in range(3):
        r.render(camera(), frame)
    with pytest.raises(ValueError, match="samples_per_frame"):
        r.render(camera(), 3, samples_per_frame=0)
    with pytest.raises(ValueError, match="frame"):
        r.render(camera(), -1)
    straight_up = ng.make_camera((0, 0, 0), (0, 5, 0), width=W, height=H)
    with pytest.raises(ValueError, match="camera"):
        r.render(straight_up, 3)
    # The rejected calls changed nothing: frame 3 continues the same average.
    assert np.array_equal(r.render(camera(), 3), accumulate(ng.Renderer(W, H, *scene()), 4))


# ------------------------------------------------------------------ meshes --

def test_make_transform_matches_the_quaternion_convention():
    # 90 degrees about z, as (x, y, z, w): x axis goes to y axis. XYZW, not WXYZ.
    q = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    m = ng.make_transform((1, 2, 3), q, scale=2.0).matrix
    assert m.shape == (3, 4) and m.dtype == np.float32
    assert np.allclose(m @ [1, 0, 0, 1], [1, 2 + 2, 3], atol=1e-6)
    assert np.allclose(m @ [0, 1, 0, 1], [1 - 2, 2, 3], atol=1e-6)


def test_transforms_come_from_3x4_or_4x4_matrices():
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (4, 5, 6)
    assert np.array_equal(ng.Transform(m).matrix, m[:3])
    assert np.array_equal(ng.Transform(m[:3].copy()).matrix, m[:3])
    with pytest.raises(ValueError):
        ng.Transform(np.zeros((2, 4), dtype=np.float32))


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError, match="quaternion"):
        ng.make_transform((0, 0, 0), (0, 0, 0, 0))


def test_mesh_from_numpy_accepts_common_dtypes():
    box = ng.make_box((1, 1, 1))
    assert box.triangle_count == 12 and box.vertex_count == 8
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)  # float64 is converted
    for dtype in (np.int32, np.int64, np.uint32):
        mesh = ng.Mesh(vertices, np.array([[0, 1, 2]], dtype=dtype))
        assert mesh.triangle_count == 1


def test_mesh_rejects_bad_shapes_and_indices(gpu):
    with pytest.raises(TypeError):
        ng.Mesh(np.zeros((4, 2), dtype=np.float32), np.zeros((1, 3), dtype=np.uint32))
    # A well-shaped mesh that points past its vertices is caught when a Renderer is made.
    bad = ng.Mesh(np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 7]], dtype=np.uint32))
    with pytest.raises(ValueError, match="index"):
        ng.Renderer(W, H, [], [], meshes=[bad], instances=[ng.Instance(0)])


def mesh_scene(x=0.0):
    red = ng.Material(diffuse_color=(0.6, 0.15, 0.15))
    spheres, lights, floor = scene()
    return dict(spheres=spheres, lights=lights, floor=floor, meshes=[ng.make_icosphere(1.0, 3)],
                instances=[ng.Instance(0, ng.make_transform((x, 0, -12), scale=2.0), red)])


def test_instances_render_and_match_an_analytic_sphere(gpu):
    spheres, lights, floor = scene()
    red = ng.Material(diffuse_color=(0.6, 0.15, 0.15))
    analytic = ng.Renderer(W, H, spheres + [ng.Sphere((3, 0, -12), 2, red)], lights, floor).render(camera())
    meshed = ng.Renderer(W, H, **mesh_scene(3.0)).render(camera())
    nothing = ng.Renderer(W, H, spheres, lights, floor).render(camera())
    assert np.abs(meshed.astype(int) - nothing.astype(int)).mean() > 0.3  # the mesh is there
    assert np.abs(meshed.astype(int) - analytic.astype(int)).mean() < 1.0  # and it is the right shape


def test_set_instance_transforms_from_array_and_list(gpu):
    r = ng.Renderer(W, H, **mesh_scene(-3.0))
    assert r.instance_count == 1
    before = r.render(camera())

    moved = np.array([ng.make_transform((3, 0, -12), scale=2.0).matrix])  # (1, 3, 4)
    r.set_instance_transforms(moved)
    after = r.render(camera())
    assert not np.array_equal(before, after)
    # Moving is the same as building there: this is the whole point of the two-level structure.
    assert np.array_equal(after, ng.Renderer(W, H, **mesh_scene(3.0)).render(camera()))

    r.set_instance_transforms([ng.make_transform((-3, 0, -12), scale=2.0)])  # a list of Transform
    assert np.array_equal(r.render(camera()), before)


def test_set_instance_transforms_rejects_bad_input_and_changes_nothing(gpu):
    r = ng.Renderer(W, H, **mesh_scene())
    before = r.render(camera())
    with pytest.raises(ValueError, match="one per instance"):
        r.set_instance_transforms(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="singular"):
        r.set_instance_transforms(np.zeros((1, 3, 4), dtype=np.float32))
    nan = ng.make_transform((0, 0, 0)).matrix.copy()
    nan[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        r.set_instance_transforms(nan[None])
    with pytest.raises(TypeError):
        r.set_instance_transforms(np.zeros((1, 4, 4), dtype=np.float32))
    assert np.array_equal(r.render(camera()), before)


def test_many_instances_in_one_call(gpu):
    rng = np.random.default_rng(0)
    n = 2000
    transforms = [ng.make_transform(rng.uniform((-9, -3, -28), (9, 3, -12)), rng.normal(size=4) + (0, 0, 0, 3),
                                    scale=rng.uniform(0.3, 0.8)) for _ in range(n)]
    box = ng.make_box((0.5, 0.5, 0.5))
    r = ng.Renderer(W, H, [], scene()[1], scene()[2], meshes=[box],
                    instances=[ng.Instance(0, t) for t in transforms])
    assert r.instance_count == n
    assert r.render(camera()).std() > 0
    r.set_instance_transforms(np.array([t.matrix for t in transforms]))  # all 2000 in one call
