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
