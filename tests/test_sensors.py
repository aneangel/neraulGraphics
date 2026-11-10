"""Tests for the sensor outputs: depth, normals and object-ID images, ray queries, and handing them to
other GPU libraries without a copy. Runs on whichever device is available; the GPU hand-off tests need PyTorch."""

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 160, 90
MAT = ng.Material(diffuse_color=(0.7, 0.4, 0.3))


def scene(device="auto"):
    """A sphere 10 away (its front 8 away), a floor, and a box instance off to one side."""
    return ng.Renderer(W, H, [ng.Sphere((0, 0, -10), 2, MAT)], [ng.Light((0, 5, 2), 1.4)],
                       ng.Plane(y=-2, half_extent_x=40, z_min=-80, z_max=-1),
                       meshes=[ng.make_box((1, 1, 1))],
                       instances=[ng.Instance(0, ng.make_transform((5, 0, -9)), MAT)], device=device)


def camera():
    return ng.make_camera((0, 0, 0), (0, 0, -10), width=W, height=H)


def test_object_id_numbers_are_the_documented_ones():
    assert ng.OBJECT_NONE == 0 and ng.OBJECT_SPHERE == 1
    assert ng.OBJECT_FLOOR == ng.OBJECT_SPHERE + ng.MAX_SPHERES and ng.OBJECT_INSTANCE == ng.OBJECT_FLOOR + 1


def test_frame_returns_only_what_was_asked_for_with_the_right_shapes():
    r = scene()
    assert set(r.render_frame(camera())) == {"rgb"}
    out = r.render_frame(camera(), rgb=False, depth=True, normals=True, object_ids=True)
    assert set(out) == {"depth", "normals", "object_ids"}
    assert out["depth"].shape == (H, W) and out["depth"].dtype == np.float32
    assert out["normals"].shape == (H, W, 3) and out["normals"].dtype == np.float32
    assert out["object_ids"].shape == (H, W) and out["object_ids"].dtype == np.int32
    assert r.render_frame(camera(), depth=True)["rgb"].shape == (H, W, 3)


def test_asking_for_nothing_is_an_error():
    with pytest.raises(ValueError, match="at least one output"):
        scene().render_frame(camera(), rgb=False)


def test_depth_normals_and_ids_have_the_values_the_geometry_gives():
    out = scene().render_frame(camera(), rgb=False, depth=True, normals=True, object_ids=True)
    cx, cy = W // 2, H // 2
    assert out["depth"][cy, cx] == pytest.approx(8.0, abs=0.05)             # the sphere's front
    assert out["object_ids"][cy, cx] == ng.OBJECT_SPHERE
    assert np.allclose(out["normals"][cy, cx], [0, 0, 1], atol=0.1)
    assert out["object_ids"][H - 3, cx] == ng.OBJECT_FLOOR                  # near the bottom: the floor
    assert np.allclose(out["normals"][H - 3, cx], [0, 1, 0], atol=1e-5)
    assert ng.OBJECT_INSTANCE in out["object_ids"]                          # the box to the side
    sky = out["object_ids"] == ng.OBJECT_NONE
    assert sky.any() and np.isinf(out["depth"][sky]).all() and (out["normals"][sky] == 0).all()


def test_rgb_output_is_the_picture_render_draws():
    r = scene()
    assert np.array_equal(r.render_frame(camera(), 0)["rgb"], r.render(camera(), 0))


def test_a_ray_query_agrees_with_the_depth_image():
    r = scene()
    depth = r.render_frame(camera(), rgb=False, depth=True)["depth"]
    hits = r.cast_rays(np.array([[0, 0, 0]], dtype=np.float32), np.array([[0, 0, -1]], dtype=np.float32))
    assert hits["distance"][0] == pytest.approx(8.0, abs=1e-3) and hits["object_id"][0] == ng.OBJECT_SPHERE
    assert depth[H // 2, W // 2] == pytest.approx(hits["distance"][0], abs=0.05)


def test_ray_queries_take_lists_misses_and_a_distance_limit():
    r = scene()
    o = np.zeros((3, 3), dtype=np.float32)
    d = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=np.float32)  # the sphere, the sky, no direction at all
    hits = r.cast_rays(o, d)
    assert list(hits["object_id"]) == [ng.OBJECT_SPHERE, ng.OBJECT_NONE, ng.OBJECT_NONE]
    assert np.isinf(hits["distance"][1:]).all() and hits["normal"].shape == (3, 3)
    assert r.cast_rays(o[:1], d[:1], max_distance=7.0)["object_id"][0] == ng.OBJECT_NONE
    assert len(r.cast_rays(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))["distance"]) == 0


def test_bad_rays_are_rejected():
    r = scene()
    z = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same number"):
        r.cast_rays(z, z[:1])
    with pytest.raises(ValueError, match="max_distance"):
        r.cast_rays(z, z, max_distance=0.0)
    bad = z.copy()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        r.cast_rays(bad, z)


def test_cpu_views_share_memory_and_are_overwritten_by_the_next_frame():
    r = scene("cpu")
    first = r.render_frame(camera(), rgb=False, depth=True, zero_copy=True)["depth"]
    kept = first.copy()
    other = ng.make_camera((2, 0, 0), (0, 0, -10), width=W, height=H)
    second = r.render_frame(other, rgb=False, depth=True, zero_copy=True)["depth"]
    assert np.shares_memory(first, second)          # the same buffer, not a new array
    assert not np.array_equal(first, kept)          # ... which the second render overwrote
    assert np.array_equal(r.render_frame(camera(), rgb=False, depth=True)["depth"], kept)  # copies are safe to keep


def test_copies_stay_valid_after_the_next_frame():
    r = scene()
    a = r.render_frame(camera(), rgb=False, depth=True)["depth"]
    kept = a.copy()
    r.render_frame(ng.make_camera((2, 0, 0), (0, 0, -10), width=W, height=H), rgb=False, depth=True)
    assert np.array_equal(a, kept)


def test_cpu_and_gpu_agree():
    try:
        gpu, cpu = scene("cuda"), scene("cpu")
    except ng.CudaError:
        pytest.skip("no usable CUDA GPU")
    a = gpu.render_frame(camera(), rgb=False, depth=True, normals=True, object_ids=True)
    b = cpu.render_frame(camera(), rgb=False, depth=True, normals=True, object_ids=True)
    assert (a["object_ids"] != b["object_ids"]).mean() < 0.001
    both = np.isfinite(a["depth"]) & np.isfinite(b["depth"])
    assert np.abs(a["depth"][both] - b["depth"][both]).max() < 0.01


# --------------------------------------------------------------- the GPU hand-off --

@pytest.fixture(scope="module")
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch has no CUDA device")
    try:
        scene("cuda")
    except ng.CudaError:
        pytest.skip("no usable CUDA GPU for neraulgraphics")
    return torch


def test_gpu_images_arrive_as_cuda_tensors_without_a_copy(torch_cuda):
    torch = torch_cuda
    r = scene("cuda")
    out = r.render_frame(camera(), depth=True, normals=True, object_ids=True, zero_copy=True)
    tensors = {k: torch.from_dlpack(v) for k, v in out.items()}
    assert all(t.is_cuda for t in tensors.values())
    assert tensors["depth"].shape == (H, W) and tensors["depth"].dtype == torch.float32
    assert tensors["rgb"].dtype == torch.uint8 and tensors["object_ids"].dtype == torch.int32
    copies = r.render_frame(camera(), depth=True, normals=True, object_ids=True)
    for key, tensor in tensors.items():
        assert np.array_equal(tensor.cpu().numpy(), copies[key]), key

    # They are views: the next frame is written into the same memory.
    pointer = tensors["depth"].data_ptr()
    r.render_frame(ng.make_camera((2, 0, 0), (0, 0, -10), width=W, height=H), rgb=False, depth=True, zero_copy=True)
    assert tensors["depth"].data_ptr() == pointer
    assert not np.array_equal(tensors["depth"].cpu().numpy(), copies["depth"])


def test_the_tensors_keep_the_renderer_alive(torch_cuda):
    torch = torch_cuda
    depth = torch.from_dlpack(scene("cuda").render_frame(camera(), rgb=False, depth=True, zero_copy=True)["depth"])
    import gc
    gc.collect()  # the renderer is no longer referenced by anything but the tensor's owner
    assert float(depth[H // 2, W // 2]) == pytest.approx(8.0, abs=0.05)


def test_ray_queries_on_torch_tensors_stay_on_the_gpu(torch_cuda):
    torch = torch_cuda
    r = scene("cuda")
    rng = np.random.default_rng(0)
    o = rng.uniform(-2, 2, (5000, 3)).astype(np.float32)
    d = rng.normal(size=(5000, 3)).astype(np.float32)
    d[:, 2] = -np.abs(d[:, 2]) - 0.1
    via_numpy = r.cast_rays(o, d, 100.0)
    via_torch = r.cast_rays(torch.from_numpy(o).cuda(), torch.from_numpy(d).cuda(), 100.0)
    dist = torch.from_dlpack(via_torch["distance"])
    assert dist.is_cuda and dist.shape == (5000,)
    assert np.array_equal(dist.cpu().numpy(), via_numpy["distance"])
    assert np.array_equal(torch.from_dlpack(via_torch["object_id"]).cpu().numpy(), via_numpy["object_id"])
    assert (via_numpy["object_id"] != 0).sum() > 100  # the comparison was between real hits


def test_cuda_arrays_are_refused_by_a_cpu_renderer(torch_cuda):
    torch = torch_cuda
    r = scene("cpu")
    t = torch.zeros(4, 3, device="cuda")
    with pytest.raises((ValueError, TypeError)):
        r.cast_rays(t, t)
