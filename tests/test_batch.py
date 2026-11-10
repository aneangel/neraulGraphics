"""Tests for BatchRenderer from Python: many worlds drawn at once. Runs on whichever device is available;
the GPU hand-off tests need PyTorch with CUDA."""

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 64, 36
FOV = 1.2
MAT = ng.Material(diffuse_color=(0.7, 0.5, 0.3))


def instances():
    green = ng.Material(diffuse_color=(0.2, 0.6, 0.3))
    return [ng.Instance(0, material=MAT), ng.Instance(1, material=green)]


def meshes():
    return [ng.make_box((1, 1, 1)), ng.make_icosphere(1.0, 2)]


def scene_kwargs():
    return dict(spheres=[ng.Sphere((0, -1, -9), 1.0, MAT)], lights=[ng.Light((-8, 8, 8), 1.4)],
                floor=ng.Plane(y=-2, half_extent_x=30, z_min=-60, z_max=5), meshes=meshes(), instances=instances())


def poses(worlds, seed=0):
    """(worlds, 2, 3, 4): two instances per world, a different place in every world."""
    rng = np.random.default_rng(seed)
    out = np.empty((worlds, 2, 3, 4), dtype=np.float32)
    for w in range(worlds):
        for i in range(2):
            out[w, i] = ng.make_transform((-2 + 4 * i + rng.uniform(-.5, .5), rng.uniform(-.5, .5), -5 - 3 * rng.random()),
                                          rng.normal(size=4) + (0, 0, 0, 3), 0.6 + 0.5 * rng.random()).matrix
    return out


def camera_matrix(eye=(0, 0.5, 0), target=(0, 0, -8)):
    cam = ng.make_camera(eye, target, width=W, height=H, fov=FOV)
    return np.array([[cam.right[0], cam.up[0], -cam.forward[0], cam.position[0]],
                     [cam.right[1], cam.up[1], -cam.forward[1], cam.position[1]],
                     [cam.right[2], cam.up[2], -cam.forward[2], cam.position[2]]], dtype=np.float32)


def single_world(world_poses, device="auto"):
    inst = [ng.Instance(i.mesh, ng.Transform(np.vstack([m, [0, 0, 0, 1]]).astype(np.float32)), i.material)
            for i, m in zip(instances(), world_poses)]
    kw = scene_kwargs()
    kw["instances"] = inst
    return ng.Renderer(W, H, device=device, **kw)


def test_shape_and_properties():
    b = ng.BatchRenderer(W, H, 5, 2, **scene_kwargs())
    assert (b.world_count, b.cameras_per_world, b.view_count, b.instances_per_world) == (5, 2, 10, 2)
    assert (b.width, b.height) == (W, H) and b.device in ("cpu", "cuda")


def test_each_view_matches_an_ordinary_renderer():
    worlds = 4
    b = ng.BatchRenderer(W, H, worlds, **scene_kwargs())
    p = poses(worlds)
    b.set_instance_transforms(p)
    cam = camera_matrix()
    out = b.render(np.tile(cam, (worlds, 1, 1)), FOV, depth=True, object_ids=True)
    assert out["rgb"].shape == (worlds, H, W, 3) and out["rgb"].dtype == np.uint8
    assert out["depth"].shape == (worlds, H, W) and out["object_ids"].dtype == np.int32
    for w in range(worlds):
        one = single_world(p[w]).render_frame(ng.make_camera((0, 0.5, 0), (0, 0, -8), width=W, height=H, fov=FOV),
                                              depth=True, object_ids=True)
        assert (one["object_ids"] != out["object_ids"][w]).mean() < 0.001
        assert np.abs(one["rgb"].astype(int) - out["rgb"][w].astype(int)).mean() < 0.1


def test_worlds_do_not_see_one_another():
    b = ng.BatchRenderer(W, H, 2, instances=[ng.Instance(0)], meshes=[ng.make_box((1, 1, 1))],
                         lights=[ng.Light((0, 5, 5), 1.0)])
    b.set_instance_transforms(np.array([[[ng.make_transform((0, 0, -8)).matrix]], [[ng.make_transform((0, 0, -3)).matrix]]]).reshape(2, 1, 3, 4))
    out = b.render(np.tile(camera_matrix((0, 0, 0), (0, 0, -1)), (2, 1, 1)), FOV, rgb=False, depth=True)
    centre = out["depth"][:, H // 2, W // 2]
    assert centre == pytest.approx([7.0, 2.0], abs=0.02)  # each sees its own box, not the nearer one next door


def test_rays_per_world():
    b = ng.BatchRenderer(W, H, 2, instances=[ng.Instance(0)], meshes=[ng.make_box((1, 1, 1))])
    b.set_instance_transforms(np.array([ng.make_transform((0, 0, -8)).matrix, ng.make_transform((0, 0, -3)).matrix],
                                       dtype=np.float32).reshape(2, 1, 3, 4))
    hits = b.cast_rays(np.zeros((2, 1, 3), np.float32), np.tile(np.array([0, 0, -1], np.float32), (2, 1, 1)))
    assert hits["distance"].shape == (2, 1) and hits["normal"].shape == (2, 1, 3)
    assert hits["distance"][:, 0] == pytest.approx([7.0, 2.0], abs=1e-3)
    assert (hits["object_id"] == ng.OBJECT_INSTANCE).all()


def test_body_poses_place_instances():
    worlds = 3
    b = ng.BatchRenderer(W, H, worlds, **scene_kwargs())
    b.bind_bodies(body_of_instance=np.array([0, 1], np.int32),
                  local_pose=np.array([[0.5, 0, 0, 0, 0, 0, 1], [0, 0.4, 0, 0, 0, 0, 1]], np.float32),
                  scale=np.array([[1, 1, 1], [0.5, 0.5, 0.5]], np.float32), bodies_per_world=2)
    bodies = np.zeros((worlds, 2, 7), np.float32)
    bodies[..., 6] = 1
    for w in range(worlds):
        bodies[w, 0, :3] = (-1.5, 0, -6 - w)
        bodies[w, 1, :3] = (1.5, 0, -6 - w)
    b.set_body_poses(bodies)
    by_hand = np.zeros((worlds, 2, 3, 4), np.float32)  # no rotation: the matrix is just scale and a position
    for w in range(worlds):
        by_hand[w, 0] = [[1, 0, 0, bodies[w, 0, 0] + 0.5], [0, 1, 0, 0], [0, 0, 1, bodies[w, 0, 2]]]
        by_hand[w, 1] = [[.5, 0, 0, bodies[w, 1, 0]], [0, .5, 0, 0.4], [0, 0, .5, bodies[w, 1, 2]]]
    other = ng.BatchRenderer(W, H, worlds, **scene_kwargs())
    other.set_instance_transforms(by_hand)
    cams = np.tile(camera_matrix(), (worlds, 1, 1))
    a, c = b.render(cams, FOV, object_ids=True), other.render(cams, FOV, object_ids=True)
    assert (a["object_ids"] != c["object_ids"]).mean() < 0.001 and (a["object_ids"] >= ng.OBJECT_INSTANCE).any()


def test_zero_copy_on_the_cpu_shares_memory():
    b = ng.BatchRenderer(W, H, 2, device="cpu", **scene_kwargs())
    b.set_instance_transforms(poses(2))
    cams = np.tile(camera_matrix(), (2, 1, 1))
    first = b.render(cams, FOV, rgb=False, depth=True, zero_copy=True)["depth"]
    kept = first.copy()
    second = b.render(np.tile(camera_matrix((2, 0, 0)), (2, 1, 1)), FOV, rgb=False, depth=True, zero_copy=True)["depth"]
    assert np.shares_memory(first, second) and not np.array_equal(first, kept)


def test_bad_input_is_rejected_with_a_message_saying_what_was_expected():
    b = ng.BatchRenderer(W, H, 3, **scene_kwargs())
    with pytest.raises(ValueError, match="world_count, instances_per_world, 3, 4"):
        b.set_instance_transforms(np.zeros((3, 1, 3, 4), np.float32))
    with pytest.raises(ValueError, match="singular"):
        b.set_instance_transforms(np.zeros((3, 2, 3, 4), np.float32))
    nan = poses(3)
    nan[1, 0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        b.set_instance_transforms(nan)
    cams = np.tile(camera_matrix(), (3, 1, 1))
    with pytest.raises(ValueError, match="3 cameras"):
        b.render(cams[:2], FOV)
    with pytest.raises(ValueError, match="at least one output"):
        b.render(cams, FOV, rgb=False)
    with pytest.raises(ValueError, match="fov"):
        b.render(cams, 0.0)
    with pytest.raises(ValueError, match="bind_bodies"):
        b.set_body_poses(np.zeros((3, 2, 7), np.float32))
    with pytest.raises(ValueError, match="world_count"):
        ng.BatchRenderer(W, H, 0)
    with pytest.raises(ValueError, match="rays"):
        b.cast_rays(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))


# --------------------------------------------------------------- the GPU hand-off --

@pytest.fixture(scope="module")
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch has no CUDA device")
    try:
        ng.BatchRenderer(W, H, 1, device="cuda")
    except ng.CudaError:
        pytest.skip("no usable CUDA GPU for neraulgraphics")
    return torch


def test_poses_and_cameras_on_the_gpu_give_the_same_images_as_numpy(torch_cuda):
    torch = torch_cuda
    worlds = 6
    p, cams = poses(worlds), np.tile(camera_matrix(), (worlds, 1, 1))
    on_cpu = ng.BatchRenderer(W, H, worlds, device="cpu", **scene_kwargs())
    on_cpu.set_instance_transforms(p)
    want = on_cpu.render(cams, FOV, depth=True, object_ids=True)

    gpu = ng.BatchRenderer(W, H, worlds, device="cuda", **scene_kwargs())
    gpu.set_instance_transforms(torch.from_numpy(p).cuda())         # poses used where they are
    out = gpu.render(torch.from_numpy(cams).cuda(), FOV, depth=True, object_ids=True, zero_copy=True)
    t = {k: torch.from_dlpack(v) for k, v in out.items()}
    assert all(v.is_cuda for v in t.values()) and tuple(t["rgb"].shape) == (worlds, H, W, 3)
    assert (t["object_ids"].cpu().numpy() != want["object_ids"]).mean() < 0.001
    both = np.isfinite(t["depth"].cpu().numpy()) & np.isfinite(want["depth"]) & (t["object_ids"].cpu().numpy() == want["object_ids"])
    assert np.abs(t["depth"].cpu().numpy()[both] - want["depth"][both]).max() < 0.01


def test_body_poses_from_a_gpu_tensor(torch_cuda):
    torch = torch_cuda
    worlds = 8
    kw = scene_kwargs()
    bind = dict(body_of_instance=np.array([0, 1], np.int32), local_pose=np.array([[0, 0, 0, 0, 0, 0, 1]] * 2, np.float32),
                scale=np.ones((2, 3), np.float32), bodies_per_world=2)
    bodies = np.zeros((worlds, 2, 7), np.float32)
    bodies[..., 6] = 1
    bodies[..., 0] = np.linspace(-2, 2, 2)[None]
    bodies[..., 2] = -6 - np.arange(worlds)[:, None]
    cams = np.tile(camera_matrix(), (worlds, 1, 1))
    results = []
    for device, convert in [("cpu", lambda a: a), ("cuda", lambda a: torch.from_numpy(a).cuda())]:
        b = ng.BatchRenderer(W, H, worlds, device=device, **kw)
        b.bind_bodies(**bind)
        b.set_body_poses(convert(bodies))
        results.append(b.render(cams, FOV, rgb=False, object_ids=True)["object_ids"])
    assert (results[0] != results[1]).mean() < 0.001 and (results[0] >= ng.OBJECT_INSTANCE).any()


def test_a_thousand_worlds_render_and_stay_in_gpu_memory(torch_cuda):
    torch = torch_cuda
    worlds = 1000
    b = ng.BatchRenderer(W, H, worlds, device="cuda", **scene_kwargs())
    b.set_instance_transforms(torch.from_numpy(poses(worlds)).cuda())
    out = b.render(torch.from_numpy(np.tile(camera_matrix(), (worlds, 1, 1))).cuda(), FOV, rgb=False, depth=True, zero_copy=True)
    depth = torch.from_dlpack(out["depth"])
    assert tuple(depth.shape) == (worlds, H, W) and depth.is_cuda
    assert float(torch.isfinite(depth).float().mean()) > 0.1  # something is drawn in every part of the batch
    assert b.last_kernel_ms > 0
