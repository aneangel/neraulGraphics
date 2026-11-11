"""The code the examples share (examples/_live.py and examples/_stage.py): cameras, the physics stage and its conversion to the renderer's
axes. None of it needs a display."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import neraulgraphics as ng
from neraulgraphics import physics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import _live  # noqa: E402


class FakeWindow:
    """What a camera reads from a LiveWindow: held keys, mouse movement, the wheel."""

    def __init__(self, held=(), look=(0.0, 0.0), scroll=0.0):
        self._held, self._look, self._scroll = set(held), look, scroll

    def held(self, name):
        return name in self._held

    def take_look(self):
        look, self._look = self._look, (0.0, 0.0)
        return look

    def take_scroll(self):
        scroll, self._scroll = self._scroll, 0.0
        return scroll


def test_the_fly_camera_looks_down_minus_z_and_moves_where_it_looks():
    cam = _live.FlyCamera((0, 2, 5), speed=2.0)
    assert np.allclose(cam.forward(), (0, 0, -1))
    assert cam.update(FakeWindow(held={"w"}), 1.0)
    assert np.allclose(cam.position, (0, 2, 3))                             # 2 m/s for 1 s, forward
    cam.update(FakeWindow(held={"d"}), 1.0)
    assert np.allclose(cam.position, (2, 2, 3))                             # right is +x when looking down -z
    cam.update(FakeWindow(held={"e"}), 0.5)
    assert cam.position[1] == pytest.approx(3.0)


def test_the_fly_camera_turns_with_the_mouse_and_never_flips():
    cam = _live.FlyCamera((0, 0, 0), sensitivity=0.01)
    cam.update(FakeWindow(look=(100.0, 0.0)), 0.016)
    assert cam.yaw == pytest.approx(1.0) and np.allclose(cam.forward(), (math.sin(1.0), 0, -math.cos(1.0)))
    cam.update(FakeWindow(look=(0.0, 1e6)), 0.016)
    assert cam.pitch <= 1.5 and cam.forward()[1] > 0.9                      # straight up at most
    assert not cam.update(FakeWindow(), 0.016)                              # nothing held, nothing moved


def test_the_fast_key_multiplies_speed():
    slow, fast = _live.FlyCamera((0, 0, 0), speed=1.0), _live.FlyCamera((0, 0, 0), speed=1.0)
    slow.update(FakeWindow(held={"w"}), 1.0)
    fast.update(FakeWindow(held={"w", "lshift"}), 1.0)
    assert np.linalg.norm(fast.position) == pytest.approx(4.0 * np.linalg.norm(slow.position))


def test_the_orbit_camera_stays_on_its_sphere_and_zooms():
    cam = _live.OrbitCamera((1, 1, 1), 5.0)
    for look in ((40.0, 10.0), (-90.0, 30.0), (10.0, -500.0)):
        cam.update(FakeWindow(look=look), 0.016)
        assert np.linalg.norm(cam.position() - cam.center) == pytest.approx(5.0)
    cam.update(FakeWindow(scroll=3.0), 0.016)
    assert cam.distance == pytest.approx(5.0 * 0.9 ** 3)
    cam.update(FakeWindow(scroll=-1000.0), 0.016)
    assert cam.distance <= 500.0                                            # zoom is bounded


def test_the_cameras_make_renderer_cameras():
    for cam in (_live.FlyCamera((0, 1, 4)), _live.OrbitCamera((0, 0, 0), 4.0)):
        view = cam.camera(160, 90, 0.8)
        assert view.aspect == pytest.approx(160 / 90, rel=1e-3)


pytestmark_physics = pytest.mark.skipif(not physics.available(), reason="built without NERAULGRAPHICS_BUILD_PHYSICS")


def test_the_smooth_sphere_has_a_unit_normal_at_every_vertex():
    import _stage

    mesh = _stage.smooth_sphere(3)
    assert mesh.has_normals and mesh.triangle_count == 20 * 4 ** 3 and mesh.vertex_count == 10 * 4 ** 3 + 2


@pytestmark_physics
def test_a_stage_converts_simulation_axes_to_the_renderers():
    import _stage

    stage = _stage.Stage([_stage.Body("sphere", 0.5, ng.Material()), _stage.Body("box", (0.3, 0.2, 0.1), ng.Material())])
    stage.place(0, (1.0, 2.0, 3.0))                                        # simulation: x right, y forward, z up
    m = stage.matrices()
    assert np.allclose(m[0, :, 3], (1.0, 3.0, -2.0), atol=1e-5)             # renderer: (x, z, -y)
    assert np.allclose(m[0, :, :3], np.eye(3) * 0.5, atol=1e-5)             # unit sphere scaled to its radius, unrotated
    stage.place(1, (0.0, 0.0, 5.0))
    box = stage.matrices()[1, :, :3]
    assert np.allclose(np.abs(np.diag(box)), (0.3, 0.1, 0.2), atol=1e-5)    # half extents, with y and z exchanged by the axis change


@pytestmark_physics
def test_a_dropped_ball_rests_on_the_ground_and_parked_bodies_wait_far_away():
    import _stage

    stage = _stage.Stage([_stage.Body("sphere", 0.5, ng.Material(), restitution=0.0), _stage.Body("sphere", 0.3, ng.Material())])
    stage.place(0, (0.0, 0.0, 2.0))
    for _ in range(240):
        stage.step(1 / 60)
    assert stage.positions()[0, 1] == pytest.approx(0.5, abs=0.02)          # resting on y = 0, its radius up
    assert stage.positions()[1, 0] >= 500.0 and not stage.active[1]         # the other waits on the ground, far off
    stage.park(0)
    assert stage.positions()[0, 0] >= 500.0 and not stage.active[0]


@pytestmark_physics
def test_a_stage_is_repeatable():
    import _stage

    def run():
        stage = _stage.Stage([_stage.Body("sphere", 0.4, ng.Material()), _stage.Body("box", (0.3, 0.3, 0.3), ng.Material())])
        stage.place(0, (0, 0, 2), velocity=(1, 0.5, 0), angular=(2, 3, 1))
        stage.place(1, (0.3, 0.1, 4))
        for _ in range(180):
            stage.step(1 / 60)
        return stage.state_hash()

    assert run() == run()


@pytestmark_physics
def test_a_thrown_spinning_body_keeps_moving_and_the_wrong_kind_is_rejected():
    import _stage

    stage = _stage.Stage([_stage.Body("sphere", 0.3, ng.Material())], gravity=(0.0, 0.0, 0.0))
    stage.place(0, (0, 0, 5), velocity=(3.0, 0.0, 0.0), angular=(0.0, 0.0, 4.0))
    for _ in range(60):
        stage.step(1 / 60)
    assert stage.positions()[0, 0] == pytest.approx(3.0, abs=0.05)         # no gravity, no drag: 3 m/s for 1 s
    assert abs(stage.world.velocities()[0, 5]) == pytest.approx(4.0, abs=0.05)   # and the spin is unchanged
    with pytest.raises(ValueError):
        _stage.Stage([_stage.Body("cone", 1.0, ng.Material())])


@pytestmark_physics
def test_a_rotated_box_is_drawn_rotated_the_same_way_in_both_axis_conventions():
    """Rotating a body 90 degrees about the simulation's up axis (z) turns it about the renderer's up axis (y), the same way."""
    import _stage

    stage = _stage.Stage([_stage.Body("box", (0.4, 0.1, 0.1), ng.Material())], gravity=(0.0, 0.0, 0.0))
    stage.place(0, (0, 0, 5))
    q = stage.poses()[0].copy()
    # set a 90 degree turn about z through reset(): poses, then velocities
    quarter = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], np.float32)
    poses = stage.poses().copy()
    poses[0, 3:7] = quarter
    stage.world.reset(poses, np.zeros((1, 6), np.float32))
    m = stage.matrices()[0, :, :3]
    # the box's long side (0.4 along simulation x) now lies along simulation y, which is the renderer's -z
    long_axis = m @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(long_axis, (0.0, 0.0, -0.4), atol=1e-5)


def test_the_up_frame_is_right_handed_and_matches_the_simulation_axes_for_y_up():
    for up in ((0, 1, 0), (0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)):
        f = _live.up_frame(up)
        assert np.allclose(f @ f.T, np.eye(3), atol=1e-9) and np.linalg.det(f) == pytest.approx(1.0)
        assert np.allclose(f[2], np.asarray(up) / np.linalg.norm(up))
    assert np.allclose(_live.up_frame((0, 1, 0)), [[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def test_the_orbit_camera_works_for_any_up_axis():
    for up in ((0, 1, 0), (0, 0, 1)):
        cam = _live.OrbitCamera((0, 0, 0), 4.0, yaw=0.0, pitch=0.0, up=up)
        offset = cam.position()
        assert np.linalg.norm(offset) == pytest.approx(4.0) and abs(np.dot(offset, up)) < 1e-9      # level with the centre
        cam.pitch = 0.5
        assert np.dot(cam.position(), up) > 0                                                        # raised means up
    assert np.allclose(_live.OrbitCamera((0, 0, 0), 4.0, yaw=0.0, pitch=0.0).position(), (0, 0, 4.0))   # y-up, as before: from +z
