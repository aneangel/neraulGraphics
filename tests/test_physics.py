"""Tests for the deterministic physics world (neraulgraphics.physics). They need the package installed with
NERAULGRAPHICS_BUILD_PHYSICS=ON and skip otherwise.

The golden hash below is the guarantee: it is the exact state of a scenario on the machine that recorded it, and every machine
that runs the same scenario must reproduce it bit for bit. If it fails on another CPU, operating system or compiler, the
simulation is no longer deterministic there (or Jolt changed, in which case record a new one)."""

import numpy as np
import pytest

from neraulgraphics import physics

pytestmark = pytest.mark.skipif(not physics.available(), reason="built without NERAULGRAPHICS_BUILD_PHYSICS")

CUBE = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32)
TOWER_HASH = 0xF438D50C98A681E7     # five cubes and a ball, 240 frames of 1/120 s in 4 substeps, recorded on Linux x86-64


def tower(threads=0, frames=240):
    world = physics.World(threads=threads, max_bodies=64)
    world.add_ground(friction=0.7)
    for k in range(5):
        world.add_convex_hull(CUBE, (0.1 * k, 0.05 * k, 1.0 + 1.2 * k), density=2400.0)
    world.add_sphere(0.3, (0.3, 0.0, 9.0), 7800.0)
    world.finish_setup()
    for _ in range(frames):
        world.step(1 / 120, 4)
    return world


def test_bodies_stack_and_come_to_rest():
    world = tower(frames=720)                                              # 6 s: long enough for the drop to settle
    z = world.poses()[:, 2]
    assert z[:5] == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5], abs=0.01)     # cubes dropped from a height stack on one another
    assert z[5] == pytest.approx(5.3, abs=0.02)                            # and the ball rests on the top one
    assert np.abs(world.velocities()[:5]).max() < 0.05                      # the cubes are at rest (the ball may still roll: nothing damps it)
    assert world.poses().shape == (6, 7) and world.body_count == 6          # the ground is not one of the numbered bodies


def test_the_same_state_hash_on_every_run_and_every_thread_count():
    hashes = {threads: tower(threads).state_hash() for threads in (1, 2, 5, 0)}
    assert len(set(hashes.values())) == 1, {k: hex(v) for k, v in hashes.items()}


def test_the_golden_hash_is_reproduced():
    assert tower(threads=1).state_hash() == TOWER_HASH


def test_a_different_input_gives_a_different_hash():
    a, b = tower(frames=60), tower(frames=60)
    assert a.state_hash() == b.state_hash()
    b.fire(0, (0.0, 0.0, 6.0), (1.0, 0.0, 0.0))
    assert a.state_hash() != b.state_hash()


def test_the_hash_matches_frame_by_frame():
    a, b = tower(threads=1, frames=0), tower(threads=4, frames=0)
    for _ in range(120):
        a.step(1 / 120, 2)
        b.step(1 / 120, 2)
        assert a.state_hash() == b.state_hash()


def test_reset_restores_the_start_exactly():
    world = tower(frames=0)
    start_q, start_qd = world.poses().copy(), world.velocities().copy()
    start = world.state_hash()
    for _ in range(90):
        world.step(1 / 120, 2)
    assert world.state_hash() != start
    world.fire(5, (0.0, 0.0, 12.0), (0.0, 0.0, 0.0))        # a body that was moved by hand comes back too
    world.reset(start_q, start_qd)
    assert world.state_hash() == start
    for _ in range(30):
        world.step(1 / 120, 2)                              # and it runs again from there
    assert np.isfinite(world.poses()).all()


def test_bad_inputs_are_rejected():
    world = physics.World(max_bodies=8)
    with pytest.raises(ValueError):
        world.add_convex_hull(np.zeros((3, 3), np.float32), (0, 0, 0))
    with pytest.raises(ValueError):
        world.add_convex_hull(np.zeros((8, 2), np.float32), (0, 0, 0))
    world.add_sphere(0.5, (0, 0, 1))
    world.finish_setup()
    with pytest.raises(ValueError):
        world.step(0.0)
    with pytest.raises(ValueError):
        world.fire(5, (0, 0, 0), (0, 0, 0))
