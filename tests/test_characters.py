"""Tests for the articulated humanoids (examples/scenery/characters.py): a skeleton keeps its bone lengths, stands on the ground, walks at
its speed, swings its limbs in opposition, and is the same every time."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "scenery"))

import characters  # noqa: E402
from characters import Humanoid  # noqa: E402

BONES = ["torso", "head", "l_thigh", "l_shin", "r_thigh", "r_shin", "l_upper_arm", "l_forearm", "r_upper_arm", "r_forearm"]


def lengths(person, t):
    pose, _ = person.pose(t)
    return {name: np.linalg.norm(pose[name][1] - pose[name][0]) for name in BONES}


def test_bones_never_stretch():
    person = Humanoid((0, 0), 0.7, 1.4)
    first = lengths(person, 0.0)
    for t in np.linspace(0, 5, 60):
        now = lengths(person, t)
        for name in BONES:
            assert now[name] == pytest.approx(first[name], rel=1e-4), name


def test_joints_stay_joined():
    person = Humanoid((0, 0), 0.0, 1.4)
    for t in np.linspace(0, 4, 40):
        pose, _ = person.pose(t)
        assert np.allclose(pose["l_thigh"][1], pose["l_shin"][0], atol=1e-5)          # the knee
        assert np.allclose(pose["r_upper_arm"][1], pose["r_forearm"][0], atol=1e-5)   # the elbow


@pytest.mark.parametrize("speed", [0.0, 1.0, 1.6])
def test_the_lower_foot_is_on_the_ground_and_nothing_goes_through_it(speed):
    person = Humanoid((3, -2), 1.1, speed)
    low = []
    for t in np.linspace(0, 6, 120):
        pose, _ = person.pose(t)
        feet = [min(pose["l_foot"][0][2], pose["l_foot"][1][2]), min(pose["r_foot"][0][2], pose["r_foot"][1][2])]
        low.append(min(feet))
        for a, b in pose.values():
            assert min(a[2], b[2]) > -0.02                                         # no segment end below the ground
    assert np.abs(np.array(low)).max() < 0.06                                       # and some foot is always down


def test_it_walks_at_its_speed_along_its_heading():
    person = Humanoid((1.0, 2.0), np.pi / 2, 1.5)
    a, _ = person.pose(0.0)
    b, _ = person.pose(4.0)
    moved = np.asarray(b["pelvis"][0]) - np.asarray(a["pelvis"][0])
    assert moved[:2] == pytest.approx((0.0, 6.0), abs=0.05)                         # 1.5 m/s for 4 s, north


def test_arms_swing_against_the_legs():
    person = Humanoid((0, 0), 0.0, 1.4)
    forward = person.forward()
    seen = 0
    for t in np.linspace(0, 3, 90):
        pose, _ = person.pose(t)
        leg = np.dot(pose["r_shin"][1] - pose["r_thigh"][0], forward)
        arm = np.dot(pose["r_forearm"][1] - pose["r_upper_arm"][0], forward)
        if abs(leg) > 0.15:
            seen += 1
            assert leg * arm < 0                                                    # the right arm is back when the right leg is forward
    assert seen > 20


def test_standing_people_do_not_travel():
    person = Humanoid((0, 0), 0.3, 0.0)
    a, _ = person.pose(0.0)
    b, _ = person.pose(10.0)
    assert np.allclose(a["pelvis"][0], b["pelvis"][0], atol=1e-5)


def test_matrices_are_renderer_axes_and_repeatable():
    person = Humanoid((0, 0), 0.0, 1.4, height=1.8)
    m = person.matrices(1.3)
    assert m.shape == (len(characters.PARTS), 3, 4) and np.isfinite(m).all()
    assert np.array_equal(m, Humanoid((0, 0), 0.0, 1.4, height=1.8).matrices(1.3))
    head = m[characters.PARTS.index("head"), :, 3]
    assert 1.5 < head[1] < 1.9                                                      # the head is up the renderer's Y axis, about 1.7 m
    for i in range(len(characters.PARTS)):                                           # every segment is a proper ellipsoid, none flattened to a sheet or a line
        assert abs(np.linalg.det(m[i, :, :3])) > 1e-7, characters.PARTS[i]


def test_a_crowd_keeps_off_the_centre_line():
    structure = SimpleNamespace(position=np.array([[0.0, 0.0, 5.0]]), half=np.array([[6.0, 4.0, 5.0]]))   # any structure: one 12 x 8 x 10 m block
    cx = (structure.position[:, 0].min() + structure.position[:, 0].max()) / 2
    people = characters.crowd(structure, seed=1)
    assert len(people) == 7 and any(p.speed == 0 for p in people) and any(p.speed > 0 for p in people)
    for p in people:
        assert abs(p.position[0] - cx) > 4.0
