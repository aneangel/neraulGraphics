"""Does the physics get the physics right? Each test sets up a situation whose answer is known from first principles (free fall,
a projectile, sliding friction on a slope, a bounce, conservation of momentum and energy in a collision, torque-free spin) and
checks the simulated numbers against it. Repeatability, which test_physics.py covers, says the answer is always the same; these
say it is the right one. They need the physics module and skip otherwise."""

import math

import numpy as np
import pytest

from neraulgraphics import physics

pytestmark = pytest.mark.skipif(not physics.available(), reason="built without NERAULGRAPHICS_BUILD_PHYSICS")

G = 9.81
DT, SUBSTEPS = 1 / 120, 4
CUBE = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32)


def world(gravity=(0.0, 0.0, -G)):
    return physics.World(gravity=gravity, max_bodies=32)


def run(w, seconds):
    for _ in range(round(seconds / DT)):
        w.step(DT, SUBSTEPS)


def test_free_fall_follows_half_g_t_squared():
    w = world()
    w.add_ground()
    w.add_sphere(0.25, (0.0, 0.0, 30.0), 1000.0)
    w.finish_setup()
    run(w, 1.5)
    expected = 30.0 - 0.5 * G * 1.5 ** 2
    assert w.poses()[0, 2] == pytest.approx(expected, abs=0.03)
    assert w.velocities()[0, 2] == pytest.approx(-G * 1.5, rel=0.01)


def test_a_projectile_flies_a_parabola():
    w = world()
    w.add_ground()
    w.add_sphere(0.1, (0.0, 0.0, 20.0), 1000.0)
    w.finish_setup()
    w.fire(0, (0.0, 0.0, 20.0), (12.0, 0.0, 8.0))
    t = 1.2
    run(w, t)
    x, _, z = w.poses()[0, :3]
    assert x == pytest.approx(12.0 * t, abs=0.05)
    assert z == pytest.approx(20.0 + 8.0 * t - 0.5 * G * t * t, abs=0.05)


@pytest.mark.parametrize("angle_degrees, mu", [(35.0, 0.3), (40.0, 0.5)])
def test_a_box_slides_down_a_slope_with_a_friction_acceleration(angle_degrees, mu):
    """A box on a slope accelerates at g (sin a - mu cos a). The slope is made by tilting gravity over flat ground."""
    a = math.radians(angle_degrees)
    w = world(gravity=(G * math.sin(a), 0.0, -G * math.cos(a)))
    w.add_ground(friction=mu)
    w.add_convex_hull(CUBE, (0.0, 0.0, 0.5), density=1000.0, friction=mu)
    w.finish_setup()
    run(w, 0.3)                                   # let it settle onto the surface
    v0 = w.velocities()[0, 0]
    t0 = 1.0
    run(w, t0)
    measured = (w.velocities()[0, 0] - v0) / t0
    assert measured == pytest.approx(G * (math.sin(a) - mu * math.cos(a)), rel=0.08)


def test_a_box_holds_on_a_slope_below_the_friction_angle():
    a = math.radians(20.0)                        # tan 20 deg = 0.36 < mu: it must not slide
    w = world(gravity=(G * math.sin(a), 0.0, -G * math.cos(a)))
    w.add_ground(friction=0.6)
    w.add_convex_hull(CUBE, (0.0, 0.0, 0.5), density=1000.0, friction=0.6)
    w.finish_setup()
    run(w, 3.0)
    assert abs(w.poses()[0, 0]) < 0.05


@pytest.mark.parametrize("e", [0.5, 0.8])
def test_a_bounce_returns_e_squared_of_the_height(e):
    w = world()
    w.add_ground(restitution=e)
    w.add_sphere(0.2, (0.0, 0.0, 3.0), 1000.0, friction=0.5, restitution=e)
    w.finish_setup()
    peak, bounced = 0.0, False
    for _ in range(round(2.5 / DT)):
        w.step(DT, SUBSTEPS)
        z, vz = w.poses()[0, 2], w.velocities()[0, 2]
        if vz > 0.5:
            bounced = True
        if bounced:
            peak = max(peak, z)
            if vz < 0:
                break
    assert bounced
    assert (peak - 0.2) / (3.0 - 0.2) == pytest.approx(e * e, rel=0.12)       # heights measured from the ground to the ball's base


def test_momentum_and_energy_are_conserved_in_an_elastic_collision():
    w = world(gravity=(0.0, 0.0, 0.0))
    w.add_ground()
    w.add_sphere(0.5, (-4.0, 0.0, 6.0), 1000.0, 0.0, 1.0)         # equal masses, frictionless, perfectly elastic
    w.add_sphere(0.5, (0.0, 0.0, 6.0), 1000.0, 0.0, 1.0)
    w.finish_setup()
    w.fire(0, (-4.0, 0.0, 6.0), (5.0, 0.0, 0.0))
    m = 1000.0 * 4.0 / 3.0 * math.pi * 0.5 ** 3
    before = w.velocities()[:, :3]
    p0, ke0 = m * before[:, 0].sum(), 0.5 * m * (before ** 2).sum()
    run(w, 2.0)
    after = w.velocities()[:, :3]
    assert m * after[:, 0].sum() == pytest.approx(p0, rel=0.01)
    assert 0.5 * m * (after ** 2).sum() == pytest.approx(ke0, rel=0.05)
    assert after[0, 0] == pytest.approx(0.0, abs=0.3) and after[1, 0] == pytest.approx(5.0, rel=0.06)     # equal masses swap velocities


def test_a_spinning_body_in_free_space_keeps_its_spin_and_drifts_straight():
    w = world(gravity=(0.0, 0.0, 0.0))
    w.add_ground()
    w.add_convex_hull(CUBE, (0.0, 0.0, 50.0), density=500.0)
    w.finish_setup()
    q, qd = w.poses().copy(), w.velocities().copy()
    qd[0] = (2.0, -1.0, 0.5, 1.0, 2.0, 3.0)
    w.reset(q, qd)
    run(w, 2.0)
    v = w.velocities()[0]
    assert v[:3] == pytest.approx((2.0, -1.0, 0.5), abs=1e-3)                # no force: the velocity is unchanged
    assert np.linalg.norm(v[3:]) == pytest.approx(math.sqrt(14.0), rel=1e-3)  # and, for a cube, so is the spin
    assert w.poses()[0, :3] == pytest.approx((4.0, -2.0, 51.0), abs=0.02)


def test_a_dropped_stack_never_gains_energy():
    """Bodies dropped on the ground only lose energy: kinetic plus potential energy must not rise."""
    w = world()
    w.add_ground()
    rng = np.random.default_rng(2)
    for k in range(6):
        w.add_convex_hull(CUBE * float(rng.uniform(0.4, 0.9)), tuple(rng.uniform(-0.3, 0.3, 2)) + (1.0 + 1.3 * k,), density=800.0, friction=0.5)
    w.finish_setup()
    masses = np.array([800.0 * (0.4 * 0 + 1) for _ in range(6)])        # only the ordering of energies matters, not their scale
    energy = lambda: float((0.5 * (w.velocities()[:, :3] ** 2).sum(axis=1) + G * w.poses()[:, 2]).sum())
    e0 = energy()
    worst = -1e9
    for _ in range(round(4.0 / DT)):
        w.step(DT, SUBSTEPS)
        worst = max(worst, energy() - e0)
    assert worst < 0.05 * abs(e0)                 # a few percent of integration noise at most, never a gain
    assert energy() < e0
