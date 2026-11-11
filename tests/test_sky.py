"""The physical sky (Preetham daylight model): the properties of a real clear sky that a gradient lacks."""

import math

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 256, 128


def direction(sky, elevation, azimuth):
    """The sky's radiance in the direction elevation (from the horizon) and azimuth (round the up axis, 0 along +x)."""
    h, w = sky.shape[:2]
    theta, phi = math.pi / 2 - elevation, azimuth % (2 * math.pi)
    return sky[min(int(theta / math.pi * h), h - 1), min(int(phi / (2 * math.pi) * w), w - 1)]


def irradiance_from(sky, normal):
    """The light the whole map (sun included) gives a surface with this unit normal: the integral of L cos over the sphere."""
    h, w = sky.shape[:2]
    theta = (np.arange(h) + 0.5) / h * math.pi
    phi = (np.arange(w) + 0.5) / w * 2 * math.pi
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    d = np.stack([st * np.cos(phi)[None, :], np.broadcast_to(ct, (h, w)), st * np.sin(phi)[None, :]], -1)
    solid = (2 * math.pi / w) * (math.pi / h) * np.broadcast_to(st, (h, w))
    cos = np.maximum(d @ np.asarray(normal), 0.0)
    return float((sky.mean(axis=-1) * cos * solid).sum())


def test_the_sky_is_a_deep_blue_overhead_and_pale_towards_the_horizon():
    sky = ng.physical_sky(sun_elevation=0.9, width=W, height=H)
    zenith, near_horizon = direction(sky, 1.5, 3.0), direction(sky, 0.08, 3.0)             # looking away from the sun
    blueness = lambda c: c[2] / (c[0] + c[1] + c[2])
    assert blueness(zenith) > blueness(near_horizon)                                       # more saturated blue overhead
    assert near_horizon.sum() > zenith.sum()                                               # but brighter towards the horizon
    assert zenith[2] > zenith[0]                                                           # and the zenith is blue, not grey


def test_the_sky_is_brightest_round_the_sun():
    sky = ng.physical_sky(sun_elevation=0.6, sun_radius=0.01, width=W, height=H)
    near = direction(sky, 0.6, 0.0 + 0.25).mean()
    far = direction(sky, 0.6, 0.0 + math.pi).mean()
    assert near > 2.0 * far


def test_the_sun_gives_exactly_the_strength_asked_for_and_the_sky_its_share():
    sky = ng.physical_sky(sun_elevation=1.0, sun_strength=3.0, sky_ratio=0.2, sun_radius=0.03, width=512, height=256)
    toward_sun = [math.cos(1.0), math.sin(1.0), 0.0]
    total = irradiance_from(sky, toward_sun)
    without_sun = irradiance_from(np.minimum(sky, np.percentile(sky, 99.9)), toward_sun)       # the sky alone: sun texels clipped
    assert total > 3.0                                                                         # the sun (3.0) plus some sky
    horizontal = irradiance_from(sky, [0, 1, 0])
    assert horizontal == pytest.approx(3.0 * math.sin(1.0) + 0.2 * 3.0 * 1.0, rel=0.25)        # sun x sin(elevation) plus the sky's share


def test_a_low_sun_is_redder_and_dimmer():
    # a wide disc (0.06 rad), so that at this resolution the probe is certain to land on the sun itself
    noon, dusk = ng.physical_sky(sun_elevation=1.4, sun_radius=0.06, width=W, height=H), ng.physical_sky(sun_elevation=0.06, sun_radius=0.06, width=W, height=H)
    sun_noon = direction(noon, 1.4, 0.0)
    sun_dusk = direction(dusk, 0.06, 0.0)
    assert sun_dusk[0] / sun_dusk[2] > 2.0 * sun_noon[0] / sun_noon[2]                         # redder: the blue is scattered away on the long path
    assert sun_dusk.sum() < sun_noon.sum()


def test_a_hazier_sky_is_whiter_and_a_clear_one_bluer():
    clear, hazy = ng.physical_sky(turbidity=2.0, sun_elevation=0.9, width=W, height=H), ng.physical_sky(turbidity=8.0, sun_elevation=0.9, width=W, height=H)
    c, h = direction(clear, 1.2, 3.0), direction(hazy, 1.2, 3.0)
    assert c[2] / c[0] > h[2] / h[0]


def test_below_the_horizon_is_a_ground_lit_by_the_sun_and_sky():
    sky = ng.physical_sky(sun_elevation=0.8, sun_strength=3.0, ground_albedo=0.3, width=W, height=H)
    ground = direction(sky, -0.5, 1.0)
    assert ground[0] == pytest.approx(ground[1], rel=1e-5) == pytest.approx(ground[2], rel=1e-5)     # grey
    assert ground[0] == pytest.approx(0.3 * (3.0 * math.sin(0.8) + 0.6) / math.pi, rel=1e-3)


def test_bad_arguments_are_rejected():
    for bad in (dict(sun_elevation=0.0), dict(sun_elevation=2.0), dict(turbidity=1.0), dict(turbidity=20.0), dict(sun_radius=0.0), dict(sun_strength=-1.0), dict(ground_albedo=1.5)):
        with pytest.raises(ValueError):
            ng.physical_sky(**bad)


def test_the_map_is_finite_non_negative_and_usable():
    sky = ng.physical_sky(width=64, height=32)
    assert sky.shape == (32, 64, 3) and sky.dtype == np.float32 and np.isfinite(sky).all() and (sky >= 0).all()
    ng.Environment(up=(0, 1, 0), map=sky)                          # the renderer accepts it
