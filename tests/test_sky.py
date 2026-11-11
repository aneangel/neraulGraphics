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


# ------------------------------------------------------------------------------------------------ the sun as a light --


def test_a_sky_without_the_sun_has_no_disc_and_the_same_sky():
    with_sun = ng.physical_sky(sun_elevation=0.8, sun_radius=0.05, width=W, height=H)
    without = ng.physical_sky(sun_elevation=0.8, sun_radius=0.05, width=W, height=H, include_sun=False)
    assert without.max() < 0.05 * with_sun.max()                           # the disc is gone
    far = np.ones(with_sun.shape[:2], bool)
    sun_dir = direction(with_sun, 0.8, 0.0)
    assert not np.allclose(with_sun[tuple(np.unravel_index(np.argmax(with_sun.mean(-1)), with_sun.shape[:2]))], without[tuple(np.unravel_index(np.argmax(with_sun.mean(-1)), with_sun.shape[:2]))])
    mask = with_sun.mean(-1) < 0.5 * with_sun.max()                        # everywhere but the disc, the sky is identical
    assert np.allclose(with_sun[mask], without[mask])


def test_the_sun_light_points_at_the_sun_with_its_strength_and_colour():
    light = ng.sun_light(sun_elevation=0.6, sun_azimuth=1.0, turbidity=2.5, sun_strength=3.0, sun_radius=0.02, distance=500.0)
    p = np.array(light.position)
    assert np.linalg.norm(p) == pytest.approx(500.0, rel=1e-5)
    in_map = np.array([math.cos(0.6) * math.cos(1.0), math.sin(0.6), math.cos(0.6) * math.sin(1.0)])
    assert np.allclose(p / 500.0, in_map @ ng.environment_frame((0, 1, 0)), atol=1e-5)          # in the environment's frame, not the world's axes
    assert light.intensity == pytest.approx(3.0 / math.pi)                 # a Light's intensity is the irradiance over pi
    assert light.radius == pytest.approx(500.0 * math.tan(0.02), rel=1e-5)
    low = ng.sun_light(sun_elevation=0.06)
    assert low.color[0] / low.color[2] > 2.0 * light.color[0] / light.color[2]   # a low sun is redder
    with pytest.raises(ValueError):
        ng.sun_light(sun_elevation=0.0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if ng.cuda_available() else []))
def test_the_sun_as_a_light_lights_a_floor_like_the_sun_in_the_map(device):
    """The same sun, two ways: as a disc in the sky map (found by random rays) and as an explicit light. The floor under it must be as bright."""
    elevation, strength = 0.9, 3.0
    floor = ng.Plane(y=0.0, half_extent_x=80.0, z_min=-120.0, z_max=10.0, checker_scale=0.0001,
                     material=ng.Material(diffuse_color=(0.7, 0.7, 0.7), diffuse_albedo=0.9, specular_albedo=0.0, roughness=1.0, metallic=0.0))

    def mean_floor(sky, lights):
        renderer = ng.Renderer(48, 48, [], lights, floor=floor, device=device)
        renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
        renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
        camera = ng.make_camera((0, 12, 0.01), (0, 0, -0.01), width=48, height=48, fov=0.6, world_up=(0, 0, -1))
        acc = 0.0
        for f in range(160):
            img = renderer.render(camera, f).astype(float) / 255.0
            lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
            acc += lin.mean()
        return acc / 160

    map_sun = mean_floor(ng.physical_sky(sun_elevation=elevation, sun_strength=strength, sun_radius=0.05, width=512, height=256), [])
    light_sun = mean_floor(ng.physical_sky(sun_elevation=elevation, sun_strength=strength, sun_radius=0.05, width=512, height=256, include_sun=False),
                           [ng.sun_light(sun_elevation=elevation, sun_strength=strength, sun_radius=0.05)])
    assert light_sun == pytest.approx(map_sun, rel=0.06)


def test_the_environment_frame_is_what_the_renderer_uses_for_y_up():
    frame = ng.environment_frame((0, 1, 0))
    assert np.allclose(frame, [[0, 0, -1], [0, 1, 0], [1, 0, 0]])           # right = -z, up = y, forward = +x, as environment.cpp builds it
    other = ng.environment_frame((0.2, -0.4, 0.9))
    assert np.allclose(other @ other.T, np.eye(3), atol=1e-9) and np.linalg.det(other) == pytest.approx(1.0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if ng.cuda_available() else []))
def test_the_explicit_sun_casts_its_shadow_where_the_sun_in_the_map_does(device):
    """A ball's shadow on a floor, with the sun as a disc in the map and as an explicit light: the pictures must agree, and must differ from
    the same sun turned a quarter circle (which is what placing the light in the world's axes instead of the environment's frame did)."""
    floor = ng.Plane(y=0.0, half_extent_x=60.0, z_min=-90.0, z_max=10.0, checker_scale=0.0001,
                     material=ng.Material(diffuse_color=(0.7, 0.7, 0.7), diffuse_albedo=0.9, specular_albedo=0.0, roughness=1.0, metallic=0.0))
    ball = [ng.Sphere((0.0, 1.0, -6.0), 1.0, ng.Material(diffuse_color=(0.7, 0.7, 0.7), roughness=1.0, metallic=0.0))]

    def picture(elevation, azimuth, explicit, light_azimuth=None):
        renderer = ng.Renderer(64, 64, ball, [], floor=floor, device=device)
        kw = dict(sun_elevation=elevation, sun_strength=3.0, sun_radius=0.05)
        sky = ng.physical_sky(sun_azimuth=azimuth, width=256, height=128, include_sun=not explicit, **kw)
        renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
        if explicit:
            renderer.set_lights([ng.sun_light(sun_azimuth=azimuth if light_azimuth is None else light_azimuth, **kw)])
        renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
        camera = ng.make_camera((0, 14, -5.99), (0, 0, -6.0), width=64, height=64, fov=0.9, world_up=(0, 0, -1))
        img = sum(renderer.render(camera, f).astype(float) for f in range(96)) / 96
        return img

    def shadow_centre(img):
        """The centroid, in pixels, of the pixels in the shadow: those much darker than the lit floor."""
        brightness = img.mean(axis=2)
        lit = np.percentile(brightness, 90)
        ys, xs = np.nonzero(brightness < 0.55 * lit)
        assert len(xs) > 25, "no shadow found"
        return np.array([xs.mean(), ys.mean()])

    in_map, explicit = shadow_centre(picture(0.5, 1.2, False)), shadow_centre(picture(0.5, 1.2, True))
    turned = shadow_centre(picture(0.5, 1.2, True, light_azimuth=1.2 + math.pi / 2))
    assert np.linalg.norm(in_map - explicit) < 2.5                          # the same place, to a couple of pixels
    assert np.linalg.norm(in_map - turned) > 6.0                            # and not where a light turned a quarter circle puts it
