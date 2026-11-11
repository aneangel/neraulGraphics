"""Renderer.set_lights: replacing the lights between frames, as a flashlight that follows a camera needs. Runs on the CPU and, when there is
one, the GPU."""

import numpy as np
import pytest

import neraulgraphics as ng

DEVICES = ["cpu"] + (["cuda"] if ng.cuda_available() else [])
W, H = 96, 64


def scene(lights, device):
    floor = ng.Plane(y=-1.0, half_extent_x=20.0, z_min=-40.0, z_max=-1.0, material=ng.Material(diffuse_color=(0.8, 0.8, 0.8), diffuse_albedo=0.9, specular_albedo=0.0))
    renderer = ng.Renderer(W, H, [], lights, floor=floor, device=device)
    renderer.ambient = 0.0
    return renderer, ng.make_camera((0, 0, 0), (0, -0.4, -6), width=W, height=H, fov=1.1)


def contribution(renderer, camera, lights):
    """What the lights add: the picture with them minus the picture without any, so the sky behind the floor drops out."""
    renderer.set_lights([])
    dark = renderer.render(camera, 0).astype(float).mean(axis=2)
    renderer.set_lights(lights)
    return np.maximum(renderer.render(camera, 0).astype(float).mean(axis=2) - dark, 0.0)


def centroid_x(added):
    ys, xs = np.mgrid[0:H, 0:W]
    return (added * xs).sum() / added.sum()


@pytest.mark.parametrize("device", DEVICES)
def test_set_lights_gives_the_picture_a_fresh_renderer_would(device):
    lights = [ng.Light((3, 4, -5), intensity=1.2, radius=0.0), ng.Light((-4, 3, -8), intensity=0.8, radius=0.0)]
    fresh, camera = scene(lights, device)
    changed, _ = scene([ng.Light((0, 9, -3), intensity=3.0)], device)
    changed.set_lights(lights)
    assert np.array_equal(fresh.render(camera, 0), changed.render(camera, 0))


@pytest.mark.parametrize("device", DEVICES)
def test_a_moving_light_moves_the_lit_spot(device):
    renderer, camera = scene([], device)
    xs = [centroid_x(contribution(renderer, camera, [ng.Light((x, 1, -10), intensity=2.5, radius=0.0)])) for x in (-6.0, 0.0, 6.0)]
    assert xs[0] < xs[1] < xs[2]
    steps = (xs[1] - xs[0], xs[2] - xs[1])
    assert min(steps) > 1.5                                         # a broad diffuse patch shifts a few pixels for a 6 m move
    assert steps[0] == pytest.approx(steps[1], rel=0.2)             # equal moves of the light give equal moves of the spot


@pytest.mark.parametrize("device", DEVICES)
def test_a_light_adds_light_and_removing_it_takes_it_away(device):
    renderer, camera = scene([], device)
    added = contribution(renderer, camera, [ng.Light((0, 3, -8), intensity=2.0)])
    assert added.mean() > 5.0
    renderer.set_lights([])
    again = renderer.render(camera, 0).astype(float).mean(axis=2)
    renderer.set_lights([ng.Light((0, 3, -8), intensity=2.0)])
    assert (renderer.render(camera, 0).astype(float).mean(axis=2) - again).mean() == pytest.approx(added.mean(), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_bad_lights_are_rejected_and_change_nothing(device):
    renderer, camera = scene([ng.Light((0, 3, -8), intensity=2.0)], device)
    before = renderer.render(camera, 0)
    with pytest.raises(ValueError):
        renderer.set_lights([ng.Light((float("nan"), 0, 0))])
    with pytest.raises(ValueError):
        renderer.set_lights([ng.Light((0, 3, -8), radius=-1.0)])
    with pytest.raises(ValueError):
        renderer.set_lights([ng.Light((0, 3, -8)) for _ in range(ng.MAX_LIGHTS + 1)])
    assert np.array_equal(renderer.render(camera, 0), before)


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_after_set_lights():
    lights = [ng.Light((2, 5, -6), intensity=1.5, radius=0.5), ng.Light((-3, 2, -9), intensity=1.0, radius=0.0)]
    images = []
    for device in ("cpu", "cuda"):
        renderer, camera = scene([ng.Light((0, 1, 0), intensity=9.0)], device)
        renderer.set_lights(lights)
        images.append(renderer.render(camera, 0).astype(float))
    assert np.abs(images[0] - images[1]).mean() < 0.6


# ------------------------------------------------------------------------------------------------ lamps and spotlights --


def floor_scene(device):
    floor = ng.Plane(y=-1.0, half_extent_x=30.0, z_min=-60.0, z_max=0.0, material=ng.Material(diffuse_color=(0.8, 0.8, 0.8), diffuse_albedo=0.9, specular_albedo=0.0))
    renderer = ng.Renderer(160, 100, [], [], floor=floor, device=device)
    renderer.ambient = 0.0
    camera = ng.make_camera((0, 6, 0), (0, -1, -8), width=160, height=100, fov=1.4)      # from above, looking down the floor
    return renderer, camera


def linear(value):
    """A display value (0..255, sRGB encoded) back to linear light, so that ratios of light can be compared."""
    v = value / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def added(renderer, camera, light):
    renderer.set_lights([])
    dark = renderer.render(camera, 0).astype(float).mean(axis=2)
    renderer.set_lights([light])
    return np.maximum(renderer.render(camera, 0).astype(float).mean(axis=2) - dark, 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_a_plain_light_is_unchanged_by_the_new_fields(device):
    renderer, camera = floor_scene(device)
    a = added(renderer, camera, ng.Light((0, 3, -8), intensity=0.6))
    b = added(renderer, camera, ng.Light((0, 3, -8), intensity=0.6, falloff=0.0, spot_angle=0.0, direction=(1, 2, 3), spot_softness=0.9))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_falloff_follows_one_over_one_plus_distance_over_falloff_squared(device):
    renderer, camera = floor_scene(device)
    falloff = 3.0
    peaks = {}
    for height in (2.0, 4.0, 8.0):                                  # light above the floor (floor at y = -1): distance to the point below is the height
        picture = added(renderer, camera, ng.Light((0, -1 + height, -8), intensity=0.5, falloff=falloff))
        peaks[height] = linear(picture.max())
    expected = lambda h: 1.0 / (1.0 + (h / falloff) ** 2)
    assert peaks[4.0] / peaks[2.0] == pytest.approx(expected(4.0) / expected(2.0), rel=0.06)
    assert peaks[8.0] / peaks[2.0] == pytest.approx(expected(8.0) / expected(2.0), rel=0.1)


@pytest.mark.parametrize("device", DEVICES)
def test_a_spotlight_lights_a_disc_the_size_of_its_cone(device):
    renderer, camera = floor_scene(device)
    plain = added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8))
    spot = added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, direction=(0, -1, 0), spot_angle=0.35, spot_softness=0.2))
    assert (spot > 1.0).sum() < 0.35 * (plain > 1.0).sum()          # the cone lights far less of the floor than an all-round light
    assert (spot > 1.0).sum() > 40                                  # and something
    ys, xs = np.nonzero(spot > 1.0)
    py, px = np.unravel_index(np.argmax(plain), plain.shape)        # the brightest point of the plain light is below it, where the cone points
    assert abs(xs.mean() - px) < 8 and abs(ys.mean() - py) < 12


@pytest.mark.parametrize("device", DEVICES)
def test_a_spotlight_pointing_away_lights_nothing(device):
    renderer, camera = floor_scene(device)
    spot = added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, direction=(0, 1, 0), spot_angle=0.5))
    assert spot.max() < 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_bad_lamp_and_spot_values_are_rejected(device):
    renderer, _ = floor_scene(device)
    for bad in (ng.Light((0, 3, -8), falloff=-1.0), ng.Light((0, 3, -8), spot_angle=3.2, direction=(0, -1, 0)),
                ng.Light((0, 3, -8), spot_angle=0.5, direction=(0, 0, 0)), ng.Light((0, 3, -8), spot_angle=0.5, direction=(0, -1, 0), spot_softness=1.5)):
        with pytest.raises(ValueError):
            renderer.set_lights([bad])


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_on_lamps_and_spots():
    light = ng.Light((1, 3, -8), intensity=0.9, falloff=4.0, direction=(0, -1, -0.2), spot_angle=0.5, spot_softness=0.3)
    images = []
    for device in ("cpu", "cuda"):
        renderer, camera = floor_scene(device)
        renderer.set_lights([light])
        images.append(renderer.render(camera, 0).astype(float))
    assert np.abs(images[0] - images[1]).mean() < 0.6


@pytest.mark.parametrize("device", DEVICES)
def test_path_tracing_honours_falloff_and_the_cone(device):
    sky = ng.procedural_sky(sun_elevation=0.5, sun_strength=0.0, zenith_color=(0, 0, 0), horizon_color=(0, 0, 0), ground_color=(0, 0, 0), width=16, height=8)

    def mean_added(light):
        renderer, camera = floor_scene(device)
        renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
        renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False))
        renderer.set_lights([light])
        return np.mean([renderer.render(camera, f).astype(float).mean() for f in range(6)])

    plain = mean_added(ng.Light((0, 3, -8), intensity=0.8))
    lamp = mean_added(ng.Light((0, 3, -8), intensity=0.8, falloff=1.5))
    spot = mean_added(ng.Light((0, 3, -8), intensity=0.8, direction=(0, -1, 0), spot_angle=0.3))
    assert lamp < 0.8 * plain and spot < 0.6 * plain


# ---------------------------------------------------------------------------------------------------------- colour --


def rgb_added(renderer, camera, light):
    renderer.set_lights([])
    dark = renderer.render(camera, 0).astype(float)
    renderer.set_lights([light])
    return np.maximum(renderer.render(camera, 0).astype(float) - dark, 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_a_white_light_is_what_it_was_and_a_coloured_one_tints_the_floor(device):
    renderer, camera = floor_scene(device)
    plain = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8))
    white = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, color=(1, 1, 1)))
    assert np.array_equal(plain, white)
    red = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, color=(1.0, 0.1, 0.0)))
    r, g, b = (linear(red[..., c].max()) for c in range(3))
    assert r > 8 * g and b < 0.001                                # a (1, 0.1, 0) light on a grey floor gives red, ten to one in linear light
    warm = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, color=(1.0, 0.7, 0.4)))
    assert warm[..., 0].max() > warm[..., 1].max() > warm[..., 2].max() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_colour_scales_the_intensity(device):
    renderer, camera = floor_scene(device)
    half = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.8, color=(0.5, 0.5, 0.5)))
    full = rgb_added(renderer, camera, ng.Light((0, 3, -8), intensity=0.4, color=(1.0, 1.0, 1.0)))
    assert np.abs(half - full).max() <= 1.0                       # intensity 0.8 x colour 0.5 is intensity 0.4 x colour 1


@pytest.mark.parametrize("device", DEVICES)
def test_a_bad_colour_is_rejected(device):
    renderer, _ = floor_scene(device)
    for bad in ((-0.1, 1, 1), (float("nan"), 1, 1)):
        with pytest.raises(ValueError):
            renderer.set_lights([ng.Light((0, 3, -8), color=bad)])


@pytest.mark.parametrize("device", DEVICES)
def test_path_tracing_uses_the_colour_too(device):
    sky = ng.procedural_sky(sun_elevation=0.5, sun_strength=0.0, zenith_color=(0, 0, 0), horizon_color=(0, 0, 0), ground_color=(0, 0, 0), width=16, height=8)
    renderer, camera = floor_scene(device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False))
    renderer.set_lights([ng.Light((0, 3, -8), intensity=0.8, color=(1.0, 0.2, 0.05))])
    image = np.mean([renderer.render(camera, f).astype(float) for f in range(6)], axis=0)
    assert image[..., 0].mean() > 3 * image[..., 2].mean()


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_on_coloured_lights():
    light = ng.Light((1, 3, -8), intensity=0.9, falloff=4.0, color=(1.0, 0.6, 0.3))
    images = []
    for device in ("cpu", "cuda"):
        renderer, camera = floor_scene(device)
        renderer.set_lights([light])
        images.append(renderer.render(camera, 0).astype(float))
    assert np.abs(images[0] - images[1]).mean() < 0.6
