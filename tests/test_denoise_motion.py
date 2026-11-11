"""The denoiser's history weight depends on whether the camera moves: a still camera accumulates deeply (a small temporal_blend), a moving one trusts
new frames more (moving_blend), because reprojected history of a moving view carries lag and misalignment that outweigh the noise it removes."""

import numpy as np
import pytest

import neraulgraphics as ng

DEVICES = ["cpu"] + (["cuda"] if ng.cuda_available() else [])
W, H = 96, 64


def scene(device, **settings):
    floor = ng.Plane(y=-1.0, half_extent_x=40.0, z_min=-60.0, z_max=5.0, checker_scale=0.0001,
                     material=ng.Material(diffuse_color=(0.7, 0.7, 0.7), diffuse_albedo=0.9, specular_albedo=0.0, roughness=1.0, metallic=0.0))
    balls = [ng.Sphere((x, 0.0, -6.0), 0.8, ng.Material(diffuse_color=c, roughness=0.6, metallic=0.0)) for x, c in ((-1.6, (0.8, 0.2, 0.2)), (1.6, (0.2, 0.3, 0.8)))]
    renderer = ng.Renderer(W, H, balls, [ng.sun_light(sun_elevation=0.8, sun_strength=3.0, sun_radius=0.03)], floor=floor, device=device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), map=ng.physical_sky(sun_elevation=0.8, sun_strength=3.0, width=256, height=128, include_sun=False)))
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=3, exposure=0.8, denoise=True, tonemap=False, **settings))
    return renderer


def lin(img):
    v = img.astype(float) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def camera(x):
    return ng.make_camera((x, 1.0, 1.0), (0.0, -0.2, -6.0), width=W, height=H, fov=0.9)


def reference(device, x, frames=200):
    r = scene(device)
    img = None
    for f in range(frames):
        img = r.render(camera(x), f)
    return lin(img)


def moving_error(device, **settings):
    """Mean RMS error against a converged picture of the same view, over a sideways move of 0.08 m a frame (about a pixel a frame: at a few
    hundredths of a pixel reprojection is nearly exact and the setting does not matter; at ordinary speeds it does)."""
    r = scene(device, **settings)
    errors = []
    positions = [0.08 * t for t in range(40)]
    refs = {i: reference(device, positions[i]) for i in (30, 39)}
    for t, x in enumerate(positions):
        img = lin(r.render(camera(x), t))
        if t in refs:
            errors.append(float(np.sqrt(((img - refs[t]) ** 2).mean())))
    return float(np.mean(errors))


@pytest.mark.parametrize("device", DEVICES)
def test_a_moving_camera_does_better_trusting_new_frames_more(device):
    slow = moving_error(device, temporal_blend=0.05, moving_blend=0.05)       # no distinction: the still camera's setting while moving
    fast = moving_error(device, temporal_blend=0.05, moving_blend=0.25)
    assert fast < 0.85 * slow, (fast, slow)                                  # measured: 0.67 on the GPU


@pytest.mark.parametrize("device", DEVICES)
def test_a_still_camera_still_accumulates_deeply(device):
    """With the camera still, moving_blend is not used: the picture converges as far as temporal_blend allows."""
    def still_error(**settings):
        r = scene(device, **settings)
        img = None
        for f in range(120):
            img = r.render(camera(0.0), f)
        return float(np.sqrt(((lin(img) - reference(device, 0.0, 240)) ** 2).mean()))

    deep = still_error(temporal_blend=0.02, moving_blend=0.5)
    shallow = still_error(temporal_blend=0.5, moving_blend=0.5)
    assert deep < 0.8 * shallow


def test_bad_blends_are_rejected():
    r = scene("cpu")
    for bad in (dict(moving_blend=0.0), dict(moving_blend=1.5), dict(temporal_blend=0.0)):
        with pytest.raises(ValueError):
            r.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(**bad))
    assert ng.PathTracing().moving_blend == pytest.approx(0.2)
