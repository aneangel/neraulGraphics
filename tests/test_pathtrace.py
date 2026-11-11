"""Tests for path tracing from Python: the settings, the sky, the helpers that make one, and the picture itself.
Runs on whichever device is available."""

import math
import struct

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 48, 27
FOV = 1.2


def scene(material=None, sky=None, **kwargs):
    material = material or ng.Material(diffuse_color=(0.8, 0.4, 0.2))
    renderer = ng.Renderer(W, H, [ng.Sphere((0, -0.5, -6), 1.5, material)], [ng.Light((-6, 9, 3), 1.2, 1.5)],
                           floor=ng.Plane(y=-2, half_extent_x=30, z_min=-60, z_max=5), **kwargs)
    camera = ng.make_camera((0, 0.5, 2), (0, -0.5, -6), width=W, height=H, fov=FOV)
    if sky is not None:
        renderer.set_environment(sky)
    return renderer, camera


def settle(renderer, camera, frames=8):
    image = None
    for frame in range(frames):
        image = renderer.render(camera, frame)
    return image.astype(np.float32)


def test_the_classic_renderer_is_the_default():
    renderer, _ = scene()
    assert renderer.render_mode == ng.RenderMode.Classic


def test_material_physical_parameters():
    m = ng.Material(roughness=0.3, metallic=1.0, emission=(1.0, 2.0, 3.0))
    assert m.roughness == pytest.approx(0.3) and m.metallic == 1.0 and m.emission == (1.0, 2.0, 3.0)
    default = ng.Material()
    assert default.roughness < 0 and default.metallic < 0 and default.emission == (0.0, 0.0, 0.0)  # worked out from the classic fields
    default.emission = (0.5, 0.5, 0.5)
    assert default.emission == (0.5, 0.5, 0.5)
    assert "roughness=0.3" in repr(m) and "emission=(1, 2, 3)" in repr(m)


def test_path_tracing_settings_round_trip():
    p = ng.PathTracing(bounces=2, exposure=2.0, tonemap=False, denoise_iterations=3)
    assert (p.bounces, p.exposure, p.tonemap, p.denoise_iterations) == (2, 2.0, False, 3)
    assert p.denoise and p.max_radiance == 20.0 and p.temporal_blend == pytest.approx(0.05)
    p.bounces = 7
    assert p.bounces == 7


def test_path_traced_picture_differs_from_classic_and_is_sensible():
    renderer, camera = scene()
    classic = renderer.render(camera).astype(np.float32)
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=3))
    assert renderer.render_mode == ng.RenderMode.PathTraced
    beauty = settle(renderer, camera)
    assert beauty.shape == (H, W, 3) and np.isfinite(beauty).all()
    assert beauty.mean() > 20 and beauty.mean() < 235  # neither black nor blown out
    assert np.abs(beauty - classic).mean() > 1.0  # it is a different picture
    renderer.set_render_mode(ng.RenderMode.Classic)  # and the way back gives the classic picture again
    assert np.array_equal(renderer.render(camera), classic.astype(np.uint8))


def test_bad_settings_are_rejected_and_change_nothing():
    renderer, _ = scene()
    for bad in (ng.PathTracing(bounces=-1), ng.PathTracing(bounces=33), ng.PathTracing(exposure=0.0),
                ng.PathTracing(denoise_iterations=9), ng.PathTracing(temporal_blend=0.0)):
        with pytest.raises(ValueError):
            renderer.set_render_mode(ng.RenderMode.PathTraced, bad)
    assert renderer.render_mode == ng.RenderMode.Classic


def test_an_emissive_surface_shows_its_emission():
    glow = ng.Material(diffuse_color=(0, 0, 0), diffuse_albedo=0.0, specular_albedo=0.0, emission=(0.5, 0.4, 0.2))
    renderer, camera = scene(glow)
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=0, tonemap=False, denoise=False))
    image = settle(renderer, camera, 4)
    centre = image[H // 2 - 2 : H // 2 + 2, W // 2 - 2 : W // 2 + 2].reshape(-1, 3).mean(axis=0)
    # sRGB of the linear emission; no tone mapping or exposure change
    expected = 255.0 * np.array([0.5, 0.4, 0.2]) ** (1 / 2.2)
    assert np.allclose(centre, expected, atol=6.0)


def test_environment_validation():
    renderer, _ = scene()
    with pytest.raises(ValueError):
        renderer.set_environment(ng.Environment(up=(0, 0, 0)))
    with pytest.raises(ValueError):
        renderer.set_environment(ng.Environment(map=np.zeros((4, 8, 3), np.float32) - 1.0))
    with pytest.raises(ValueError):
        renderer.set_environment(ng.Environment(intensity=-1.0))
    with pytest.raises(TypeError):
        ng.Environment(map=np.zeros((4, 8, 3), np.uint8))  # a map is float radiance, not 8-bit


def test_sky_light_lights_a_scene_with_no_lights():
    renderer = ng.Renderer(W, H, [ng.Sphere((0, -0.5, -6), 1.5, ng.Material(diffuse_color=(0.7, 0.7, 0.7)))], [],
                           floor=ng.Plane(y=-2, half_extent_x=30, z_min=-60, z_max=5))
    camera = ng.make_camera((0, 0.5, 2), (0, -0.5, -6), width=W, height=H, fov=FOV)
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=2, denoise=False))
    renderer.set_environment(ng.Environment(color_up=(1, 1, 1), color_down=(1, 1, 1)))
    lit = settle(renderer, camera)
    renderer.set_environment(ng.Environment(color_up=(0, 0, 0), color_down=(0, 0, 0)))
    dark = settle(renderer, camera)
    assert lit.mean() > 60 and dark.mean() < 3


def test_up_axis_turns_the_sky():
    # Bright above and black below, in a scene with no lights whose floor faces +y: the floor is lit when the sky's up is +y, and dark when it is -y.
    renderer = ng.Renderer(W, H, [], [], floor=ng.Plane(y=-2, half_extent_x=30, z_min=-60, z_max=5, checker_scale=100.0))
    camera = ng.make_camera((0, 0.5, 2), (0, -0.5, -6), width=W, height=H, fov=FOV)
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False))
    renderer.set_environment(ng.Environment(up=(0, 1, 0), color_up=(1, 1, 1), color_down=(0, 0, 0)))
    right_way_up = (settle(renderer, camera, 4)[H * 2 // 3 :] / 255.0).mean() ** 2.2  # roughly linear light
    renderer.set_environment(ng.Environment(up=(0, -1, 0), color_up=(1, 1, 1), color_down=(0, 0, 0)))
    upside_down = (settle(renderer, camera, 4)[H * 2 // 3 :] / 255.0).mean() ** 2.2
    assert right_way_up > 0.1 and upside_down < 0.3 * right_way_up


def camera_matrix(cam):
    return np.array([[cam.right[0], cam.up[0], -cam.forward[0], cam.position[0]],
                       [cam.right[1], cam.up[1], -cam.forward[1], cam.position[1]],
                       [cam.right[2], cam.up[2], -cam.forward[2], cam.position[2]]], dtype=np.float32)


def test_batch_renderer_accepts_an_environment():
    batch = ng.BatchRenderer(W, H, 2, 1, spheres=[], lights=[ng.Light((0, 5, 0), 1.0)],
                             floor=ng.Plane(y=-1, half_extent_x=20, z_min=-20, z_max=5),
                             meshes=[ng.make_box((1, 1, 1))], instances=[ng.Instance(0)])
    batch.set_environment(ng.Environment(color_up=(0.2, 0.2, 0.9), color_down=(0.2, 0.2, 0.9)))
    out = batch.render(np.tile(camera_matrix(ng.make_camera((0, 0, 3), (0, 0, 0), width=W, height=H, fov=FOV)), (2, 1, 1)), FOV)
    assert out["rgb"].shape == (2, H, W, 3)
    # Looking up at the sky, its blue shows.
    sky = batch.render(np.tile(camera_matrix(ng.make_camera((0, 0, 3), (0, 1, 0.5), width=W, height=H, fov=0.3)), (2, 1, 1)), 0.3)["rgb"]
    assert sky[..., 2].mean() > 1.5 * sky[..., 0].mean()


# ----------------------------------------------------------------------------------------------- the sky helpers --


def write_hdr(path, pixels, rle):
    """A Radiance file of the given (H, W, 3) float array, in the flat or the run-length format."""
    h, w, _ = pixels.shape
    rgbe = np.zeros((h, w, 4), np.uint8)
    peak = pixels.max(axis=2)
    for y in range(h):
        for x in range(w):
            if peak[y, x] > 1e-32:
                mantissa, exponent = math.frexp(peak[y, x])
                rgbe[y, x, :3] = (pixels[y, x] * (mantissa * 256.0 / peak[y, x])).astype(np.uint8)
                rgbe[y, x, 3] = exponent + 128
    body = bytearray()
    for y in range(h):
        if not rle:
            body += rgbe[y].tobytes()
            continue
        body += bytes([2, 2, w >> 8, w & 255])
        for c in range(4):
            x = 0
            while x < w:  # runs of 3 or more are coded as runs, the rest as literals of up to 128
                run = 1
                while x + run < w and run < 127 and rgbe[y, x + run, c] == rgbe[y, x, c]:
                    run += 1
                if run >= 3:
                    body += bytes([128 + run, rgbe[y, x, c]])
                    x += run
                else:
                    end = min(x + 1, w)
                    body += bytes([end - x]) + rgbe[y, x:end, c].tobytes()
                    x = end
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode() + bytes(body))


@pytest.mark.parametrize("rle", [False, True])
def test_load_hdr_reads_radiance_files(tmp_path, rle):
    rng = np.random.default_rng(1)
    w = 16  # RLE needs a width of 8 or more
    pixels = (rng.random((6, w, 3)) * 40).astype(np.float32)
    pixels[2] = 0.5  # a run of one repeated value
    pixels[3, 4] = 0.0
    write_hdr(tmp_path / "sky.hdr", pixels, rle)
    loaded = ng.load_hdr(tmp_path / "sky.hdr")
    assert loaded.shape == pixels.shape and loaded.dtype == np.float32
    # RGBE shares one exponent between the channels, so each is right to about 1 part in 128 of the pixel's brightest
    assert (np.abs(loaded - pixels) <= pixels.max(axis=2, keepdims=True) / 120.0 + 1e-6).all()


def test_load_hdr_rejects_other_files(tmp_path):
    (tmp_path / "a.hdr").write_bytes(b"not a radiance file\n")
    (tmp_path / "b.hdr").write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n\0\0\0\0")
    (tmp_path / "c.hdr").write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 4\n\0\0")  # cut short
    for name in "abc":
        with pytest.raises(ValueError):
            ng.load_hdr(tmp_path / f"{name}.hdr")


def test_procedural_sky_sun_has_the_strength_asked_for():
    sky = ng.procedural_sky(sun_elevation=0.8, sun_strength=4.0, sun_radius=0.05, width=512, height=256)
    assert sky.shape == (256, 512, 3) and sky.dtype == np.float32 and np.isfinite(sky).all()
    # The sun's brightest texels: a surface facing the sun receives sun_strength from them.
    bright = sky[..., 1] > 20
    assert bright.any()
    theta = (np.arange(256) + 0.5) / 256 * math.pi
    phi = (np.arange(512) + 0.5) / 512 * 2 * math.pi
    d = np.stack([np.sin(theta)[:, None] * np.cos(phi)[None], np.cos(theta)[:, None] * np.ones((1, 512)),
                  np.sin(theta)[:, None] * np.sin(phi)[None]], -1)
    sun = np.array([math.cos(0.8), math.sin(0.8), 0.0])
    texel = (2 * math.pi / 512) * (math.pi / 256) * np.sin(theta)[:, None] * np.ones((1, 512))
    irradiance = float((sky[..., 1] * texel * np.clip(d @ sun, 0, None))[bright].sum())
    assert irradiance == pytest.approx(4.0, rel=1e-3)
    assert sky[0, 0].max() < 2.0 and sky[-1, 0].max() < 2.0  # the poles are plain sky and ground


def test_procedural_sky_validation():
    for bad in (dict(sun_elevation=-0.1), dict(sun_elevation=2.0), dict(sun_radius=0.0), dict(sun_strength=-1.0), dict(width=4)):
        with pytest.raises(ValueError):
            ng.procedural_sky(**bad)


def test_a_sunny_sky_casts_a_shadow():
    sky = ng.Environment(map=ng.procedural_sky(sun_elevation=1.0, sun_strength=3.0, sun_radius=0.06))
    renderer = ng.Renderer(W, H, [ng.Sphere((0, -0.5, -6), 1.5, ng.Material(diffuse_color=(0.8, 0.8, 0.8)))], [],
                           floor=ng.Plane(y=-2, half_extent_x=30, z_min=-60, z_max=5, checker_scale=100.0))
    camera = ng.make_camera((0, 3, 2), (0, -1, -6), width=W, height=H, fov=FOV)
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=2))
    renderer.set_environment(sky)
    image = settle(renderer, camera, 16).mean(axis=2)
    floor_rows = image[H * 2 // 3 :]
    assert floor_rows.max() - floor_rows.min() > 40  # lit floor and the sphere's shadow on it


def test_simulator_adapters_switch_to_beauty_mode_with_their_up_axis():
    from neraulgraphics._adapter import SimRenderer

    class Scene:
        meshes = [ng.make_box((1, 1, 1))]
        instances = [ng.Instance(0)]
        textures = []

    # A Z-up simulator's scene: the box sits at the origin and the sky must be bright along +z.
    sim = SimRenderer(W, H, Scene(), up=(0, 0, 1), center=(0, 0, 0), extent=2.0, lights=[], fov=FOV)
    sim.beauty(denoise=False, bounces=1)
    assert sim.renderer.render_mode == ng.RenderMode.PathTraced
    top = ng.make_camera((0, -0.01, 5), (0, 0, 0), width=W, height=H, fov=FOV, world_up=(0, 1, 0))
    from_above = settle(sim.renderer, top, 4)
    sim.beauty(ng.procedural_sky(), denoise=False, bounces=1)  # an array is turned to the up axis too
    assert settle(sim.renderer, top, 4).shape == from_above.shape
    sim.classic()
    assert sim.renderer.render_mode == ng.RenderMode.Classic
    # The box's top face looks up into a bright sky: with no lights, it is lit only because up is +z.
    assert (from_above[H // 2 - 2 : H // 2 + 2, W // 2 - 2 : W // 2 + 2] / 255.0).mean() ** 2.2 > 0.08
