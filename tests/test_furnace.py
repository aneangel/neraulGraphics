"""White-furnace tests: a convex object in an environment of uniform radiance E has an exactly known appearance, so they check that the path
tracer neither makes nor loses light. A Lambertian sphere of albedo a returns a E; a white metal (F0 = 1) returns E at any roughness, because
nothing absorbs the light; a white dielectric returns at most E. Before multiple-scattering compensation a white metal at roughness 0.6
returned 81 % of E, at 0.8 54 % and at 1.0 29 %: rough metals came out too dark."""

import numpy as np
import pytest

import neraulgraphics as ng

DEVICES = ["cpu"] + (["cuda"] if ng.cuda_available() else [])
E = 0.5


def srgb_to_linear(v):
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def furnace(material, device, frames=48, bounces=8):
    """The radiance (as a multiple of E) at the middle of a sphere of this material in a uniform environment."""
    size = 32
    renderer = ng.Renderer(size, size, [ng.Sphere((0, 0, -4), 1.0, material)], [], device=device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), color_up=(1, 1, 1), color_down=(1, 1, 1), intensity=E))
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=bounces, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
    camera = ng.make_camera((0, 0, 0), (0, 0, -4), width=size, height=size, fov=0.2)
    acc = sum(srgb_to_linear(renderer.render(camera, f).astype(float)) for f in range(frames)) / frames
    return float(acc[12:20, 12:20].mean() / E)


def lambert(albedo):
    return ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=albedo, specular_albedo=0.0, reflectivity=0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("albedo", [0.3, 0.6, 1.0])
def test_a_lambertian_sphere_returns_exactly_its_albedo(device, albedo):
    assert furnace(lambert(albedo), device) == pytest.approx(albedo, abs=0.015)


@pytest.mark.parametrize("device", DEVICES)
def test_the_output_is_srgb_encoded(device):
    """A dark sky must come out as the sRGB code (the linear toe), not the brighter code a plain 2.2 power would give."""
    renderer = ng.Renderer(8, 8, [], [], device=device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), color_up=(1, 1, 1), color_down=(1, 1, 1), intensity=0.002))
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=1, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
    code = int(renderer.render(ng.make_camera((0, 0, 0), (0, 0, -4), width=8, height=8, fov=0.2), 0)[4, 4, 0])
    assert code == 7                                                       # in the sRGB curve's linear toe (12.92 x 0.002 x 255 = 6.6); a 2.2 power gives 14


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("roughness, lowest", [(0.05, 0.98), (0.2, 0.98), (0.4, 0.98), (0.6, 0.98), (0.8, 0.95), (1.0, 0.88)])
def test_a_white_metal_neither_loses_nor_makes_light(device, roughness, lowest):
    returned = furnace(ng.Material(diffuse_color=(1, 1, 1), roughness=roughness, metallic=1.0), device)
    assert lowest <= returned <= 1.03, f"roughness {roughness}: {returned:.3f} x E"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("roughness", [0.1, 0.5, 1.0])
def test_a_white_dielectric_returns_at_most_what_it_receives(device, roughness):
    returned = furnace(ng.Material(diffuse_color=(1, 1, 1), roughness=roughness, metallic=0.0), device)
    assert 0.94 <= returned <= 1.03, f"roughness {roughness}: {returned:.3f} x E"


@pytest.mark.parametrize("device", DEVICES)
def test_a_coloured_metal_keeps_its_colour_and_gains_nothing(device):
    gold = (1.0, 0.77, 0.34)
    size = 32
    renderer = ng.Renderer(size, size, [ng.Sphere((0, 0, -4), 1.0, ng.Material(diffuse_color=gold, roughness=0.7, metallic=1.0))], [], device=device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), color_up=(1, 1, 1), color_down=(1, 1, 1), intensity=E))
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=8, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
    camera = ng.make_camera((0, 0, 0), (0, 0, -4), width=size, height=size, fov=0.2)
    rgb = (sum(srgb_to_linear(renderer.render(camera, f).astype(float)) for f in range(48)) / 48)[12:20, 12:20].mean(axis=(0, 1)) / E
    assert rgb[0] > rgb[1] > rgb[2]                                       # still gold
    assert all(rgb[c] <= gold[c] * 1.03 for c in range(3))                # and never brighter than the metal's colour allows


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_on_the_furnace():
    m = ng.Material(diffuse_color=(1, 1, 1), roughness=0.7, metallic=1.0)
    assert furnace(m, "cpu") == pytest.approx(furnace(m, "cuda"), abs=0.02)
