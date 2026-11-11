"""Smooth glass (Material.transmission, ior): a dielectric that reflects by the Fresnel equations and refracts by Snell's law. Each test is a
situation with a known answer: a clear ball in a uniform environment shows the environment (nothing is absorbed); a ball lens inverts what is
behind it; an index of 1 is invisible; tinted glass absorbs with the distance travelled."""

import numpy as np
import pytest

import neraulgraphics as ng

DEVICES = ["cpu"] + (["cuda"] if ng.cuda_available() else [])
E = 0.5


def to_linear(v):
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def glass(color=(1, 1, 1), ior=1.5, transmission=1.0):
    return ng.Material(diffuse_color=color, roughness=0.0, metallic=0.0, transmission=transmission, ior=ior)


def trace(renderer, camera, frames, bounces):
    renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=bounces, denoise=False, tonemap=False, exposure=1.0, max_radiance=0.0))
    return sum(to_linear(renderer.render(camera, f).astype(float)) for f in range(frames)) / frames


def furnace_ball(material, device, size=32, frames=32, bounces=10):
    renderer = ng.Renderer(size, size, [ng.Sphere((0, 0, -4), 1.0, material)], [], device=device)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), color_up=(1, 1, 1), color_down=(1, 1, 1), intensity=E))
    camera = ng.make_camera((0, 0, 0), (0, 0, -4), width=size, height=size, fov=0.3)
    return trace(renderer, camera, frames, bounces)[12:20, 12:20].mean(axis=(0, 1)) / E


def wall(x0, x1, colour, z=-14.0):
    """An emissive quad in the plane z, from x0 to x1 and 10 m up and down."""
    vertices = np.array([[x0, 5, z], [x1, 5, z], [x1, -5, z], [x0, -5, z]], np.float32)
    return ng.Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.uint32))


def lens_scene(ior, device, size=64):
    red, blue = ng.Material(diffuse_color=(0, 0, 0), emission=(1.0, 0.0, 0.0)), ng.Material(diffuse_color=(0, 0, 0), emission=(0.0, 0.0, 1.0))
    meshes = [wall(-30, 0, None), wall(0, 30, None)]
    instances = [ng.Instance(0, material=red), ng.Instance(1, material=blue)]
    renderer = ng.Renderer(size, size, [ng.Sphere((0, 0, -5), 1.0, glass(ior=ior))], [], meshes=meshes, instances=instances, device=device)
    camera = ng.make_camera((0, 0, 0), (0, 0, -5), width=size, height=size, fov=0.9)      # the ball covers about half the frame
    return trace(renderer, camera, 24, 10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ior", [1.33, 1.5, 2.4])
def test_a_clear_ball_shows_the_environment_and_absorbs_nothing(device, ior):
    rgb = furnace_ball(glass(ior=ior), device)
    assert rgb == pytest.approx([1.0, 1.0, 1.0], abs=0.04), f"ior {ior}: {rgb}"


@pytest.mark.parametrize("device", DEVICES)
def test_a_ball_lens_inverts_what_is_behind_it(device):
    image = lens_scene(1.5, device)
    h = image.shape[0] // 2
    left = image[h - 4:h + 4, 24:30].mean(axis=(0, 1))                  # just left of the middle of the ball
    right = image[h - 4:h + 4, 34:40].mean(axis=(0, 1))
    assert left[2] > 3 * left[0]                                        # left of centre it shows the blue wall that is on the right
    assert right[0] > 3 * right[2]                                      # and right of centre the red one
    outside = image[h, 3:8]
    assert outside[:, 0].mean() > 3 * outside[:, 2].mean()              # away from the ball the red wall is on the left, as it should be


@pytest.mark.parametrize("device", DEVICES)
def test_an_index_of_one_is_invisible(device):
    image = lens_scene(1.0, device)
    h = image.shape[0] // 2
    left = image[h - 4:h + 4, 24:30].mean(axis=(0, 1))
    right = image[h - 4:h + 4, 34:40].mean(axis=(0, 1))
    assert left[0] > 3 * left[2] and right[2] > 3 * right[0]            # nothing bends: red on the left, blue on the right, through the ball


@pytest.mark.parametrize("device", DEVICES)
def test_tinted_glass_absorbs_with_distance(device):
    rgb = furnace_ball(glass(color=(0.3, 0.95, 0.5)), device)
    assert rgb[1] > rgb[2] > rgb[0]                                     # green glass: most green, some blue, little red
    assert rgb[1] <= 1.03 and rgb[0] < 0.75                             # and it never makes light
    clear = furnace_ball(glass(), device)
    assert rgb.mean() < clear.mean()


@pytest.mark.parametrize("device", DEVICES)
def test_half_glass_is_between_glass_and_the_opaque_material(device):
    opaque = ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0.4, specular_albedo=0.0, reflectivity=0.0)
    half = ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0.4, specular_albedo=0.0, reflectivity=0.0, roughness=1.0, metallic=0.0, transmission=0.5)
    assert furnace_ball(opaque, device).mean() == pytest.approx(0.4, abs=0.03)
    assert 0.5 < furnace_ball(half, device).mean() < 1.03              # a mixture: brighter than the opaque ball, no brighter than clear


def test_bad_glass_values_are_rejected():
    for bad in (ng.Material(transmission=1.5), ng.Material(transmission=-0.1), ng.Material(ior=0.5), ng.Material(ior=9.0)):
        with pytest.raises(ValueError):
            ng.Renderer(8, 8, [ng.Sphere((0, 0, -4), 1.0, bad)], [], device="cpu")


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_on_glass():
    a, b = lens_scene(1.5, "cpu"), lens_scene(1.5, "cuda")
    assert np.abs(a - b).mean() < 0.03


# ----------------------------------------------------------------------------------------------------------- shadows --


def shadow_scene(material, device):
    """A ball (or none) lit from the side by a light 28 degrees above the horizon, seen from above, so that its shadow lies on the floor."""
    size = 64
    floor = ng.Plane(y=0.0, half_extent_x=60.0, z_min=-90.0, z_max=10.0, checker_scale=0.0001,
                     material=ng.Material(diffuse_color=(0.7, 0.7, 0.7), diffuse_albedo=0.9, specular_albedo=0.0, roughness=1.0, metallic=0.0))
    spheres = [ng.Sphere((0, 1.0, -6), 1.0, material)] if material is not None else []
    renderer = ng.Renderer(size, size, spheres, [ng.Light((-17, 10, -6), intensity=1.0, radius=0.0)], floor=floor, device=device)
    renderer.ambient = 0.0
    sky = ng.procedural_sky(sun_elevation=0.5, sun_strength=0.0, zenith_color=(0, 0, 0), horizon_color=(0, 0, 0), ground_color=(0, 0, 0), width=16, height=8)
    renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
    camera = ng.make_camera((0, 14, -3.0), (0, 0, -6.5), width=size, height=size, fov=0.9)
    return trace(renderer, camera, 24, 8)


def shadow_of(material, device):
    """What fraction of the light the floor in an opaque ball's shadow still gets when the ball is made of material."""
    bare = shadow_scene(None, device)
    opaque = shadow_scene(ng.Material(diffuse_color=(1, 1, 1), roughness=1.0, metallic=0.0), device)
    mask = (opaque.mean(axis=2) < 0.25 * bare.mean(axis=2)) & (bare.mean(axis=2) > 0.1)
    assert mask.sum() > 30                                           # there is a shadow to measure
    image = shadow_scene(material, device)
    return image[mask].mean(axis=0) / bare[mask].mean(axis=0)


@pytest.mark.parametrize("device", DEVICES)
def test_glass_casts_a_light_shadow_and_an_opaque_ball_a_dark_one(device):
    opaque = shadow_of(ng.Material(diffuse_color=(1, 1, 1), roughness=1.0, metallic=0.0), device)
    clear = shadow_of(glass(), device)
    assert opaque.mean() < 0.1                                       # an opaque ball's shadow is dark (by definition of the mask)
    assert clear.mean() > 0.55                                       # clear glass: most of the light gets through


@pytest.mark.parametrize("device", DEVICES)
def test_the_shadow_of_coloured_glass_is_coloured(device):
    shadow = shadow_of(glass(color=(0.2, 0.9, 0.3)), device)
    assert shadow[1] > 2 * shadow[0] and shadow[1] > 1.5 * shadow[2]
