"""Normal maps and occlusion/roughness/metal maps. A quad faces the camera with uvs chosen so that the texture's +x is the screen's right and
the top of the image is the top of the screen, so what a map should do to the lighting is known from the geometry. Every test runs on the CPU
and, when there is one, the GPU."""

import numpy as np
import pytest

import neraulgraphics as ng

DEVICES = ["cpu"] + (["cuda"] if ng.cuda_available() else [])
W, H = 64, 64


def quad():
    """A 4 x 4 square at z = -5 facing the camera (+z), uvs: u grows to the right, v down the screen (the top of the image is v = 0)."""
    vertices = np.array([[-2, 2, -5], [2, 2, -5], [2, -2, -5], [-2, -2, -5]], np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    normals = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return ng.Mesh(vertices, triangles, normals, uvs)


def solid(rgb, size=4):
    return np.tile(np.array(rgb, np.uint8), (size, size, 1))


def render(device, light, texture_args, material=None, frames=1, mode_classic=True):
    textures, kwargs = texture_args
    instance = ng.Instance(0, material=material or ng.Material(diffuse_color=(1, 1, 1), diffuse_albedo=0.9, specular_albedo=0.0), **kwargs)
    renderer = ng.Renderer(W, H, [], [ng.Light(light, intensity=1.5)], meshes=[quad()], instances=[instance], textures=textures, device=device)
    renderer.ambient = 0.0
    camera = ng.make_camera((0, 0, 0), (0, 0, -5), width=W, height=H, fov=0.6)
    for f in range(frames):
        image = renderer.render(camera, f)
    return image.astype(float)


def tilted(nx, ny):
    """A tangent-space normal map of one constant normal, as an image: (nx, ny, sqrt(1 - nx^2 - ny^2)) mapped to 0..255."""
    n = np.array([nx, ny, np.sqrt(max(1 - nx * nx - ny * ny, 0.0))])
    return solid(np.round((n * 0.5 + 0.5) * 255).astype(np.uint8))


def centre(image):
    return image[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4].mean()


@pytest.mark.parametrize("device", DEVICES)
def test_a_flat_normal_map_changes_nothing(device):
    flat = ([ng.Texture(solid((128, 128, 255)), linear=True)], {"normal_texture": 0})
    plain = ([], {})
    a, b = render(device, (6, 0, 0), flat), render(device, (6, 0, 0), plain)
    assert np.abs(a - b).mean() < 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_a_normal_map_tilted_towards_the_light_brightens_and_away_darkens(device):
    tilt_right = ([ng.Texture(tilted(0.6, 0.0), linear=True)], {"normal_texture": 0})
    right, left = render(device, (6, 0, -2), tilt_right), render(device, (-6, 0, -2), tilt_right)
    flat = render(device, (6, 0, -2), ([], {}))
    assert centre(right) > centre(flat) * 1.15                    # the normal leans to +x, towards a light on the right
    assert centre(left) < centre(flat) * 0.6                      # and away from one on the left


@pytest.mark.parametrize("device", DEVICES)
def test_green_is_up_the_image(device):
    tilt_up = ([ng.Texture(tilted(0.0, 0.6), linear=True)], {"normal_texture": 0})
    above, below = render(device, (0, 6, -2), tilt_up), render(device, (0, -6, -2), tilt_up)
    assert centre(above) > centre(below) * 1.8                    # +y of the map is up the screen: a light above it lights it


@pytest.mark.parametrize("device", DEVICES)
def test_a_data_texture_is_not_srgb_decoded(device):
    """A grey of 128 as colour is 0.216 in linear light; as data it is 0.5, so a linear occlusion map of 128 halves the colour and the same
    image as colour would darken it far more."""
    base = render(device, (0, 0, 3), ([ng.Texture(solid((255, 255, 255)), linear=True)], {"orm_texture": 0}))
    data = render(device, (0, 0, 3), ([ng.Texture(solid((128, 255, 255)), linear=True)], {"orm_texture": 0}))
    ratio = centre(data) / centre(base)
    assert 0.55 < ratio < 0.85                                    # roughly half the light, after the display's gamma


@pytest.mark.parametrize("device", DEVICES)
def test_occlusion_darkens_where_the_map_is_dark(device):
    orm = np.full((8, 8, 3), 255, np.uint8)
    orm[:, 4:, 0] = 40                                            # the right half of the map is occluded
    image = render(device, (0, 0, 3), ([ng.Texture(orm, linear=True)], {"orm_texture": 0}))
    assert image[H // 2 - 4:H // 2 + 4, 40:56].mean() < 0.5 * image[H // 2 - 4:H // 2 + 4, 8:24].mean()


@pytest.mark.parametrize("device", DEVICES)
def test_roughness_and_metal_maps_reach_the_path_tracer(device):
    """A mirror-smooth metal and a rough dielectric look different under path tracing; the map alone must be able to make that difference."""
    sky = ng.procedural_sky(sun_elevation=0.6, sun_strength=8.0, zenith_color=(0.2, 0.4, 0.9), horizon_color=(0.8, 0.8, 0.9),
                            ground_color=(0.2, 0.2, 0.2), width=64, height=32)

    def traced(orm_value):
        instance = ng.Instance(0, material=ng.Material(diffuse_color=(0.8, 0.5, 0.3), roughness=1.0, metallic=1.0), orm_texture=0)
        renderer = ng.Renderer(W, H, [], [], meshes=[quad()], instances=[instance], textures=[ng.Texture(solid(orm_value), linear=True)], device=device)
        renderer.set_environment(ng.Environment(up=(0, 1, 0), map=sky))
        renderer.set_render_mode(ng.RenderMode.PathTraced, ng.PathTracing(bounces=2, denoise=False))
        camera = ng.make_camera((0, 0, 0), (0, 0, -5), width=W, height=H, fov=0.6)
        out = [renderer.render(camera, f).astype(float) for f in range(24)]
        return np.mean(out, axis=0)

    rough_dielectric = traced((255, 255, 0))                      # roughness 1 x 1, metal 0 x 1 = 0
    smooth_metal = traced((255, 20, 255))                         # roughness 0.08, metal 1
    assert np.abs(rough_dielectric - smooth_metal).mean() > 6.0


@pytest.mark.skipif(not ng.cuda_available(), reason="needs a GPU")
def test_cpu_and_gpu_agree_with_maps():
    maps = ([ng.Texture(tilted(0.5, 0.3), linear=True), ng.Texture(np.random.default_rng(0).integers(60, 255, (16, 16, 3), dtype=np.uint8), linear=True)],
            {"normal_texture": 0, "orm_texture": 1})
    a, b = render("cpu", (5, 3, -1), maps), render("cuda", (5, 3, -1), maps)
    assert np.abs(a - b).mean() < 0.5


def test_bad_map_indices_are_rejected():
    with pytest.raises(ValueError):
        ng.Renderer(W, H, [], [], meshes=[quad()], instances=[ng.Instance(0, normal_texture=3)], device="cpu")
    with pytest.raises(ValueError):
        ng.Renderer(W, H, [], [], meshes=[ng.make_box((1, 1, 1))], instances=[ng.Instance(0, orm_texture=0)], textures=[ng.Texture(solid((9, 9, 9)))], device="cpu")
