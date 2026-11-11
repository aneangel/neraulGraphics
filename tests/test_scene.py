"""Tests for the street scene and its procedural textures (examples/scenery): the textures tile without seams, the box meshes carry
uvs in real-world units, and a street is laid out around a structure without anything standing on its centre line."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "scenery"))

import textures  # noqa: E402
from street import street  # noqa: E402


@pytest.mark.parametrize("name", ["brick", "asphalt", "concrete", "stucco"])
def test_textures_are_seamless_images(name):
    image = textures.all_textures()[name][0]
    assert image.shape == (512, 512, 3) and image.dtype == np.uint8
    assert image.std() > 4.0                                                   # not a flat colour
    img = image.astype(float)
    # where the image wraps (last column next to the first) must be no rougher than the step between neighbouring columns inside it
    seam = np.abs(img[:, 0] - img[:, -1]).mean()
    inside = np.abs(np.diff(img, axis=1)).mean()
    assert seam < 2.5 * inside + 2.0


def test_textures_are_repeatable():
    a, b = textures.concrete(seed=3)[0], textures.concrete(seed=3)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, textures.concrete(seed=4)[0])


def test_a_textured_box_has_uvs_in_world_units():
    vertices, triangles, normals, uvs = textures.textured_box((10.0, 3.0, 0.5), tile=2.0)
    assert vertices.shape == (24, 3) and triangles.shape == (12, 3) and normals.shape == (24, 3) and uvs.shape == (24, 2)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # the +z face is 20 m wide and 6 m tall: its uvs span 10 repeats by 3
    face = uvs[8:12]
    assert np.ptp(face[:, 0]) == pytest.approx(10.0, abs=1e-4) and np.ptp(face[:, 1]) == pytest.approx(3.0, abs=1e-4)
    assert (np.abs(vertices).max(axis=0) == pytest.approx((10.0, 3.0, 0.5)))


def test_projected_uvs_follow_the_dominant_axis():
    corners = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    normals = np.array([[0, 0, 1], [1, 0, 0]], np.float32)
    uv = textures.box_projected_uvs(corners, normals, tile=2.0)
    assert uv[0] == pytest.approx([0.5, 1.0]) and uv[1] == pytest.approx([2.5, 3.0])     # z-facing uses x, y; x-facing uses y, z


class Structure:
    """A stand-in for any structure: street() reads only the centres and half-sizes of its parts (here one 12 x 8 x 10 m block)."""

    position = np.array([[0.0, 0.0, 5.0]])
    half = np.array([[6.0, 4.0, 5.0]])


@pytest.fixture(scope="module")
def props():
    structure = Structure()
    return structure, street(structure, seed=0)


def test_a_street_surrounds_the_structure(props):
    structure, items = props
    assert len(items) > 200
    names = {p.texture for p in items if p.texture}
    assert {"brick", "asphalt", "concrete"} <= names and names <= set(textures.all_textures())
    assert any(p.emission[0] > 1.0 for p in items)                              # lit windows and lamps
    lo, hi = (structure.position - structure.half).min(axis=0), (structure.position + structure.half).max(axis=0)
    shells = [p for p in items if p.texture in ("brick", "stucco") and p.half[2] > 4.0]
    for p in shells:                                                            # no building stands inside the structure's footprint
        x0, x1 = p.position[0] - p.half[0], p.position[0] + p.half[0]
        y0, y1 = p.position[1] - p.half[1], p.position[1] + p.half[1]
        assert x1 <= lo[0] + 0.01 or x0 >= hi[0] - 0.01 or y1 <= lo[1] + 0.01 or y0 >= hi[1] - 0.01


def test_nothing_stands_on_the_centre_line(props):
    """A path is kept clear along x = the structure's centre, from the far kerb to the sidewalk: nothing taller than a kerb may be
    there, thin or not (a lamp post counts)."""
    structure, items = props
    cx = (structure.position[:, 0].min() + structure.position[:, 0].max()) / 2
    front = (structure.position - structure.half).min(axis=0)[1]
    for p in items:
        if p.position[2] + p.half[2] <= 0.25:
            continue                                                            # flat ground detail: asphalt, markings, kerb
        across = abs(p.position[0] - cx) < p.half[0] + 0.7
        along = p.position[1] + p.half[1] > front - 24.0 and p.position[1] - p.half[1] < front - 4.5
        assert not (across and along), f"a prop stands on the path: {p}"


@pytest.mark.parametrize("name", ["brick", "asphalt", "concrete", "stucco"])
def test_every_material_has_a_unit_normal_map_and_a_plausible_orm_map(name):
    colour, normal, orm = textures.all_textures()[name]
    assert normal.shape == colour.shape == orm.shape
    n = normal.astype(float) / 255.0 * 2.0 - 1.0
    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 0.03                # 8-bit rounding only
    assert (n[..., 2] > 0.5).all()                                              # it bends the normal, it does not flip it
    assert n[..., :2].std() > 0.01                                              # and it does bend it
    assert orm[..., 1].min() >= 0 and orm[..., 2].max() == 0                    # roughness varies, nothing is metal


def test_the_normal_map_leans_away_from_a_rise():
    """A bump rising to the right (height grows with x) leans the normal to -x; one rising up the image leans it to -y."""
    h = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    n = textures.normal_map(h, 20.0).astype(float) / 255 * 2 - 1
    assert n[32, 32, 0] < -0.3 and abs(n[32, 32, 1]) < 0.05
    up = np.tile(np.linspace(1, 0, 64, dtype=np.float32)[:, None], (1, 64))     # larger at the top row: rising up the image
    n = textures.normal_map(up, 20.0).astype(float) / 255 * 2 - 1
    assert n[32, 32, 1] < -0.3 and abs(n[32, 32, 0]) < 0.05


def test_asphalt_has_wet_patches_that_are_smoother():
    _, _, orm = textures.all_textures()["asphalt"]
    rough = orm[..., 1].astype(float) / 255
    assert rough.max() > 0.85 and rough.max() - rough.min() > 0.1                  # damp areas are visibly smoother than dry ones
