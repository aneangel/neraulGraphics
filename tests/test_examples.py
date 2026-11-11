"""Every example, run for real without a display: each must finish, and write a picture that is a picture and not a blank. They run at a small
size and refine for a few frames, so they take seconds; what they prove is that the examples still work against the library as it is now."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import neraulgraphics as ng
from neraulgraphics import physics

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def have(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


def run(script, *args, timeout=240):
    env = {**os.environ, "DISPLAY": "", "WAYLAND_DISPLAY": ""}              # no window can open: the example must fall back to a file
    return subprocess.run([sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True, env=env, timeout=timeout)


def check_picture(path, minimum_size=(60, 30)):
    from PIL import Image

    assert path.exists(), "the example wrote no picture"
    image = np.asarray(Image.open(path).convert("RGB"))
    assert image.shape[0] >= minimum_size[1] and image.shape[1] >= minimum_size[0]
    assert image.std() > 4.0, "the picture is blank"
    return image


needs_physics = pytest.mark.skipif(not physics.available(), reason="built without NERAULGRAPHICS_BUILD_PHYSICS")


@needs_physics
def test_materials(tmp_path):
    out = tmp_path / "m.png"
    result = run("materials.py", "--size", "320x180", "--frames", "6", "--save", str(out))
    assert result.returncode == 0, result.stderr[-1500:]
    image = check_picture(out)
    assert image.shape == (180, 320, 3)


@needs_physics
def test_view_model_default_and_a_loaded_model(tmp_path):
    out = tmp_path / "v.png"
    result = run("view_model.py", "--size", "320x180", "--frames", "4", "--save", str(out))
    assert result.returncode == 0, result.stderr[-1500:]
    check_picture(out)
    obj = tmp_path / "cube.obj"
    obj.write_text("v -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\nv -1 2 -1\nv 1 2 -1\nv 1 2 1\nv -1 2 1\n"
                   "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 4 8 5 1\n")
    out2 = tmp_path / "v2.png"
    result = run("view_model.py", str(obj), "--size", "320x180", "--frames", "4", "--save", str(out2))
    assert result.returncode == 0, result.stderr[-1500:]
    check_picture(out2)


@needs_physics
def test_sensor_images(tmp_path):
    out = tmp_path / "s.png"
    result = run("sensor_images.py", "--size", "160x90", "--frames", "3", "--save", str(out))
    assert result.returncode == 0, result.stderr[-1500:]
    image = check_picture(out)
    assert image.shape == (180, 480, 3)                                      # six panels: three across, two down


@pytest.mark.skipif(not have("newton", "warp"), reason="needs Newton")
def test_newton_shapes(tmp_path):
    out = tmp_path / "n.png"
    result = run("newton_shapes.py", "--size", "320x180", "--frames", "3", "--save", str(out), timeout=400)
    assert result.returncode == 0, result.stderr[-1500:]
    check_picture(out)


@pytest.mark.skipif(not have("mujoco"), reason="needs MuJoCo")
def test_mujoco_humanoid(tmp_path):
    out = tmp_path / "h.png"
    result = run("mujoco_humanoid.py", "--size", "320x180", "--frames", "3", "--save", str(out))
    assert result.returncode == 0, result.stderr[-1500:]
    check_picture(out)


@pytest.mark.skipif(not have("torch"), reason="needs PyTorch")
def test_batch_render(tmp_path):
    out = tmp_path / "b.png"
    result = run("batch_render.py", "--worlds", "32", "--size", "32", "--steps", "3", "--save", str(out))
    assert result.returncode == 0, result.stderr[-1500:]
    image = check_picture(out)
    assert image.shape == (128, 128, 3)                                      # a 4 x 4 mosaic of 32-pixel views


def test_the_launcher_lists_the_examples_and_refuses_an_unknown_one():
    listing = subprocess.run([sys.executable, str(EXAMPLES / "run.py")], capture_output=True, text=True)
    assert listing.returncode == 0
    for name in ("materials", "view_model", "sensors", "newton", "mujoco", "batch"):
        assert name in listing.stdout
    unknown = subprocess.run([sys.executable, str(EXAMPLES / "run.py"), "nothing"], capture_output=True, text=True)
    assert unknown.returncode != 0


def test_every_example_named_by_the_launcher_exists():
    sys.path.insert(0, str(EXAMPLES))
    import run as launcher

    for name, (script, about, needs) in launcher.EXAMPLES.items():
        assert (EXAMPLES / script).exists(), name
        assert about and needs
