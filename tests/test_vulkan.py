"""The Vulkan ray-query prototype against the CPU reference. Skipped unless the tool is built (cmake -DNERAULGRAPHICS_BUILD_VULKAN=ON)
and its location is given in NERAULGRAPHICS_VK_BUILD."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

build = os.environ.get("NERAULGRAPHICS_VK_BUILD")
pytestmark = pytest.mark.skipif(not build or not (Path(build) / "vk_rayquery").exists(), reason="set NERAULGRAPHICS_VK_BUILD to a build with NERAULGRAPHICS_BUILD_VULKAN=ON")


def test_every_vulkan_device_agrees_with_the_cpu_reference():
    script = Path(__file__).resolve().parent.parent / "tools" / "vulkan" / "compare.py"
    run = subprocess.run([sys.executable, str(script), "--tool", str(Path(build) / "vk_rayquery"), "--spv", str(Path(build) / "rayquery.spv"), "--rays", "200000"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
