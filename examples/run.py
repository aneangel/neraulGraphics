"""Lists the examples and runs one.

    python examples/run.py                      # what there is
    python examples/run.py materials            # run one (a window opens)
    python examples/run.py materials --save a.png     # anything after the name goes to the example

Every example opens a live window when there is a display, and writes a picture (or a GIF) when there is not.
"""

import importlib.util
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name: (script, what it shows, modules it needs besides neraulgraphics and numpy)
EXAMPLES = {
    "materials": ("materials.py", "path-traced glass, metal and plastic under a physical sky; throw balls and boxes into it (Jolt physics)", ["pyglet", "PIL"]),
    "view_model": ("view_model.py", "orbit a glTF/OBJ model or the built-in torus knot, path traced; drop it and watch it tumble", ["pyglet", "PIL"]),
    "sensors": ("sensor_images.py", "RGB, depth, normals, object IDs and LiDAR of a scene that physics keeps rearranging", ["pyglet", "PIL"]),
    "newton": ("newton_shapes.py", "Newton's GPU rigid-body solver, path traced: drop shapes onto the ground", ["pyglet", "newton", "warp", "PIL"]),
    "mujoco": ("mujoco_humanoid.py", "a MuJoCo humanoid ragdoll, path traced: throw balls at it, switch its motors on", ["pyglet", "mujoco", "PIL"]),
    "batch": ("batch_render.py", "thousands of simulated worlds rendered at once, on the GPU: an RL environment loop", ["pyglet", "PIL"]),
}


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "list"):
        print(__doc__)
        for name, (script, about, needs) in EXAMPLES.items():
            missing = [m for m in needs if not available(m)]
            note = f"   (needs {', '.join(missing)})" if missing else ""
            print(f"  {name:12s} {about}{note}")
        return
    name = sys.argv[1]
    if name not in EXAMPLES:
        sys.exit(f"no example called {name!r}; run without arguments for the list")
    script, _, needs = EXAMPLES[name]
    missing = [m for m in needs if not available(m)]
    if missing:
        sys.exit(f"{name} needs {', '.join(missing)}: pip install {' '.join('pillow' if m == 'PIL' else m for m in missing)}")
    path = HERE / script
    sys.argv = [str(path)] + sys.argv[2:]
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
