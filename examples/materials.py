"""A materials playground: path-traced glass, metal and plastic under a physical sky, with real rigid-body physics you can throw things into.

    python examples/materials.py                      # opens the viewer
    python examples/materials.py --hdri sky.hdr       # light it with your own HDR sky
    python examples/materials.py --save still.png     # no window: throw a few things, let the picture refine, write it

What it shows: the renderer's fidelity (a path tracer with glass that refracts and tints its shadow, rough metals that keep their energy,
textured and normal-mapped concrete, a physically based sky and sun), and its physics (the same bodies are simulated by Jolt, deterministically,
and drawn as they tumble and settle).

Click the window to look around with the mouse (Esc lets go, a second Esc quits). W A S D move, Q and E go down and up, shift is faster.
1 to 8 choose a material, Space or the left button throws a ball of it from where you are looking, B or the right button throws a box,
R puts everything back, [ and ] lower and raise the sun, , and . swing it round, - and = change the exposure, X slows time,
H hides the text, P saves a picture. Hold still and the picture keeps getting cleaner.

Needs: pip install pillow pyglet, and neraulgraphics built with the physics module (pip install -C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON .).
An NVIDIA GPU makes it fast; without one it uses the CPU.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import neraulgraphics as ng

sys.path.insert(0, str(Path(__file__).resolve().parent / "scenery"))

from _live import FlyCamera, open_window, save_picture  # noqa: E402
from _stage import Body, Scene  # noqa: E402

MATERIALS = {      # name -> material; the numbers on the keyboard pick from this list, in order
    "clear glass": ng.Material(diffuse_color=(1, 1, 1), roughness=0.0, metallic=0.0, transmission=1.0, ior=1.5),
    "green glass": ng.Material(diffuse_color=(0.25, 0.85, 0.45), roughness=0.0, metallic=0.0, transmission=1.0, ior=1.45),
    "polished gold": ng.Material(diffuse_color=(1.0, 0.78, 0.34), roughness=0.08, metallic=1.0),
    "brushed steel": ng.Material(diffuse_color=(0.78, 0.80, 0.84), roughness=0.5, metallic=1.0),
    "red plastic": ng.Material(diffuse_color=(0.75, 0.08, 0.06), roughness=0.12, metallic=0.0),
    "chrome": ng.Material(diffuse_color=(0.95, 0.95, 0.97), roughness=0.02, metallic=1.0),
    "blue glass": ng.Material(diffuse_color=(0.35, 0.55, 0.95), roughness=0.0, metallic=0.0, transmission=0.6, ior=1.33),
    "matte clay": ng.Material(diffuse_color=(0.65, 0.62, 0.58), roughness=1.0, metallic=0.0),
}
NAMES = list(MATERIALS)
HEROES = [   # (material, x, depth, radius): the still life that is there when you start, as bodies that can be knocked about
    ("clear glass", -4.4, 7.0, 0.9), ("green glass", -2.2, 6.0, 0.9), ("polished gold", 0.0, 5.2, 0.9), ("brushed steel", 2.2, 6.0, 0.9),
    ("red plastic", 4.4, 7.0, 0.9), ("matte clay", -1.1, 9.5, 0.9), ("blue glass", 1.3, 9.8, 1.0),
]
THROWN_BALLS, THROWN_BOXES = 24, 10


class Playground(Scene):
    """The scene of this example: the still life of HEROES, and the throwable balls and boxes, on the shared ground and sky."""

    def __init__(self, width, height, hdri=None):
        bodies = [Body("sphere", r, MATERIALS[m], density=2400.0 if "glass" in m else 3000.0, restitution=0.2) for m, _, _, r in HEROES]
        bodies += [Body("sphere", 0.32 + 0.05 * (i % 3), MATERIALS[NAMES[i % len(NAMES)]], density=1500.0, restitution=0.3) for i in range(THROWN_BALLS)]
        bodies += [Body("box", 0.3 + 0.05 * (i % 2), MATERIALS[NAMES[(i + 4) % len(NAMES)]], density=800.0, restitution=0.1) for i in range(THROWN_BOXES)]
        super().__init__(width, height, bodies, hdri=hdri)
        self.selected, self.next_ball, self.next_box = 0, 0, 0
        self.reset()

    def reset(self):
        self.stage.park_all()
        for i, (_, x, depth, r) in enumerate(HEROES):
            self.stage.place(i, (x, depth, r))                  # simulation axes: y is forward, which is the renderer's -z
        self.stage.step(1 / 60)
        self.push()
        self.frame = 0

    def throw_ball(self, origin, direction, material=None):
        """Throws a ball of the selected (or named) material; each material has a few balls, used in turn."""
        wanted = MATERIALS[material or NAMES[self.selected]]
        slots = [i for i in range(len(HEROES), len(HEROES) + THROWN_BALLS) if self.stage.bodies[i].material is wanted]
        self.throw(slots[self.next_ball % len(slots)], origin, direction)
        self.next_ball += 1

    def throw_box(self, origin, direction):
        self.throw(len(HEROES) + THROWN_BALLS + self.next_box % THROWN_BOXES, origin, direction)
        self.next_box += 1

    def hud(self, fps):
        return (f"{fps:4.0f} fps   {self.frame} frames refining\n"
                f"selected: {self.selected + 1} {NAMES[self.selected]}     sun {math.degrees(self.sun_elevation):.0f} deg     exposure {self.exposure:.2f}\n"
                "Space throw   B box   1-8 material   [ ] sun height   , . sun turn   - = exposure   X slow   R reset   P picture   Esc quit")


def intro(play):
    """A short scripted throw, so there is something to see at once: (seconds, material, from, towards), in renderer axes."""
    return [(0.6, "chrome", (-5.5, 3.5, 1.0), (1.0, -0.2, -3.6)), (1.4, "polished gold", (5.5, 4.0, 1.0), (-1.0, -0.25, -4.0)),
            (2.2, "clear glass", (0.0, 5.0, 0.5), (0.0, -0.35, -5.0))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--hdri", help="an equirectangular Radiance .hdr sky, in place of the physical sky")
    parser.add_argument("--save", metavar="FILE", help="write a picture to FILE (a PNG); with no window, instead of showing one")
    parser.add_argument("--frames", type=int, default=192, help="frames to refine for when saving without a window")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    parser.add_argument("--no-intro", action="store_true", help="skip the scripted throws at the start")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    try:
        play = Playground(width, height, args.hdri)
    except ImportError as problem:
        sys.exit(f"{problem}")
    camera = FlyCamera((0.0, 2.1, 3.0), yaw=0.0, pitch=-0.12, speed=4.0)
    script = [] if args.no_intro else intro(play)
    window = open_window("neraulGraphics: materials", width, height)

    if window is None:                                           # no display: play the script, let the picture refine, write it
        clock = [0.0]
        for step in range(150):
            t = step / 60
            while script and script[0][0] <= t:
                _, material, origin, direction = script.pop(0)
                play.throw_ball(origin, direction, material)
            play.advance(1 / 60, clock)
        view = camera.camera(width, height, 0.7)
        play.frame = 0
        for _ in range(args.frames):
            image = play.render(view)
        save_picture(args.save or "materials.png", image)
        return

    clock, started, fps = [0.0], time.perf_counter(), 0.0
    last_frame = time.perf_counter()
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            dt = window.tick()
            t = time.perf_counter() - started
            while script and script[0][0] <= t:
                _, material, origin, direction = script.pop(0)
                play.throw_ball(origin, direction, material)
            camera.update(window, dt)
            forward = camera.forward()
            for key in window.take_keys():
                if key.isdigit() and 1 <= int(key) <= len(NAMES):
                    play.selected = int(key) - 1
                elif key == "space":
                    play.throw_ball(camera.position + forward * 1.2, forward)
                elif key == "b":
                    play.throw_box(camera.position + forward * 1.2, forward)
                elif key == "r":
                    play.reset()
                    script = []
                elif key in ("bracketleft", "bracketright"):
                    play.sun_elevation = float(np.clip(play.sun_elevation + (0.06 if key == "bracketright" else -0.06), 0.03, 1.5))
                    play.set_sky()
                elif key in ("comma", "period"):
                    play.sun_azimuth += 0.2 if key == "period" else -0.2
                    play.set_sky()
                elif key in ("minus", "equal"):
                    play.exposure = float(np.clip(play.exposure * (1.15 if key == "equal" else 1 / 1.15), 0.1, 8.0))
                    play.set_exposure()
                elif key == "x":
                    play.slow = not play.slow
                elif key == "p":
                    window.screenshot("materials.png")
            for click in window.take_clicks():
                (play.throw_ball if click == "left" else play.throw_box)(camera.position + forward * 1.2, forward)
            play.advance(dt, clock)                                   # the denoiser follows what moved; only a change of light starts the picture over
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_frame, 1e-4)
            last_frame = now
            window.show(play.render(camera.camera(width, height, 0.7)), play.hud(fps))
    if args.save:
        save_picture(args.save, play.render(camera.camera(width, height, 0.7)))


if __name__ == "__main__":
    main()
