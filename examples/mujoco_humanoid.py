"""A MuJoCo humanoid, path traced live: a ragdoll with joint limits, motors, and balls to throw at it.

    python examples/mujoco_humanoid.py                      # opens the viewer, with the built-in humanoid
    python examples/mujoco_humanoid.py model.xml            # any MuJoCo model
    python examples/mujoco_humanoid.py --save humanoid.png  # no window: let it fall, write a picture

MuJoCo simulates the body; neraulgraphics turns each step's geom poses into instance matrices and draws them with the path tracer under a physical
sky. The humanoid starts standing with its motors off, so it collapses as a ragdoll against its joint limits; T switches the motors on, which drive
the legs and arms in a stride (it will not balance, only move). Space throws a ball at it, G shoves it, R puts everything back. Drag with the left
button to orbit, the wheel zooms, [ and ] move the sun, B switches between path tracing and the fast classic picture, P saves a picture, Esc quits.

Needs: pip install mujoco pillow pyglet
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import neraulgraphics as ng
from neraulgraphics.mujoco import MuJoCoRenderer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import OrbitCamera, open_window, save_picture  # noqa: E402

BALLS = 6


def humanoid_xml() -> str:
    """A humanoid of capsules and hinges (about 1.7 m), a floor, and a few balls parked off to the side."""
    def leg(side, y):
        return f"""
        <body name="thigh_{side}" pos="0 {y} -.04">
          <joint name="hip_{side}" type="hinge" axis="0 1 0" range="-100 25" damping="4"/>
          <geom name="thigh_{side}" type="capsule" fromto="0 0 0 0 0 -.38" size=".06" rgba=".25 .25 .32 1"/>
          <body name="shin_{side}" pos="0 0 -.4">
            <joint name="knee_{side}" type="hinge" axis="0 1 0" range="2 130" damping="3"/>
            <geom name="shin_{side}" type="capsule" fromto="0 0 0 0 0 -.36" size=".05" rgba=".25 .25 .32 1"/>
            <body name="foot_{side}" pos="0 0 -.4">
              <joint name="ankle_{side}" type="hinge" axis="0 1 0" range="-40 40" damping="2"/>
              <geom name="foot_{side}" type="box" pos=".06 0 -.03" size=".12 .05 .03" rgba=".1 .1 .1 1"/>
            </body>
          </body>
        </body>"""

    def arm(side, y):
        return f"""
      <body name="upper_arm_{side}" pos="0 {y} .08">
        <joint name="shoulder_{side}" type="hinge" axis="0 1 0" range="-170 60" damping="2"/>
        <geom name="upper_arm_{side}" type="capsule" fromto="0 0 0 0 0 -.28" size=".045" rgba=".2 .35 .75 1"/>
        <body name="forearm_{side}" pos="0 0 -.3">
          <joint name="elbow_{side}" type="hinge" axis="0 1 0" range="-120 0" damping="1.5"/>
          <geom name="forearm_{side}" type="capsule" fromto="0 0 0 0 0 -.26" size=".04" rgba=".9 .75 .6 1"/>
        </body>
      </body>"""

    balls = "".join(f"""
    <body name="ball{i}" pos="{30 + 2 * i} 0 .2"><freejoint name="ball{i}"/>
      <geom type="sphere" size=".2" density="600" rgba="{('.9 .2 .2 1', '.2 .7 .3 1', '.95 .75 .2 1')[i % 3]}"/></body>""" for i in range(BALLS))
    motors = "".join(f'<motor name="m_{j}" joint="{j}" gear="{g}" ctrllimited="true" ctrlrange="-1 1"/>' for j, g in [
        ("hip_right", 80), ("knee_right", 60), ("ankle_right", 30), ("hip_left", 80), ("knee_left", 60), ("ankle_left", 30),
        ("shoulder_right", 25), ("elbow_right", 15), ("shoulder_left", 25), ("elbow_left", 15)])
    return f"""
<mujoco model="humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.004" iterations="40"/>
  <default>
    <joint armature="0.05" limited="true"/>
    <geom contype="1" conaffinity="1" condim="3" friction="1 .1 .1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="600 600 .1" rgba=".5 .55 .62 1"/>
    <body name="torso" pos="0 0 1.2">
      <freejoint name="root"/>
      <geom name="chest" type="capsule" fromto="0 -.08 .06 0 .08 .06" size=".08" rgba=".2 .35 .75 1"/>
      <geom name="waist" type="capsule" fromto="0 -.06 -.1 0 .06 -.1" size=".065" rgba=".2 .35 .75 1"/>
      <geom name="head" type="sphere" pos="0 0 .27" size=".095" rgba=".9 .75 .6 1"/>
      <geom name="neck" type="capsule" fromto="0 0 .1 0 0 .2" size=".04" rgba=".9 .75 .6 1"/>
      {arm('right', -.19)}{arm('left', .19)}
      <body name="pelvis" pos="-.01 0 -.2">
        <geom name="pelvis" type="capsule" fromto="0 -.09 0 0 .09 0" size=".085" rgba=".25 .25 .32 1"/>
        {leg('right', -.1)}{leg('left', .1)}
      </body>
    </body>{balls}
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>"""


class Humanoid:
    def __init__(self, width, height, path=None):
        self.model = mujoco.MjModel.from_xml_path(path) if path else mujoco.MjModel.from_xml_string(humanoid_xml())
        self.data = mujoco.MjData(self.model)
        self.view = MuJoCoRenderer(self.model, width, height)
        self.balls = [b for b in range(BALLS) if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ball{b}") >= 0]
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.sun_elevation, self.sun_azimuth = 0.6, 2.2
        self.beauty, self.motors_on, self.frame, self.next_ball, self.t = True, False, 0, 0, 0.0
        self.set_mode()
        mujoco.mj_forward(self.model, self.data)

    def set_mode(self):
        if self.beauty:
            sky = ng.physical_sky(sun_elevation=self.sun_elevation, sun_azimuth=self.sun_azimuth, turbidity=2.6, sun_strength=4.0, sun_radius=0.02,
                                  sky_ratio=0.22, width=768, height=384)
            self.view.beauty(sky, bounces=6, exposure=0.8, denoise=True, denoise_iterations=3, max_radiance=60.0)
        else:
            self.view.classic()
        self.frame = 0

    def step(self, frame_time=1 / 60):
        """Advances the simulation by frame_time seconds, driving the motors in a stride if they are on."""
        for _ in range(max(1, int(round(frame_time / self.model.opt.timestep)))):
            if self.model.nu and self.motors_on:
                w = 2 * math.pi * 1.1 * self.t
                phase = np.array([0.0, 0.0, 0.0, math.pi, math.pi, math.pi, math.pi, 0.0, 0.0, 0.0])[:self.model.nu]
                self.data.ctrl[:] = np.array([0.9, 0.7, 0.2, 0.9, 0.7, 0.2, 0.8, 0.5, 0.8, 0.5])[:self.model.nu] * np.sin(w + phase)
            elif self.model.nu:
                self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.t += self.model.opt.timestep

    def throw_ball(self, origin, direction, speed=11.0):
        """Throws a ball from origin along direction (world axes, Z up)."""
        if not self.balls:
            return
        b = self.balls[self.next_ball % len(self.balls)]
        self.next_ball += 1
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ball{b}")
        qpos, dof = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        self.data.qpos[qpos:qpos + 7] = (*origin, 1, 0, 0, 0)
        self.data.qvel[dof:dof + 6] = (*(d * speed), 0, 0, 0)

    def shove(self):
        """A blow to the torso: an impulse of 60 N s towards -y, a little upward."""
        self.data.qvel[:3] += np.array([0.0, -60.0, 15.0]) / float(self.model.body_subtreemass[self.torso])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.next_ball, self.t = 0, 0.0

    def moving(self):
        return float(np.abs(self.data.qvel).max()) > 0.05

    def centre(self):
        return np.array(self.data.xpos[self.torso]) if self.torso >= 0 else np.zeros(3)

    def render(self, camera):
        image = self.view.render(self.data, camera, frame=self.frame, samples_per_frame=4 if (self.beauty and self.view.renderer.device == "cuda") else 1)
        self.frame += 1
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="a MuJoCo XML model; with none, the built-in humanoid")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--save", metavar="FILE", help="write a picture to FILE (a PNG); with no window, instead of showing one")
    parser.add_argument("--gif", metavar="FILE", help="with no window, write the fall as an animated GIF")
    parser.add_argument("--frames", type=int, default=96, help="frames to refine for when saving without a window")
    parser.add_argument("--seconds", type=float, default=0.0, help="close the window after this long (0: stay open)")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    sim = Humanoid(width, height, args.model)
    orbit = OrbitCamera(sim.centre() * np.array([1, 1, 0.45]), 3.3, yaw=0.9, pitch=0.2, up=(0.0, 0.0, 1.0))
    window = open_window("neraulGraphics: MuJoCo humanoid", width, height)

    if window is None:
        frames = []
        for step in range(150):
            if step == 20:
                sim.throw_ball((0.0, 4.5, 1.1), (0.0, -1.0, 0.0), 9.0)
            sim.step()
            if args.gif and step % 3 == 0:
                sim.frame = 0
                frames.append(sim.render(orbit.camera(width, height)))
        sim.frame = 0
        for _ in range(args.frames):
            picture = sim.render(orbit.camera(width, height))
        if args.gif:
            from PIL import Image

            images = [Image.fromarray(f) for f in frames]
            images[0].save(args.gif, save_all=True, append_images=images[1:], duration=50, loop=0)
            print("wrote", args.gif)
        save_picture(args.save or "humanoid.png", picture)
        return

    window.mouse_mode = "orbit"
    started, last, fps, owed = time.perf_counter(), time.perf_counter(), 0.0, 0.0
    with window:
        while window.running and not (args.seconds and time.perf_counter() - started > args.seconds):
            dt = window.tick()
            orbit.update(window, dt)
            for key in window.take_keys():
                if key == "space":
                    eye = orbit.position()
                    aim = sim.centre() + (0.0, 0.0, 0.3)
                    sim.throw_ball(eye + 0.4 * (aim - eye) / np.linalg.norm(aim - eye), aim - eye)
                elif key == "g":
                    sim.shove()
                elif key == "t":
                    sim.motors_on = not sim.motors_on
                elif key == "r":
                    sim.reset()
                elif key in ("bracketleft", "bracketright"):
                    sim.sun_elevation = float(np.clip(sim.sun_elevation + (0.06 if key == "bracketright" else -0.06), 0.03, 1.5))
                    sim.set_mode()
                elif key == "b":
                    sim.beauty = not sim.beauty
                    sim.set_mode()
                elif key == "p":
                    window.screenshot("humanoid.png")
            owed += dt
            steps = 0
            while owed >= 1 / 60 and steps < 3:
                sim.step(1 / 60)
                owed -= 1 / 60
                steps += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-4)
            last = now
            hud = (f"{fps:4.0f} fps   {sim.frame} frames refining   motors {'on' if sim.motors_on else 'off (ragdoll)'}   MuJoCo {mujoco.__version__}\n"
                   "drag orbit   wheel zoom   Space throw a ball   G shove   T motors   R reset   [ ] sun   B beauty/classic   P picture   Esc quit")
            window.show(sim.render(orbit.camera(width, height)), hud)
    if args.save:
        save_picture(args.save, sim.render(orbit.camera(width, height)))


if __name__ == "__main__":
    main()
