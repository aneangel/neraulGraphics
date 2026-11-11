"""People for the street: articulated humanoids made of ellipsoids, walked by forward kinematics.

A skeleton of rigid segments (pelvis, torso, head, two arms of two segments, two legs of two segments and feet) is posed from a gait
cycle: hips and shoulders swing, knees and elbows bend, and the pelvis is lowered or raised so that the lower foot is always on the
ground. Every segment keeps its length, so what moves is the pose and nothing stretches. Each segment is an instance of one unit sphere
squashed into an ellipsoid and laid along its bone, which is how the renderer draws any articulated body: one matrix a segment a frame.

Positions are in Z-up axes; matrices() returns them in the renderer's (Y up).
"""

from dataclasses import dataclass, field

import numpy as np

PARTS = ["pelvis", "torso", "head", "l_upper_arm", "l_forearm", "r_upper_arm", "r_forearm",
         "l_thigh", "l_shin", "l_foot", "r_thigh", "r_shin", "r_foot"]
# which colour group each segment takes: skin, jacket, trousers or shoes
GROUP = {"pelvis": "trousers", "torso": "jacket", "head": "skin", "l_upper_arm": "jacket", "l_forearm": "jacket", "r_upper_arm": "jacket",
         "r_forearm": "jacket", "l_thigh": "trousers", "l_shin": "trousers", "l_foot": "shoes", "r_thigh": "trousers", "r_shin": "trousers", "r_foot": "shoes"}
SIM_TO_RENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], np.float32)       # (x, y, z) -> (x, z, -y)


def segment(a, b, radius, sideways, flatten=1.0):
    """The 3x4 matrix that lays a unit sphere along the bone from a to b: its z axis along the bone, scaled to half the length."""
    d = b - a
    length = np.linalg.norm(d)
    z = d / max(length, 1e-9)
    x = sideways - z * np.dot(z, sideways)
    if np.linalg.norm(x) < 1e-4:                                     # the bone lies along the reference (a pelvis, left to right): any perpendicular will do
        axis = np.eye(3, dtype=np.float32)[np.abs(z).argmin()]
        x = axis - z * np.dot(z, axis)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x * radius, y * radius * flatten, z * max(length / 2.0, radius)], axis=1)
    return np.concatenate([rotation, ((a + b) / 2.0)[:, None]], axis=1)


@dataclass
class Humanoid:
    position: tuple                 # where it is at t = 0, (x, y) on the ground
    heading: float = 0.0            # radians from +x, counter-clockwise seen from above
    speed: float = 1.4              # m/s along the heading (0: standing)
    height: float = 1.75
    phase: float = 0.0              # where in the gait cycle it starts, 0..1
    colors: dict = field(default_factory=lambda: {"skin": (0.62, 0.45, 0.36), "jacket": (0.10, 0.12, 0.18), "trousers": (0.09, 0.09, 0.11), "shoes": (0.03, 0.03, 0.03)})

    def __post_init__(self):
        h = self.height
        self.thigh, self.shin = 0.245 * h, 0.246 * h
        self.upper_arm, self.forearm = 0.186 * h, 0.146 * h
        self.hip_width, self.shoulder_width = 0.095 * h, 0.12 * h
        self.torso_length = 0.30 * h
        self.foot_length = 0.14 * h

    @property
    def stride_frequency(self):
        return 0.0 if self.speed == 0 else 1.0 / (2.0 * 0.4 + 0.3 * 0.0) * (0.55 + 0.25 * self.speed) * 1.3     # steps a second, rising with speed

    def forward(self):
        return np.array([np.cos(self.heading), np.sin(self.heading), 0.0], np.float32)

    def pose(self, t):
        """Joint positions at time t, in Z-up axes: a dict of name -> (a, b) bone end points, plus the root frame."""
        f = self.forward()
        up = np.array([0.0, 0.0, 1.0], np.float32)
        right = np.cross(f, up)                                      # to the person's right
        cycle = (self.phase + t * self.stride_frequency * 0.5) % 1.0
        phi = 2.0 * np.pi * cycle
        moving = 1.0 if self.speed > 0 else 0.0
        swing = 0.52 * min(self.speed / 1.4, 1.4) * moving

        origin = np.array([self.position[0], self.position[1], 0.0], np.float32) + f * (self.speed * t)

        def leg(side):
            p = phi + (0.0 if side > 0 else np.pi)
            hip_angle = swing * np.sin(p)                            # positive swings the thigh forward
            knee_flex = moving * 0.2 + moving * 0.9 * max(0.0, np.sin(p + 0.9)) * min(self.speed / 1.4, 1.2)
            return hip_angle, knee_flex

        # pelvis centre; the legs are solved relative to it, then it is lowered or raised so the lower foot is on the ground
        hips, knees, ankles = {}, {}, {}
        for side in (1, -1):
            hip_angle, knee_flex = leg(side)
            hip = right * (side * self.hip_width)
            knee = hip + (f * np.sin(hip_angle) - up * np.cos(hip_angle)) * self.thigh
            shin_angle = hip_angle - knee_flex
            ankle = knee + (f * np.sin(shin_angle) - up * np.cos(shin_angle)) * self.shin
            hips[side], knees[side], ankles[side] = hip, knee, ankle
        ankle_height = 0.045 * self.height                           # the foot is a slab under the ankle
        lowest = min(ankles[1][2], ankles[-1][2])
        bob = moving * 0.012 * np.cos(2.0 * phi)
        pelvis = origin + np.array([0.0, 0.0, ankle_height - lowest + bob], np.float32)

        pose = {"pelvis": (pelvis + right * (-self.hip_width) * 0.9, pelvis + right * self.hip_width * 0.9)}
        lean = f * (0.06 * moving * min(self.speed / 1.4, 1.2))
        neck = pelvis + up * self.torso_length * 0.95 + lean
        pose["torso"] = (pelvis + up * 0.02 * self.height, neck)
        pose["head"] = (neck + up * 0.02 * self.height, neck + up * 0.205 * self.height + lean * 0.3)
        for side, name in ((1, "r"), (-1, "l")):
            hip_angle, knee_flex = leg(side)
            hip = pelvis + hips[side]
            knee = pelvis + knees[side]
            ankle = pelvis + ankles[side]
            shin_angle = hip_angle - knee_flex
            pose[f"{name}_thigh"] = (hip, knee)
            pose[f"{name}_shin"] = (knee, ankle)
            toe = ankle + f * self.foot_length + up * (-ankle_height * 0.6)
            heel = ankle - f * 0.03 * self.height + up * (-ankle_height * 0.6)
            pose[f"{name}_foot"] = (heel, toe)
            # the arm on the same side swings against the leg
            shoulder = neck - up * 0.03 * self.height + right * (side * self.shoulder_width)
            arm_angle = -0.8 * swing * np.sin(phi + (0.0 if side > 0 else np.pi))
            elbow_bend = 0.25 + moving * 0.45 * (0.5 + 0.5 * np.sin(phi + (0.0 if side > 0 else np.pi) + np.pi / 2))
            elbow = shoulder + (f * np.sin(arm_angle) - up * np.cos(arm_angle)) * self.upper_arm + right * (side * 0.015 * self.height)
            hand = elbow + (f * np.sin(arm_angle + elbow_bend) - up * np.cos(arm_angle + elbow_bend)) * self.forearm
            pose[f"{name}_upper_arm"] = (shoulder, elbow)
            pose[f"{name}_forearm"] = (elbow, hand)
        return pose, right

    def matrices(self, t):
        """(len(PARTS), 3, 4) renderer matrices (Y up) of every segment at time t."""
        pose, right = self.pose(t)
        h = self.height
        radii = {"pelvis": 0.075 * h, "torso": 0.10 * h, "head": 0.062 * h, "l_upper_arm": 0.032 * h, "l_forearm": 0.027 * h, "r_upper_arm": 0.032 * h,
                 "r_forearm": 0.027 * h, "l_thigh": 0.052 * h, "l_shin": 0.036 * h, "l_foot": 0.03 * h, "r_thigh": 0.052 * h, "r_shin": 0.036 * h, "r_foot": 0.03 * h}
        out = np.zeros((len(PARTS), 3, 4), np.float32)
        for i, name in enumerate(PARTS):
            a, b = pose[name]
            flatten = 0.7 if name in ("torso", "pelvis") else 1.0
            m = segment(np.asarray(a, np.float32), np.asarray(b, np.float32), radii[name], right, flatten)
            out[i, :, :3] = SIM_TO_RENDER @ m[:, :3]
            out[i, :, 3] = SIM_TO_RENDER @ m[:, 3]
        return out

    def materials(self):
        return [self.colors[GROUP[name]] for name in PARTS]


def crowd(structure, seed=0, count=7):
    """People for the street in front of a structure: some walking along the sidewalk and the road, a few standing."""
    rng = np.random.default_rng(seed)
    lo = (structure.position - structure.half).min(axis=0)
    cx = (structure.position[:, 0].min() + structure.position[:, 0].max()) / 2
    front = lo[1]
    jackets = [(0.10, 0.12, 0.18), (0.35, 0.08, 0.07), (0.20, 0.25, 0.20), (0.45, 0.42, 0.36), (0.08, 0.08, 0.09)]
    skins = [(0.62, 0.45, 0.36), (0.45, 0.30, 0.22), (0.78, 0.60, 0.50), (0.30, 0.20, 0.15)]
    people = []
    for k in range(count):
        side = rng.choice([-1.0, 1.0])
        x = cx + side * rng.uniform(5.0, 17.0)                      # never on the centre line
        y = front - rng.uniform(2.2, 4.0) if k % 3 else front - rng.uniform(8.0, 14.0)
        walking = k % 4 != 3
        heading = float(rng.choice([0.0, np.pi])) if walking else float(rng.uniform(0, 2 * np.pi))
        colors = {"skin": skins[int(rng.integers(len(skins)))], "jacket": jackets[int(rng.integers(len(jackets)))],
                  "trousers": (0.09, 0.09, 0.12), "shoes": (0.03, 0.03, 0.03)}
        people.append(Humanoid((x, y), heading, float(rng.uniform(1.1, 1.6)) if walking else 0.0, float(rng.uniform(1.62, 1.88)), float(rng.random()), colors))
    return people
