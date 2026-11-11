"""The one scene every benchmark draws, defined once so every renderer is handed the same triangles.

Each world holds `nbody` free bodies falling onto the ground: even ones a ball, odd ones a box. Physics runs on analytic
primitives (gen_poses.py); drawing uses the matching triangle meshes, with no ground, so no renderer is credited or
charged for something another does differently (an analytic sphere, a textured floor). Pure numpy: it is imported from
environments that share nothing else.
"""

import numpy as np

BALL_RADIUS = 0.12
BOX_HALF = 0.1
COLUMNS = 8
SPACING = 0.3
FOVY = 45.0  # degrees, vertical
LIGHT_POSITION = (0.0, 0.0, 4.0)


def icosphere(subdivisions=2):
    """Unit icosphere (vertices, faces); 20 * 4**subdivisions triangles, wound counter-clockwise seen from outside."""
    t = (1.0 + 5.0**0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = faces
    return np.array(verts, dtype=np.float32), np.array(f, dtype=np.int32)


def unit_box():
    """Cube spanning [-1, 1] on each axis (vertices, faces), 12 triangles wound counter-clockwise seen from outside."""
    v = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=np.float32)
    f = np.array([(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)],
                 dtype=np.int32)
    return v, f


def layout(nbody):
    """Starting (x, y, z) of each body: a grid of COLUMNS across, centred on x = 0, dropped from staggered heights."""
    return [((i % COLUMNS - (COLUMNS - 1) / 2) * SPACING, (i // COLUMNS) * SPACING - 0.2, 0.3 + 0.02 * i) for i in range(nbody)]


def camera(nbody):
    """Position and MuJoCo xyaxes of a camera that frames the whole grid from the front and above."""
    rows = (nbody + COLUMNS - 1) // COLUMNS
    eye = np.array([0.0, -1.9, 1.1])
    target = np.array([0.0, (rows - 1) * SPACING / 2 - 0.2, 0.15])
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, right, up


def _mesh_asset(name, mesh, scale):
    v, f = mesh
    return (f'<mesh name="{name}" vertex="{" ".join(f"{x:.9g}" for x in v.ravel())}" '
            f'face="{" ".join(str(i) for i in f.ravel())}" scale="{scale} {scale} {scale}"/>')


def physics_xml(nbody):
    bodies = []
    for i, (x, y, z) in enumerate(layout(nbody)):
        geom = (f'<geom type="sphere" size="{BALL_RADIUS}"/>' if i % 2 == 0 else f'<geom type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}"/>')
        bodies.append(f'<body pos="{x} {y} {z}"><freejoint/>{geom}</body>')
    return ('<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="5 5 0.1"/>' + "".join(bodies) + "</worldbody></mujoco>")


def render_xml(nbody, subdivisions=2):
    """The same bodies as physics_xml (same joints, same order) carrying triangle meshes instead of primitives."""
    assets = _mesh_asset("ball", icosphere(subdivisions), BALL_RADIUS) + _mesh_asset("box", unit_box(), BOX_HALF)
    bodies = []
    for i, (x, y, z) in enumerate(layout(nbody)):
        bodies.append(f'<body pos="{x} {y} {z}"><freejoint/><geom type="mesh" mesh="{"ball" if i % 2 == 0 else "box"}" '
                      f'contype="0" conaffinity="0" rgba="{"0.8 0.2 0.2 1" if i % 2 == 0 else "0.2 0.4 0.8 1"}"/></body>')
    eye, right, up = camera(nbody)
    cam = (f'<camera name="cam" pos="{eye[0]} {eye[1]} {eye[2]}" xyaxes="{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}" '
           f'fovy="{FOVY}"/>')
    light = f'<light pos="{LIGHT_POSITION[0]} {LIGHT_POSITION[1]} {LIGHT_POSITION[2]}" dir="0 0 -1" directional="true" castshadow="true"/>'
    return f"<mujoco><asset>{assets}</asset><worldbody>{light}{cam}{''.join(bodies)}</worldbody></mujoco>"
