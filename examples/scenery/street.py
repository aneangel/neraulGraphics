"""The surroundings of a structure: a street with a sidewalk, lane markings, parked cars, lamp posts, barriers and crates, the
buildings either side of it and a skyline behind it. Everything here is static; the viewer draws it as instances of box
meshes. Positions are in Z-up axes; a viewer approaches from the -y side, across the street."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Prop:
    half: tuple               # half extents (x, y, z), metres
    position: tuple           # centre (x, y, z), metres
    color: tuple              # diffuse colour
    roughness: float = 0.9
    metallic: float = 0.0
    emission: tuple = (0.0, 0.0, 0.0)
    texture: str = ""          # one of textures.all_textures(); the colour is multiplied into it
    tile: float = 2.0          # metres per repeat of the texture


WALLS = [(0.42, 0.30, 0.25), (0.50, 0.47, 0.42), (0.35, 0.37, 0.40), (0.46, 0.40, 0.30), (0.30, 0.31, 0.30)]
CARS = [(0.62, 0.62, 0.60), (0.10, 0.10, 0.11), (0.40, 0.08, 0.07), (0.10, 0.18, 0.34), (0.42, 0.36, 0.24), (0.25, 0.26, 0.27)]
GLASS = (0.025, 0.03, 0.04)


def box(half, position, color, **kw):
    return Prop(tuple(float(v) for v in half), tuple(float(v) for v in position), tuple(float(v) for v in color), **kw)


def facade(x0, x1, y_face, height, rng, lit=0.28, floor_height=3.4, pitch=3.2, glow=1.0):
    """The windows of a wall facing -y: a grid of dark glass, a few lit from within, with a plinth and a cornice."""
    props = []
    columns = max(int((x1 - x0) // pitch), 1)
    margin = ((x1 - x0) - (columns - 1) * pitch) / 2.0
    glow_scale = glow
    for floor in range(max(int((height - 1.0) // floor_height), 1)):
        z = 1.6 + floor * floor_height + 1.0
        for c in range(columns):
            x = x0 + margin + c * pitch
            lit_here = rng.random() < lit
            warm = tuple(c * glow_scale for c in (2.2, 1.45, 0.65)) if lit_here else (0.0, 0.0, 0.0)
            props.append(box((0.75, 0.07, 0.95), (x, y_face - 0.05, z), (0.9, 0.7, 0.4) if lit_here else GLASS, roughness=0.08, metallic=0.0,
                             emission=warm))
            props.append(box((0.9, 0.05, 0.06), (x, y_face - 0.08, z - 1.0), (0.55, 0.55, 0.55)))        # sill
    props.append(box(((x1 - x0) / 2 + 0.1, 0.18, 0.55), ((x0 + x1) / 2, y_face - 0.1, 0.55), (0.28, 0.28, 0.27)))      # plinth
    props.append(box(((x1 - x0) / 2 + 0.2, 0.3, 0.25), ((x0 + x1) / 2, y_face - 0.2, height - 0.2), (0.55, 0.54, 0.5)))  # cornice
    return props


def building(x0, x1, y0, y1, height, rng, **kw):
    which = int(rng.integers(len(WALLS)))
    color = WALLS[which]
    # the warm wall is brick; the others are stucco or concrete, tinted (the textures are pale, so the tint shows through)
    texture, tile, tint = ("brick", 1.5, (1.0, 1.0, 1.0)) if which == 0 else (("stucco", 3.0, color) if which % 2 else ("concrete", 2.6, color))
    shell = box(((x1 - x0) / 2, (y1 - y0) / 2, height / 2), ((x0 + x1) / 2, (y0 + y1) / 2, height / 2), tuple(min(c * 1.25, 1.0) for c in tint) if texture != "brick" else tint,
                roughness=0.95, texture=texture, tile=tile)
    return [shell] + facade(x0, x1, y0, height, rng, **kw)


def car(x, y, heading_x, rng):
    """A parked car as boxes: body, cabin, glass, wheels. heading_x is 1 for pointing along x (the usual along the kerb)."""
    colour = CARS[int(rng.integers(len(CARS)))]
    sx, sy = (2.2, 0.92) if heading_x else (0.92, 2.2)
    cabin = (1.15, 0.8) if heading_x else (0.8, 1.15)
    props = [box((sx, sy, 0.36), (x, y, 0.62), colour, roughness=0.35, metallic=0.2),
             box((cabin[0], cabin[1], 0.3), (x, y, 1.1), colour, roughness=0.35, metallic=0.2),
             box((cabin[0] + 0.01, cabin[1] + 0.01, 0.22), (x, y, 1.14), GLASS, roughness=0.05)]
    for dx in (-1.35, 1.35):
        for dy in (-0.9, 0.9):
            wx, wy = (x + dx, y + dy) if heading_x else (x + dy, y + dx)
            props.append(box((0.32, 0.12, 0.32) if heading_x else (0.12, 0.32, 0.32), (wx, wy, 0.32), (0.02, 0.02, 0.02), roughness=0.8))
    return props


def street(structure, seed=0, night=False):
    """The props for a structure: everything is placed relative to the structure's footprint. night lights the windows more often and
    far brighter, and the lamps harder, for a scene lit by them alone."""
    lit, glow, lamp = (0.14, 1.1, 3.2) if night else (0.28, 1.0, 6.0)      # at night a few windows and the lamps, not a lit-up street
    rng = np.random.default_rng(seed)
    lo = (structure.position - structure.half).min(axis=0)
    hi = (structure.position + structure.half).max(axis=0)
    cx = (lo[0] + hi[0]) / 2
    front = lo[1]                                  # the face a viewer approaching from -y sees
    depth = hi[1] - lo[1]
    props = []

    # ground: asphalt out to the far kerb, a raised sidewalk in front of the building, lane markings
    props.append(box((70, 14.5, 0.015), (cx, front - 18.5, 0.015), (1.0, 1.0, 1.0), roughness=0.97, texture="asphalt", tile=4.0))
    props.append(box((70, 1.75, 0.08), (cx, front - 2.9, 0.08), (1.0, 1.0, 1.0), roughness=0.95, texture="concrete", tile=2.4))
    props.append(box((70, 0.12, 0.09), (cx, front - 4.75, 0.09), (1.0, 1.0, 1.0), roughness=0.9, texture="concrete", tile=1.0))     # kerb
    for x in np.arange(cx - 60, cx + 60, 7.0):
        props.append(box((1.6, 0.09, 0.004), (x, front - 18.5, 0.032), (0.62, 0.60, 0.50), roughness=0.8))
    props.append(box((70, 0.07, 0.004), (cx, front - 23.5, 0.032), (0.55, 0.5, 0.2), roughness=0.8))

    # the buildings either side of the structure, set a few metres off, and the skyline behind
    left_w, right_w = float(rng.uniform(10, 16)), float(rng.uniform(10, 16))
    props += building(lo[0] - 3.5 - left_w, lo[0] - 3.5, front, front + depth, float(rng.uniform(9, 15)), rng, lit=lit, glow=glow)
    props += building(hi[0] + 3.5, hi[0] + 3.5 + right_w, front, front + depth, float(rng.uniform(9, 15)), rng, lit=lit, glow=glow)
    x = cx - 55.0
    while x < cx + 55.0:
        w = float(rng.uniform(12, 22))
        props += building(x, x + w, hi[1] + 7.0, hi[1] + 7.0 + float(rng.uniform(10, 16)), float(rng.uniform(14, 34)), rng, lit=min(lit + 0.12, 0.9), glow=glow)
        x += w + float(rng.uniform(1.5, 4.5))
    for side, edge in ((-1, lo[0] - 3.5 - left_w - 2.5), (1, hi[0] + 3.5 + right_w + 2.5)):     # and blocks further along the street
        x = edge
        for _ in range(3):
            w = float(rng.uniform(12, 18))
            xa, xb = (x - w, x) if side < 0 else (x, x + w)
            props += building(xa, xb, front, front + float(rng.uniform(8, 12)), float(rng.uniform(10, 20)), rng, lit=lit, glow=glow)
            x = xa - 2.5 if side < 0 else xb + 2.5

    # parked cars along the kerb and in the road, lamp posts, barriers, crates, cones
    for x in cx + rng.uniform(-30, 30, 7):
        props += car(x if abs(x - cx) > 4.0 else x + 8.0, front - 7.2, True, rng)       # not on the centre line
    for x in cx + rng.uniform(-26, 26, 4):
        props += car(x, front - 29.0, True, rng)
    for x in np.arange(cx - 30, cx + 37, 12.0):               # none on the centre line, which is kept clear
        props.append(box((0.07, 0.07, 3.0), (x, front - 5.3, 3.0), (0.12, 0.12, 0.13), metallic=0.6, roughness=0.5))
        props.append(box((0.5, 0.18, 0.1), (x, front - 5.2, 6.05), (0.9, 0.8, 0.6), emission=(lamp, lamp * 0.73, lamp * 0.4)))
    for i in range(7):
        if abs(cx - 9.0 + i * 2.2 - cx) < 2.6:
            continue                                              # a gap in the line, a crossing on the centre line
        props.append(box((1.0, 0.2, 0.4), (cx - 9.0 + i * 2.2, front - 11.0, 0.4), (0.80, 0.80, 0.78), roughness=0.9, texture="concrete", tile=0.9))          # barriers
    for i in range(9):
        props.append(box((0.14, 0.14, 0.3), (cx + 4.0 + 0.9 * (i % 3), front - 8.5 - 0.9 * (i // 3), 0.3), (0.95, 0.38, 0.06), roughness=0.7))
    for i in range(5):
        props.append(box(tuple(rng.uniform(0.3, 0.6, 3)), (hi[0] + 1.5 + rng.uniform(0, 1.5), front - 1.0 - rng.uniform(0, 1.5), 0.45), (0.38, 0.28, 0.17), roughness=0.95))
    props.append(box((1.8, 0.9, 0.75), (lo[0] - 1.5, front - 2.5, 0.8), (0.10, 0.26, 0.15), roughness=0.6, metallic=0.3))       # dumpster
    return props
