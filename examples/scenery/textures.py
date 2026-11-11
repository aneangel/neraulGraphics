"""Procedural surface textures for the street scene, made with numpy: brick, asphalt, weathered concrete and stucco. Each is a
seamless (size, size, 3) uint8 sRGB image that repeats, with grain at the scale of the material, slow stains and, on walls, vertical
streaks of grime, so that no large surface is a flat colour."""

import numpy as np


def noise(size, cells, rng, octaves=1, persistence=0.5):
    """Seamless value noise in 0..1: random values on a cells x cells grid, smoothly interpolated, summed over octaves."""
    total = np.zeros((size, size), np.float32)
    amplitude, norm = 1.0, 0.0
    for _ in range(octaves):
        grid = rng.random((cells, cells)).astype(np.float32)
        x = np.linspace(0, cells, size, endpoint=False, dtype=np.float32)
        i = x.astype(int)
        f = x - i
        f = f * f * (3 - 2 * f)                                    # smoothstep, so the grid does not show
        i1 = (i + 1) % cells
        rows = grid[i][:, :] * (1 - f)[:, None] + grid[i1][:, :] * f[:, None]            # blend along y
        total += amplitude * (rows[:, i] * (1 - f)[None, :] + rows[:, i1] * f[None, :])   # then along x
        norm += amplitude
        amplitude *= persistence
        cells *= 2
    return total / norm


def to_srgb(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def grime(size, rng, strength=0.18):
    """Vertical streaks where water has run down a wall: long in y, narrow in x."""
    across = noise(size, 48, rng, 2)[0, :]                                        # one value per column ...
    fade = 0.5 + 0.5 * noise(size, 3, rng, 1)                                      # ... that fades in and out down the wall
    streaks = across[None, :] * fade
    blotch = noise(size, 6, rng, 3)
    return 1.0 - strength * (0.6 * streaks + 0.8 * blotch)


def normal_map(height, strength):
    """A tangent-space normal map from a height field (seamless): the slope along +u and up the image, bent into a unit normal and stored as
    0..255 for -1..1. strength is how steep one unit of height over one texel is."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    d_row = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5        # rows run down the image, the map's +y runs up
    n = np.stack([-dx * strength, d_row * strength, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return to_srgb(n * 0.5 + 0.5)                                                   # not colour: it is uploaded as linear data


def orm_map(ao, roughness, metal=0.0):
    """Occlusion, roughness and metal in red, green and blue, as glTF packs them."""
    return to_srgb(np.stack([ao, roughness, np.broadcast_to(np.float32(metal), ao.shape)], axis=-1))


def brick(size=512, rows=20, columns=7, seed=1):
    rng = np.random.default_rng(seed)
    h, w = size / rows, size / columns
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    row = np.floor(y / h).astype(int)
    shifted = x + (row % 2) * (w / 2)                                # every other course is offset by half a brick
    col = np.floor(shifted / w).astype(int) % columns
    tone = rng.uniform(0.75, 1.15, (rows, columns)).astype(np.float32)[row, col]
    hue = rng.uniform(-0.04, 0.04, (rows, columns)).astype(np.float32)[row, col]
    fx, fy = (shifted % w) / w, (y % h) / h
    mortar = (np.minimum(fx, 1 - fx) < 0.035) | (np.minimum(fy, 1 - fy) < 0.09)
    grain = 0.9 + 0.2 * noise(size, 64, rng, 2)
    base = np.stack([0.50 + hue, 0.24 + hue * 0.5, np.full_like(hue, 0.17)], axis=-1) * (tone * grain)[..., None]
    base[mortar] = np.array([0.55, 0.53, 0.49], np.float32) * (0.85 + 0.3 * noise(size, 128, rng, 1))[mortar][:, None]
    rgb = to_srgb(base * grime(size, rng, 0.22)[..., None])
    soft = noise(size, 256, rng, 1)
    height = np.where(mortar, 0.0, 1.0).astype(np.float32) * 6.0 + soft * 1.2         # the bricks stand proud of the mortar
    ao = np.where(mortar, 0.55, 1.0).astype(np.float32)
    return rgb, normal_map(height, 0.35), orm_map(ao, np.where(mortar, 0.98, 0.82 + 0.1 * soft).astype(np.float32))


def asphalt(size=512, seed=2):
    rng = np.random.default_rng(seed)
    grit = noise(size, 256, rng, 1)
    stones = (rng.random((size, size)) > 0.985).astype(np.float32) * 0.18               # bright chips of aggregate
    patch = noise(size, 5, rng, 3)
    value = 0.085 + 0.045 * grit + 0.03 * (patch - 0.5) + stones * 0.4
    cracks = noise(size, 12, rng, 4)
    crack = np.zeros_like(value, bool)                                                  # (contour-line cracks read as blotches in the dark, so none)
    rgb = to_srgb(np.stack([value, value, value * 1.03], axis=-1) * 3.2)
    # wet patches: smoother, so that they mirror the sky; the rest is dry and rough
    wet = np.clip((noise(size, 2, rng, 2) - 0.5) * 4.0, 0.0, 1.0) * 0.8                 # broad, soft damp areas
    height = grit * 2.0 + stones * 3.0 - crack * 2.0
    return rgb, normal_map(height, 0.3), orm_map(np.ones_like(value), 0.92 - 0.42 * wet)


def concrete(size=512, seed=3, tint=(1.0, 1.0, 1.0), stains=0.3, streaks=0.2):
    rng = np.random.default_rng(seed)
    grain = 0.88 + 0.16 * noise(size, 128, rng, 2)
    pits = 1.0 - 0.35 * (rng.random((size, size)) > 0.992)                             # air voids
    stain = (1.0 - stains / 2.0) + stains * noise(size, 4, rng, 4)           # slow stains: the part of a texture that shows its repetition
    value = 0.78 * grain * pits * stain * grime(size, rng, streaks)
    rgb = to_srgb(value[..., None] * np.asarray(tint, np.float32))
    height = grain * 2.5 - (pits < 1.0) * 4.0
    return rgb, normal_map(height, 0.4), orm_map(np.where(pits < 1.0, 0.6, 1.0).astype(np.float32), (0.86 + 0.1 * noise(size, 24, rng, 2)).astype(np.float32))


def stucco(size=512, seed=4):
    rng = np.random.default_rng(seed)
    bump = noise(size, 192, rng, 2)
    value = 0.80 * (0.9 + 0.2 * bump) * (0.9 + 0.2 * noise(size, 6, rng, 3)) * grime(size, rng, 0.25)
    rgb = to_srgb(value[..., None] * np.array([1.0, 0.99, 0.96], np.float32))
    return rgb, normal_map(bump * 3.0, 0.5), orm_map(np.ones_like(bump), (0.9 + 0.08 * bump).astype(np.float32))


def all_textures():
    """name -> (colour, normal map, occlusion/roughness/metal map): three images per material, the first sRGB colour and the others data."""
    return {"brick": brick(), "asphalt": asphalt(), "concrete": concrete(), "stucco": stucco()}


def textured_box(half, tile):
    """Vertices, triangles, normals and uvs of a box with half extents half (x, y, z, Y up), 24 vertices so that each face has its own
    normal, with uvs in units of tile metres so that a texture repeats at the same real size on a wall and a crate."""
    hx, hy, hz = half
    faces = [  # normal, u axis, v axis (v runs down the texture, so for walls it is -y), in-plane sizes
        ((1, 0, 0), (0, 0, -1), (0, -1, 0)), ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((0, 0, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ]
    h = np.array([hx, hy, hz], np.float32)
    vertices, uvs, normals, triangles = [], [], [], []
    for n, u, v in faces:
        n, u, v = (np.array(a, np.float32) for a in (n, u, v))
        centre = n * h
        eu, ev = u * h, v * h                      # half vectors along the texture's axes
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(centre + su * eu + sv * ev)
            uvs.append((su * np.abs(eu).sum() / tile + np.abs(eu).sum() / tile, sv * np.abs(ev).sum() / tile + np.abs(ev).sum() / tile))
            normals.append(n)
        k = len(vertices) - 4
        triangles += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return (np.array(vertices, np.float32), np.array(triangles, np.uint32), np.array(normals, np.float32), np.array(uvs, np.float32))


def box_projected_uvs(corners, face_normals, tile):
    """uvs for a triangle soup, per triangle corner: the object-space position projected along the face normal's dominant axis, so
    a texture wraps a box of any size without stretching."""
    dominant = np.abs(face_normals).argmax(axis=1)
    axes = np.array([[1, 2], [0, 2], [0, 1]])[dominant]
    return np.take_along_axis(corners, axes, axis=1) / tile
