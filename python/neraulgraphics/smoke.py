"""Smoke and dust as a volume, drawn over a rendering.

The renderer draws surfaces, so a cloud of dust drawn as spheres looks like gravel. Smoke is turned into light absorption instead:
each particle is a soft Gaussian blob that adds optical depth tau where it lands, the blobs of overlapping particles add up, and
the picture is blended with the cloud's colour by 1 - exp(-tau) (the Beer-Lambert law), so a thin haze is transparent and a thick
core is opaque. A particle behind something solid is dropped, found by casting a ray from the camera to it, so rubble in front of
a cloud stays in front::

    smoke = ng.Smoke(*cam.source_size)
    image = smoke.composite(renderer.render(view, frame), view, positions, life, renderer)

positions are (n, 3) in the renderer's axes and life is (n,), 1 for a fresh particle and 0 for a gone one. numpy, or CuPy on the
GPU when it is there.
"""

import math

import numpy as np



def _array_module(device):
    """numpy, or CuPy when device asks for the GPU (or is "auto" and CuPy is there with a GPU to use)."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError('device is "auto", "cuda" or "cpu"')
    if device != "cpu":
        try:
            import cupy

            if cupy.cuda.runtime.getDeviceCount() > 0:
                return cupy
        except Exception:
            pass
        if device == "cuda":
            raise RuntimeError('device="cuda" needs CuPy and an NVIDIA GPU (pip install cupy-cuda12x)')
    return np


class Smoke:
    def __init__(self, width, height, device="auto", radius=0.9, growth=1.6, density=0.045, color=(0.50, 0.46, 0.41), max_pixels=20):
        """radius is how big a fresh particle is in metres and growth how much larger it swells; density is how much optical depth
        a particle adds at its centre; color is the cloud's colour; max_pixels caps a blob's size on screen."""
        self.width, self.height = int(width), int(height)
        self.xp = _array_module(device)
        self.radius, self.growth, self.density, self.color, self.max_pixels = radius, growth, density, color, int(max_pixels)

    def composite(self, image, camera, positions, life, renderer=None):
        """image (a (height, width, 3) uint8 rendering) with the smoke over it, as a numpy array. Pass the renderer to hide
        particles that are behind surfaces."""
        xp = self.xp
        positions = np.asarray(positions, np.float32)
        life = np.asarray(life, np.float32)
        live = life > 0.0
        if not live.any():
            return np.asarray(image)
        p, life = positions[live], life[live]
        eye = np.array(camera.position, np.float32)
        v = p - eye
        z = v @ np.array(camera.forward, np.float32)
        near = z > 0.4
        p, life, v, z = p[near], life[near], v[near], z[near]
        distance = np.linalg.norm(v, axis=1)
        size = self.radius * (1.0 + self.growth * (1.0 - life)) * np.minimum(1.0, life * 3.0)
        if renderer is not None and len(p):
            hit = renderer.cast_rays(np.tile(eye, (len(p), 1)), (v / distance[:, None]).astype(np.float32), float(distance.max() + 1.0))["distance"]
            visible = hit > distance - 0.6 * size
            p, life, v, z, distance, size = p[visible], life[visible], v[visible], z[visible], distance[visible], size[visible]
        if not len(p):
            return np.asarray(image)

        x = v @ np.array(camera.right, np.float32)
        y = v @ np.array(camera.up, np.float32)
        u = (x / (z * camera.fov_scale * camera.aspect) + 1.0) * 0.5 * self.width
        w = (1.0 - y / (z * camera.fov_scale)) * 0.5 * self.height
        focal = 0.5 * self.height / camera.fov_scale
        true_px = size * focal / z
        r_px = np.clip(true_px, 1.0, self.max_pixels)
        # a blob that had to be cut down to max_pixels stays as dark as it would have been, up to a limit
        weight = self.density * np.minimum(life * 2.0, 1.0) * np.minimum((true_px / r_px) ** 2, 4.0)
        k = int(min(math.ceil(2.0 * float(r_px.max())), 2 * self.max_pixels))
        offsets = xp.arange(-k, k + 1, dtype=xp.float32)

        u, w, r_px, weight = (xp.asarray(a, xp.float32) for a in (u, w, r_px, weight))
        cu, cv = xp.floor(u), xp.floor(w)
        dx = (offsets[None, :, None] + (cu - u)[:, None, None]) * xp.ones((1, 1, 2 * k + 1), xp.float32)
        dy = (offsets[None, None, :] + (cv - w)[:, None, None]) * xp.ones((1, 2 * k + 1, 1), xp.float32)
        sigma = (r_px * 0.55)[:, None, None]
        value = xp.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * weight[:, None, None]
        ix = (cu[:, None, None] + offsets[None, :, None] + 0 * dy).astype(xp.int32)
        iy = (cv[:, None, None] + offsets[None, None, :] + 0 * dx).astype(xp.int32)
        inside = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height) & (value > 1e-4)
        tau = xp.bincount((iy * self.width + ix)[inside], weights=value[inside], minlength=self.width * self.height)
        tau = tau.reshape(self.height, self.width).astype(xp.float32)

        alpha = 1.0 - xp.exp(-tau)
        base = xp.asarray(image, xp.float32)
        tint = xp.asarray(self.color, xp.float32) * 255.0 * (1.0 - 0.35 * alpha)[..., None]     # the thick core is darker: it shadows itself
        out = base * (1.0 - alpha)[..., None] + tint * alpha[..., None]
        out = xp.clip(out + 0.5, 0, 255).astype(xp.uint8)
        return out.get() if xp is not np else out
