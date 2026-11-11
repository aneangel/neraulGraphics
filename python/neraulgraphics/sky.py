"""Sky maps for path tracing: load an HDR image, or make a sky with a sun.

Both return an equirectangular ``(height, width, 3)`` float32 array of linear radiance, which is
what ``Environment(map=...)`` takes: rows go from straight up (the top) to straight down, and the
columns go once round the up axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_hdr(path: str | Path) -> np.ndarray:
    """Reads a Radiance ``.hdr`` (RGBE) image as a ``(height, width, 3)`` float32 array of linear radiance.

    Most free HDR skies (Poly Haven and others) are equirectangular images in this format, and can be
    passed straight to ``Environment(map=...)``. Raises ValueError for a file that is not one.
    """
    data = Path(path).read_bytes()
    position = 0

    def line() -> bytes:
        nonlocal position
        end = data.find(b"\n", position)
        if end < 0:
            raise ValueError(f"{path}: not a Radiance HDR file (the header ends too soon)")
        text = data[position:end]
        position = end + 1
        return text

    if not line().startswith(b"#?"):
        raise ValueError(f"{path}: not a Radiance HDR file (no #? signature)")
    while True:  # header variables, up to the blank line
        text = line()
        if text.strip() == b"":
            break
        if text.startswith(b"FORMAT=") and b"32-bit_rle_rgbe" not in text:
            raise ValueError(f"{path}: only the 32-bit_rle_rgbe format is supported, got {text.decode(errors='replace')}")
    resolution = line().split()
    if len(resolution) != 4 or resolution[0] not in (b"-Y", b"+Y") or resolution[2] not in (b"+X", b"-X"):
        raise ValueError(f"{path}: unsupported image orientation {b' '.join(resolution).decode(errors='replace')!r}")
    height, width = int(resolution[1]), int(resolution[3])
    if height < 1 or width < 1:
        raise ValueError(f"{path}: the image is {width}x{height}")

    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if 8 <= width < 32768 and data[position : position + 2] == b"\x02\x02" and not data[position + 2] & 0x80:
            if (data[position + 2] << 8 | data[position + 3]) != width:
                raise ValueError(f"{path}: a scanline's width does not match the image's")
            position += 4
            for channel in range(4):  # each channel is run-length coded on its own
                x = 0
                while x < width:
                    if position >= len(data):
                        raise ValueError(f"{path}: the file ends in the middle of the image")
                    count = data[position]
                    position += 1
                    if count > 128:  # a run of one repeated value
                        count -= 128
                        if x + count > width:
                            raise ValueError(f"{path}: a run goes past the end of its scanline")
                        rgbe[y, x : x + count, channel] = data[position]
                        position += 1
                    else:  # that many values, as they are
                        if count == 0 or x + count > width:
                            raise ValueError(f"{path}: a bad run length in a scanline")
                        rgbe[y, x : x + count, channel] = np.frombuffer(data, np.uint8, count, position)
                        position += count
                    x += count
        else:  # flat: four bytes a pixel
            end = position + 4 * width
            if end > len(data):
                raise ValueError(f"{path}: the file ends in the middle of the image")
            rgbe[y] = np.frombuffer(data, np.uint8, 4 * width, position).reshape(width, 4)
            position = end

    exponent = rgbe[..., 3].astype(np.int32)
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - (128 + 8)), 0.0).astype(np.float32)
    image = rgbe[..., :3].astype(np.float32) * scale[..., None]
    if resolution[0] == b"+Y":  # stored bottom to top
        image = image[::-1]
    if resolution[2] == b"-X":
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


def procedural_sky(
    sun_elevation: float = 0.9,
    sun_azimuth: float = 0.0,
    sun_strength: float = 3.0,
    sun_radius: float = 0.03,
    zenith_color: tuple[float, float, float] = (0.12, 0.30, 0.80),
    horizon_color: tuple[float, float, float] = (0.70, 0.80, 0.95),
    ground_color: tuple[float, float, float] = (0.30, 0.27, 0.24),
    width: int = 512,
    height: int = 256,
) -> np.ndarray:
    """Makes a sky map: a blue gradient, a ground colour below the horizon, and a sun.

    sun_elevation is the sun's angle above the horizon in radians (0 to pi / 2) and sun_azimuth its
    angle round the up axis (turn the whole sky later with ``Environment(rotation=...)``). sun_strength
    is the light the sun gives a surface facing it, in the same units as a Light's intensity, however
    large its disc: sun_radius (radians; the real sun's is 0.0046) sets how wide the disc is and so how
    soft the shadows are. A larger disc is also found more reliably by the sampler at few samples.
    """
    if not (0.0 <= sun_elevation <= math.pi / 2):
        raise ValueError(f"sun_elevation must be between 0 and pi / 2, got {sun_elevation}")
    if not 0.0 < sun_radius < math.pi / 2:
        raise ValueError(f"sun_radius must be in (0, pi / 2), got {sun_radius}")
    if sun_strength < 0.0:
        raise ValueError(f"sun_strength must not be negative, got {sun_strength}")
    if width < 8 or height < 4:
        raise ValueError(f"the map must be at least 8x4, got {width}x{height}")

    theta = (np.arange(height) + 0.5) / height * math.pi  # angle from straight up
    phi = (np.arange(width) + 0.5) / width * 2.0 * math.pi
    sin_t, cos_t = np.sin(theta)[:, None], np.cos(theta)[:, None]
    toward = np.stack([sin_t * np.cos(phi)[None, :], np.broadcast_to(cos_t, (height, width)),
                       sin_t * np.sin(phi)[None, :]], axis=-1)  # (height, width, 3): x along phi = 0, y up, z along phi = pi / 2

    up = np.clip(cos_t, 0.0, 1.0)  # 1 straight up, 0 at the horizon
    sky = np.array(horizon_color) * (1.0 - up[..., None]) + np.array(zenith_color) * up[..., None]
    sky = np.broadcast_to(sky, (height, width, 3))
    below = np.broadcast_to((cos_t < 0.0)[..., None], (height, width, 1))
    image = np.where(below, np.array(ground_color), sky).astype(np.float64)

    sun = np.array([math.cos(sun_elevation) * math.cos(sun_azimuth), math.sin(sun_elevation),
                    math.cos(sun_elevation) * math.sin(sun_azimuth)])
    disc = toward @ sun > math.cos(sun_radius)
    if disc.any() and sun_strength > 0.0:
        # The texels' solid angle, so that what they add up to is exactly the strength asked for.
        texel = (2.0 * math.pi / width) * (math.pi / height) * np.broadcast_to(sin_t, (height, width))
        irradiance_per_radiance = float((texel * (toward @ sun))[disc].sum())  # what unit radiance in the disc gives head-on
        image[disc] = sun_strength / irradiance_per_radiance
    return np.ascontiguousarray(image.astype(np.float32))
