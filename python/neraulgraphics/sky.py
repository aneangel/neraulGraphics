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
    is the irradiance the sun gives a surface facing it (a white diffuse surface facing it shows sun_strength / pi, which is the intensity of a Light that
    does the same), however large its disc: sun_radius (radians; the real sun's is 0.0046) sets how wide the disc is and so how
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


# sRGB (linear) from CIE XYZ, D65
_XYZ_TO_RGB = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])


def _perez(theta, gamma, a, b, c, d, e):
    """The Perez sky distribution: how much brighter or bluer a direction is than the zenith, from its angle to the zenith and to the sun."""
    return (1.0 + a * np.exp(b / np.maximum(np.cos(theta), 0.01))) * (1.0 + c * np.exp(d * gamma) + e * np.cos(gamma) ** 2)


def _sun_transmittance(elevation: float, turbidity: float) -> np.ndarray:
    """The fraction of the sun's red, green and blue that gets through the atmosphere at this elevation, relative to the zenith's."""
    def transmittance(e):
        air_mass = 1.0 / (math.sin(e) + 0.50572 * (math.degrees(e) + 6.07995) ** -1.6364)
        wavelengths = np.array([0.65, 0.55, 0.45])
        tau = 0.008735 * wavelengths ** -4.08 + (0.04608 * turbidity - 0.04586) * wavelengths ** -1.3
        return np.exp(-air_mass * tau)

    return transmittance(elevation) / transmittance(math.pi / 2)


def environment_frame(up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """The frame the renderer gives an Environment with this up: rows (right, up, forward) as unit vectors in the world. A direction (x, y, z) in a
    sky map's own layout (x along azimuth 0, y up, z along azimuth pi / 2) is the world direction x * right + y * up + z * forward. For up = +y
    that is right = (0, 0, -1) and forward = (1, 0, 0), not the world's own x and z, which is why a sun placed by azimuth needs this."""
    u = np.asarray(up, np.float64)
    u = u / np.linalg.norm(u)
    probe = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = np.cross(u, probe)
    r /= np.linalg.norm(r)
    return np.stack([r, u, np.cross(r, u)])


def sun_light(sun_elevation: float = 0.7, sun_azimuth: float = 0.0, turbidity: float = 2.5, sun_strength: float = 3.0, sun_radius: float = 0.0093,
              distance: float = 1000.0, up=(0.0, 1.0, 0.0)):
    """The sun as an explicit Light, to go with physical_sky(..., include_sun=False): the same direction, the same strength (sun_strength is
    the irradiance on a surface facing the sun, as in physical_sky; the Light's own intensity is that over pi, see below) and the same colour (the atmosphere's transmittance, so a low sun is redder). sun_radius is the disc's angular
    radius in radians and sets how soft shadow edges are; distance only has to be far. up is the Environment's up: the sun's azimuth is
    measured in the sky map's own frame (see environment_frame), so that the light sits where the map's sun would."""
    from ._core import Light

    if not 0.02 <= sun_elevation <= math.pi / 2:
        raise ValueError(f"sun_elevation must be between 0.02 and pi / 2, got {sun_elevation}")
    in_map = np.array([math.cos(sun_elevation) * math.cos(sun_azimuth), math.sin(sun_elevation), math.cos(sun_elevation) * math.sin(sun_azimuth)])
    toward = in_map @ environment_frame(up)                                    # x * right + y * up + z * forward
    colour = _sun_transmittance(sun_elevation, turbidity)
    # A Light's intensity is the radiance a white diffuse surface facing it shows, which is the irradiance over pi; a sky map's sun_strength is
    # the irradiance itself (a white surface facing it shows sun_strength / pi), so the same sun is a Light of intensity sun_strength / pi.
    return Light(tuple(float(v) for v in toward * distance), intensity=float(sun_strength) / math.pi, radius=float(distance * math.tan(sun_radius)),
                 color=tuple(float(c) for c in colour))


def physical_sky(
    sun_elevation: float = 0.7,
    sun_azimuth: float = 0.0,
    turbidity: float = 2.5,
    sun_strength: float = 3.0,
    sun_radius: float = 0.0093,
    sky_ratio: float = 0.2,
    ground_albedo: float = 0.25,
    width: int = 1024,
    height: int = 512,
    include_sun: bool = True,
) -> np.ndarray:
    """A sky map from the Preetham daylight model (1999): the colour and brightness of a clear sky for a given sun height and haziness.

    What it gets right that a plain gradient does not: the zenith is a deep blue and the horizon pale, the sky is brightest around the sun and
    along the horizon, a hazier sky (higher turbidity, 2 is very clear and 6 is hazy) is whiter and brighter near the horizon, and a low sun
    turns the sky and the sun itself orange and red, because light that crosses more air loses its blue.

    sun_elevation is the sun's angle above the horizon in radians (at least 0.02), sun_strength the irradiance the sun gives a surface facing it
    (a white diffuse surface facing it shows sun_strength / pi; a Light's intensity is that radiance, so sun_strength / pi is the intensity of
    the same sun as a Light; the sun disc's colour is the atmosphere's transmittance, so a low sun is dimmer and redder), sun_radius
    the disc's radius in radians (the real sun's is 0.0047, and a larger disc is found more reliably at few samples). sky_ratio is how much
    light the whole sky gives a horizontal surface, as a fraction of what the sun gives one facing it (about 0.15 to 0.3 for a clear sky).
    Below the horizon the map is the ground, a diffuse surface of ground_albedo lit by the sun and the sky. Returns linear radiance,
    (height, width, 3) float32, in the layout Environment(map=...) takes.

    include_sun=False leaves the sun's disc out of the map, for use with sun_light(), which adds the sun as an explicit light instead. That is
    the steadier choice for a moving camera: a highlight on a glossy surface is then computed from the light's direction and a shadow ray, where
    a sun in the map is found only by a random ray that happens to land on the disc, so the highlight twinkles as the surface moves under a
    pixel. The sky and ground light are the same either way.
    """
    if not 0.02 <= sun_elevation <= math.pi / 2:
        raise ValueError(f"sun_elevation must be between 0.02 and pi / 2, got {sun_elevation}")
    if not 1.7 <= turbidity <= 10.0:
        raise ValueError(f"turbidity must be between 1.7 and 10, got {turbidity}")
    if not 0.0 < sun_radius < math.pi / 4 or sun_strength < 0.0 or sky_ratio < 0.0 or not 0.0 <= ground_albedo <= 1.0:
        raise ValueError("sun_radius must be in (0, pi / 4); sun_strength and sky_ratio must not be negative; ground_albedo is 0 to 1")
    if width < 8 or height < 4:
        raise ValueError(f"the map must be at least 8x4, got {width}x{height}")

    T = turbidity
    theta_s = math.pi / 2 - sun_elevation                         # the sun's angle from the zenith
    # Perez coefficients (A to E) for the luminance Y and the chromaticities x and y
    coeff = {
        "Y": (0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703),
        "x": (-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452),
        "y": (-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529),
    }
    t2, ts2, ts3 = T * T, theta_s ** 2, theta_s ** 3
    chi = (4.0 / 9.0 - T / 120.0) * (math.pi - 2.0 * theta_s)
    zenith_Y = (4.0453 * T - 4.9710) * math.tan(chi) - 0.2155 * T + 2.4192                    # kcd/m^2
    zenith_x = (t2 * (0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * theta_s) + T * (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * theta_s + 0.00394)
                + (0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * theta_s + 0.25886))
    zenith_y = (t2 * (0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * theta_s) + T * (-0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * theta_s + 0.00516)
                + (0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * theta_s + 0.26688))

    theta = (np.arange(height) + 0.5) / height * math.pi            # from straight up
    phi = (np.arange(width) + 0.5) / width * 2.0 * math.pi
    sin_t, cos_t = np.sin(theta)[:, None], np.cos(theta)[:, None]
    toward = np.stack([sin_t * np.cos(phi)[None, :], np.broadcast_to(cos_t, (height, width)), sin_t * np.sin(phi)[None, :]], axis=-1)
    sun = np.array([math.cos(sun_elevation) * math.cos(sun_azimuth), math.sin(sun_elevation), math.cos(sun_elevation) * math.sin(sun_azimuth)])
    gamma = np.arccos(np.clip(toward @ sun, -1.0, 1.0))             # the angle to the sun
    th = np.broadcast_to(np.minimum(theta[:, None], math.pi / 2 - 1e-3), (height, width))

    def channel(name, zenith):
        a, b, c, d, e = coeff[name]
        return zenith * _perez(th, gamma, a, b, c, d, e) / _perez(0.0, theta_s, a, b, c, d, e)

    Y, x, y = channel("Y", zenith_Y), channel("x", zenith_x), channel("y", zenith_y)
    X = x / np.maximum(y, 1e-6) * Y
    Z = (1.0 - x - y) / np.maximum(y, 1e-6) * Y
    sky = np.clip(np.stack([X, Y, Z], axis=-1) @ _XYZ_TO_RGB.T, 0.0, None)

    # Scale so that the whole sky lights a horizontal surface with sky_ratio of the sun's strength: that fixes the sky's brightness against
    # the sun's, whatever units the model's luminance is in.
    solid_angle = (2.0 * math.pi / width) * (math.pi / height) * np.broadcast_to(sin_t, (height, width))
    upper = (cos_t > 0.0)
    dome = (sky.mean(axis=-1) * np.broadcast_to(np.maximum(cos_t, 0.0), (height, width)) * solid_angle * upper).sum()
    sky_scale = sun_strength * sky_ratio / max(dome, 1e-9)
    sky *= sky_scale

    # The ground: a diffuse surface lit by the sun and the sky, seen from above.
    ground_light = sun_strength * math.sin(sun_elevation) + sun_strength * sky_ratio
    below = np.broadcast_to((cos_t < 0.0)[..., None], (height, width, 1))
    image = np.where(below, np.array([ground_albedo * ground_light / math.pi] * 3), sky)

    # The sun: the atmosphere's transmittance along the slant path (Kasten and Young's air mass; Rayleigh and aerosol optical depths for red,
    # green and blue), relative to the zenith's so that a sun at the zenith has exactly sun_strength.
    sun_colour = _sun_transmittance(sun_elevation, T)
    disc = toward @ sun > math.cos(sun_radius)
    if include_sun and disc.any() and sun_strength > 0.0:
        texel_irradiance = float((solid_angle * (toward @ sun))[disc].sum())
        image[disc] = sun_strength * sun_colour / texel_irradiance
    return np.ascontiguousarray(image.astype(np.float32))
