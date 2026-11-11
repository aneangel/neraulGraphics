"""A real-time window for the examples, built on pyglet: each frame goes to the GPU as a texture, so it is fast at any size.

    with LiveWindow("title", 1280, 720) as window:
        camera = FlyCamera(position=(0, 2, 5))
        while window.running:
            dt = window.tick()                          # handles events; seconds since the last call
            camera.update(window, dt)                   # W A S D move, Q E down and up, mouse looks (click to capture it)
            for key in window.take_keys(): ...          # names of the keys pressed since the last call: "space", "r", "1", "left"
            window.show(image, hud="text")              # (height, width, 3) uint8

LiveWindow() raises RuntimeError when there is no display (or no pyglet), so an example can fall back to writing a file; use
open_window() to get None instead, and finish_headless() to write the picture the example would have shown.
"""

from __future__ import annotations

import math
import time

import numpy as np

import neraulgraphics as ng


class LiveWindow:
    def __init__(self, title: str, width: int, height: int, scale: int = 1, vsync: bool = False) -> None:
        try:
            import pyglet
            from pyglet import gl
        except ImportError as error:
            raise RuntimeError("the live window needs pyglet: pip install pyglet") from error
        try:
            self.window = pyglet.window.Window(width * scale, height * scale, title, vsync=vsync)
        except Exception as error:   # no display, no GL
            raise RuntimeError(f"cannot open a window: {error}") from error
        self._pyglet, self._gl = pyglet, gl
        self.title, self.width, self.height, self.scale = title, width, height, scale
        self.texture = self._make_texture(width, height)
        self.hud = pyglet.text.Label("", x=12, y=height * scale - 12, anchor_y="top", multiline=True, width=width * scale - 24,
                                     font_size=11, color=(255, 255, 255, 255))
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.captured = False
        self.mouse_mode = "fly"        # "fly": click captures the mouse and it looks; "orbit": dragging looks and the wheel zooms
        self._pressed: list[str] = []
        self._clicks: list[str] = []
        self._look = [0.0, 0.0]
        self._scroll = 0.0
        self._quit = False
        self._last_image: np.ndarray | None = None
        self._last, self._clock, self._frames = time.perf_counter(), time.perf_counter(), 0
        self._bind()

    # ------------------------------------------------------------------------------------------------ plumbing --

    def _make_texture(self, width: int, height: int):
        texture = self._pyglet.image.Texture.create(width, height, internalformat=self._gl.GL_RGB8)
        self._gl.glBindTexture(texture.target, texture.id)
        self._gl.glTexParameteri(texture.target, self._gl.GL_TEXTURE_MAG_FILTER, self._gl.GL_LINEAR)
        self._gl.glTexParameteri(texture.target, self._gl.GL_TEXTURE_MIN_FILTER, self._gl.GL_LINEAR)
        return texture

    def _bind(self) -> None:
        pyglet, window = self._pyglet, self.window
        key, mouse = pyglet.window.key, pyglet.window.mouse

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                if self.captured:
                    self.release()
                else:
                    self._quit = True
                return True
            self._pressed.append(key.symbol_string(symbol).lower().lstrip("_"))

        @window.event
        def on_mouse_press(x, y, button, modifiers):
            if self.mouse_mode == "fly" and not self.captured:
                window.set_exclusive_mouse(True)
                self.captured = True
            elif button == mouse.LEFT:
                self._clicks.append("left")
            elif button == mouse.RIGHT:
                self._clicks.append("right")

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            if self.captured:
                self._look[0] += dx
                self._look[1] += dy

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if self.captured or self.mouse_mode == "orbit":
                self._look[0] += dx
                self._look[1] += dy

        @window.event
        def on_mouse_scroll(x, y, sx, sy):
            self._scroll += sy

        @window.event
        def on_draw():
            self._draw()

        @window.event
        def on_close():
            self._quit = True

    def _draw(self) -> None:
        gl = self._gl
        self.window.clear()
        if self._last_image is not None:
            image = self._last_image
            gl.glBindTexture(self.texture.target, self.texture.id)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexSubImage2D(self.texture.target, 0, 0, 0, image.shape[1], image.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image.ctypes.data)
            self.texture.blit(0, self.window.height, 0, self.window.width, -self.window.height)      # rows arrive top first
        if self.hud.text:
            self.hud.draw()

    # ------------------------------------------------------------------------------------------------ per frame --

    @property
    def running(self) -> bool:
        return not self._quit

    def tick(self) -> float:
        """Handles the window's events and returns the seconds since the last tick (at most 0.1, so a stall is not a jump)."""
        self.window.dispatch_events()
        now = time.perf_counter()
        dt = min(now - self._last, 0.1)
        self._last = now
        return dt

    def show(self, image: np.ndarray, hud: str | None = None) -> None:
        """Draws an (height, width, 3) uint8 picture, with a line or more of text over it."""
        image = np.ascontiguousarray(image)
        if image.shape[1] != self.texture.width or image.shape[0] != self.texture.height:
            self.texture = self._make_texture(image.shape[1], image.shape[0])
        self._last_image = image
        if hud is not None:
            self.hud.text = hud
        self.window.switch_to()
        self._draw()
        self.window.flip()
        self._frames += 1
        if self._frames % 20 == 0:
            self.window.set_caption(f"{self.title}   {self._frames / (time.perf_counter() - self._clock):.0f} fps")

    def held(self, name: str) -> bool:
        """Whether a key is down now: held("w"), held("lshift"), held("space")."""
        return bool(self.keys[getattr(self._pyglet.window.key, name.upper() if not name[0].isdigit() else "_" + name)])

    def take_keys(self) -> list[str]:
        keys, self._pressed = self._pressed, []
        return keys

    def take_clicks(self) -> list[str]:
        clicks, self._clicks = self._clicks, []
        return clicks

    def take_look(self) -> tuple[float, float]:
        look, self._look = tuple(self._look), [0.0, 0.0]
        return look

    def take_scroll(self) -> float:
        scroll, self._scroll = self._scroll, 0.0
        return scroll

    def release(self) -> None:
        self.window.set_exclusive_mouse(False)
        self.captured = False

    def screenshot(self, path: str) -> None:
        if self._last_image is not None:
            from PIL import Image

            Image.fromarray(self._last_image).save(path)
            print("wrote", path)

    def close(self) -> None:
        try:
            self.window.close()
        except Exception:
            pass

    def __enter__(self) -> "LiveWindow":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_window(title: str, width: int, height: int, scale: int = 1) -> LiveWindow | None:
    """A LiveWindow, or None (after saying why) when there is no display to open one on."""
    try:
        return LiveWindow(title, width, height, scale)
    except RuntimeError as problem:
        print(problem, "- no window, so the example will write a picture instead")
        return None


# ------------------------------------------------------------------------------------------------------ cameras --


class FlyCamera:
    """A free camera: W A S D move along where it looks, Q and E go down and up, shift is faster, the mouse turns it.

    yaw 0 looks down -z (the way the examples lay their scenes out), pitch up is positive. moved says whether the last update() changed
    the view, so a renderer that refines a still picture knows when to start again."""

    def __init__(self, position, yaw: float = 0.0, pitch: float = 0.0, speed: float = 5.0, sensitivity: float = 0.0025) -> None:
        self.position = np.asarray(position, np.float64).copy()
        self.yaw, self.pitch, self.speed, self.sensitivity = yaw, pitch, speed, sensitivity
        self.moved = True

    def forward(self) -> np.ndarray:
        return np.array([math.cos(self.pitch) * math.sin(self.yaw), math.sin(self.pitch), -math.cos(self.pitch) * math.cos(self.yaw)])

    def right(self) -> np.ndarray:
        r = np.cross(self.forward(), (0.0, 1.0, 0.0))
        return r / np.linalg.norm(r)

    def update(self, window: LiveWindow, dt: float) -> bool:
        dx, dy = window.take_look()
        before = (self.position.copy(), self.yaw, self.pitch)
        self.yaw += dx * self.sensitivity
        self.pitch = float(np.clip(self.pitch + dy * self.sensitivity, -1.5, 1.5))
        speed = self.speed * (4.0 if window.held("lshift") else 1.0) * dt
        move = ((window.held("w") - window.held("s")) * self.forward() + (window.held("d") - window.held("a")) * self.right()
                + (window.held("e") - window.held("q")) * np.array([0.0, 1.0, 0.0]))
        self.position = self.position + move * speed
        self.moved = bool(dx or dy or np.any(move))
        return self.moved

    def camera(self, width: int, height: int, fov: float = 0.9):
        return ng.make_camera(tuple(float(v) for v in self.position), tuple(float(v) for v in self.position + self.forward()), width=width, height=height, fov=fov)


def up_frame(up) -> np.ndarray:
    """A right-handed frame (x, y, z) as the rows of a 3 x 3 matrix, with z along up: the axes a simulation in which up is z would use.
    For up = +y it is x = (1, 0, 0), y = (0, 0, -1), the same axes as the examples' simulations (see _stage.SIM_TO_RENDER)."""
    z = np.asarray(up, np.float64)
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = ref - z * np.dot(ref, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


class OrbitCamera:
    """A camera on a sphere round a point: drag to turn, the wheel to zoom. up is the scene's up axis (any direction)."""

    def __init__(self, center, distance: float, yaw: float = 0.6, pitch: float = 0.25, up=(0.0, 1.0, 0.0)) -> None:
        self.center = np.asarray(center, np.float64)
        self.distance, self.yaw, self.pitch = distance, yaw, pitch
        self.up = np.asarray(up, np.float64)
        self.moved = True

    def position(self) -> np.ndarray:
        x, y, z = up_frame(self.up)                     # yaw 0 looks from the -y side of the frame, towards +y (for up = +y: from +z, along -z)
        offset = math.cos(self.pitch) * math.sin(self.yaw) * x - math.cos(self.pitch) * math.cos(self.yaw) * y + math.sin(self.pitch) * z
        return self.center + self.distance * offset

    def update(self, window: LiveWindow, dt: float) -> bool:
        dx, dy = window.take_look()
        scroll = window.take_scroll()
        self.yaw -= dx * 0.006
        self.pitch = float(np.clip(self.pitch + dy * 0.006, -1.4, 1.4))
        self.distance = float(np.clip(self.distance * (0.9 ** scroll), 0.2, 500.0))
        self.moved = bool(dx or dy or scroll)
        return self.moved

    def camera(self, width: int, height: int, fov: float = 0.9):
        return ng.make_camera(tuple(float(v) for v in self.position()), tuple(float(v) for v in self.center), width=width, height=height,
                              world_up=tuple(float(v) for v in self.up), fov=fov)


# ------------------------------------------------------------------------------------------------------- helpers --


def save_picture(path: str, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(path)
    print("wrote", path)
