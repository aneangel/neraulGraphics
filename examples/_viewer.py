"""A small window for the examples, built on tkinter, which comes with Python: nothing to install.

    viewer = Viewer("title", width, height)
    while viewer.open:
        viewer.show(rgb_array)          # (height, width, 3) uint8; scale=N enlarges it N times
        dx, dy = viewer.take_look()     # mouse drag since the last call, in pixels
        if viewer.held("w"): ...        # keys currently down; viewer.on_key gets each new press

Viewer() raises RuntimeError when there is no display to open a window on, so an example can fall back to saving a file.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

try:
    import tkinter as tk
except ImportError:  # some Python builds leave it out
    tk = None


class Viewer:
    def __init__(self, title: str, width: int, height: int, scale: int = 1) -> None:
        if tk is None:
            raise RuntimeError("this Python has no tkinter, so no window can be opened")
        try:
            self.root = tk.Tk()
        except tk.TclError as error:  # no DISPLAY, or no X server to talk to
            raise RuntimeError(f"cannot open a window: {error}") from error
        self.title, self.scale = title, scale
        self.root.title(title)
        self.root.resizable(False, False)
        self.label = tk.Label(self.root, borderwidth=0, width=width * scale, height=height * scale)
        self.label.pack()
        self.open = True
        self.on_key: Callable[[str], None] | None = None  # called with the name of each key as it goes down
        self._down: set[str] = set()
        self._release_token: dict[str, int] = {}
        self._look = [0.0, 0.0]
        self._drag_from: tuple[int, int] | None = None
        self._photo = None
        self._frames, self._clock = 0, time.perf_counter()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.label.bind("<ButtonPress-1>", self._grab)
        self.label.bind("<B1-Motion>", self._drag)
        self.label.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_drag_from", None))
        self.root.focus_force()

    def close(self) -> None:
        """Closes the window and lets go of everything it held (call it when done, even if the window is still open)."""
        self.open = False
        self.on_key = None
        self._photo = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # --- input ---------------------------------------------------------------------------------------------------

    def _press(self, event) -> None:
        key = event.keysym.lower()
        self._release_token[key] = self._release_token.get(key, 0) + 1  # cancels a release that was only auto-repeat
        if key not in self._down:
            self._down.add(key)
            if self.on_key:
                self.on_key(key)

    def _release(self, event) -> None:
        # X11 auto-repeat sends a release and an immediate press for a key that is held; only a release that is
        # not followed by a press within a few milliseconds counts.
        key = event.keysym.lower()
        token = self._release_token.get(key, 0)

        def really_released() -> None:
            if self._release_token.get(key, 0) == token:
                self._down.discard(key)

        self.root.after(8, really_released)

    def _grab(self, event) -> None:
        self._drag_from = (event.x, event.y)

    def _drag(self, event) -> None:
        if self._drag_from is not None:
            self._look[0] += event.x - self._drag_from[0]
            self._look[1] += event.y - self._drag_from[1]
        self._drag_from = (event.x, event.y)

    def held(self, key: str) -> bool:
        return key in self._down

    def take_look(self) -> tuple[float, float]:
        """The mouse drag (dx, dy) in pixels since the last call."""
        look = (self._look[0], self._look[1])
        self._look = [0.0, 0.0]
        return look

    # --- output --------------------------------------------------------------------------------------------------

    def show(self, rgb, status: str = "") -> None:
        """Draws a (height, width, 3) uint8 image, handles events, and puts the frame rate in the title."""
        if not self.open:
            return
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.scale > 1:
            image = np.repeat(np.repeat(image, self.scale, axis=0), self.scale, axis=1)
        header = f"P6 {image.shape[1]} {image.shape[0]} 255\n".encode()
        try:
            self._photo = tk.PhotoImage(data=header + image.tobytes(), format="PPM")  # kept: Tk does not hold a reference
            self.label.configure(image=self._photo)
            self._frames += 1
            elapsed = time.perf_counter() - self._clock
            if elapsed >= 0.5:
                self.root.title(f"{self.title}  {self._frames / elapsed:.0f} fps" + (f"  {status}" if status else ""))
                self._frames, self._clock = 0, time.perf_counter()
            self.root.update()
        except tk.TclError:  # the window was closed under us
            self.open = False
