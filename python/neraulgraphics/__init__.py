"""GPU raytracing from Python.

Build a scene from spheres, lights and an optional floor, make a Renderer, and
call render() to get a (height, width, 3) uint8 RGB numpy array::

    import neraulgraphics as ng

    red = ng.Material(diffuse_color=(0.5, 0.1, 0.1))
    renderer = ng.Renderer(800, 450, spheres=[ng.Sphere((0, 0, -10), 2, red)],
                           lights=[ng.Light((-10, 20, 10), intensity=1.5)])
    camera = ng.make_camera((0, 1, 0), (0, 0, -10), width=800, height=450)
    for frame in range(64):  # more frames, smoother image
        image = renderer.render(camera, frame)

Needs an NVIDIA GPU and driver. Errors in the scene raise ValueError; a failure
inside CUDA raises neraulgraphics.CudaError.
"""

from ._core import (
    MAX_LIGHTS,
    MAX_SPHERES,
    Camera,
    CudaError,
    Light,
    Material,
    Plane,
    Renderer,
    Sphere,
    __version__,
    make_camera,
)

__all__ = [
    "MAX_LIGHTS",
    "MAX_SPHERES",
    "Camera",
    "CudaError",
    "Light",
    "Material",
    "Plane",
    "Renderer",
    "Sphere",
    "__version__",
    "make_camera",
]
