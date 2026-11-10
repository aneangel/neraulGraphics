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

Load a model with ``ng.load("model.glb")`` (also ``.gltf`` and ``.obj``) and call
``.renderer(width, height).render()`` on the result.

Triangle meshes are shared geometry that instances place in the world. Each mesh's
BVH is built once; moving instances afterwards only updates the small structure
over them, so a simulation can move thousands of bodies per frame::

    ball = ng.make_icosphere(1.0, subdivisions=3)
    bodies = [ng.Instance(0, ng.make_transform((x, 0, -10)), red) for x in range(-5, 6)]
    renderer = ng.Renderer(800, 450, [], lights, meshes=[ball], instances=bodies)
    renderer.set_instance_transforms(matrices)  # (N, 3, 4) float array, every frame

It traces on an NVIDIA GPU when there is one and on the CPU otherwise (``device="cpu"``
forces the CPU), drawing the same picture either way. Errors in the scene raise ValueError; a failure
inside CUDA raises neraulgraphics.CudaError.
"""

from .loaders import LoadedScene, SceneRenderer, load, load_gltf, load_obj
from ._core import (
    MAX_LIGHTS,
    MAX_SPHERES,
    OBJECT_FLOOR,
    OBJECT_INSTANCE,
    OBJECT_NONE,
    OBJECT_SPHERE,
    BatchRenderer,
    Camera,
    CudaError,
    Instance,
    Light,
    Material,
    Mesh,
    Plane,
    Renderer,
    Sphere,
    Texture,
    Transform,
    __version__,
    cuda_available,
    identity_transform,
    make_box,
    make_camera,
    make_icosphere,
    make_transform,
)

__all__ = [
    "LoadedScene",
    "SceneRenderer",
    "load",
    "load_gltf",
    "load_obj",
    "MAX_LIGHTS",
    "MAX_SPHERES",
    "OBJECT_FLOOR",
    "OBJECT_INSTANCE",
    "OBJECT_NONE",
    "OBJECT_SPHERE",
    "BatchRenderer",
    "Camera",
    "CudaError",
    "Instance",
    "Light",
    "Material",
    "Mesh",
    "Plane",
    "Renderer",
    "Sphere",
    "Texture",
    "Transform",
    "__version__",
    "cuda_available",
    "identity_transform",
    "make_box",
    "make_camera",
    "make_icosphere",
    "make_transform",
]
