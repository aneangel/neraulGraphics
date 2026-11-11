"""A rigid-body world that gives the same answer on every machine.

The solver is Jolt Physics built for cross-platform determinism (see python/physics.cpp). Fed the same bodies with the same dt, a
World produces the same bits on any CPU and with any number of threads, so a server and a client that run it agree, and a recorded
simulation replays exactly. state_hash() is a 64-bit fingerprint of every body's exact state, for checking that::

    from neraulgraphics import physics

    world = physics.World(threads=0)
    world.add_ground(friction=0.7)
    box = world.add_convex_hull(corners, position=(0, 0, 3), density=2400.0)   # corners: (n, 3) float32, relative to position
    world.finish_setup()
    for frame in range(240):
        world.step(1 / 120, substeps=4)
    print(world.poses()[box], hex(world.state_hash()))

Z is up; poses are (position, quaternion x y z w). The module is built only when the package is installed with
``-C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON`` (it fetches Jolt); available() says whether it is there.
"""

try:
    from ._physics import World

    _error = None
except ImportError as error:   # built without NERAULGRAPHICS_BUILD_PHYSICS
    World = None
    _error = error


def available() -> bool:
    """Whether the physics module was built into this installation."""
    return World is not None


def require():
    """The World class, or an ImportError that says how to get it."""
    if World is None:
        raise ImportError("neraulgraphics was installed without the deterministic physics module; reinstall it with "
                          "pip install -C cmake.define.NERAULGRAPHICS_BUILD_PHYSICS=ON neraulgraphics") from _error
    return World
