"""Tests for the Newton and MuJoCo adapters. Each skips unless its simulator is installed,
and rendering needs an NVIDIA GPU.

The core check is the same for both: an adapter must draw exactly what a hand-built
Renderer draws for the same shapes, poses, lights and camera. That fails if the adapter
picks the wrong mesh, scale, pose composition or up axis.
"""

import numpy as np
import pytest

import neraulgraphics as ng
from neraulgraphics import shapes

W, H = 160, 90
LIGHTS = [ng.Light((4, -6, 8), intensity=1.0)]
RED = (0.8, 0.1, 0.1)


@pytest.fixture(scope="module")
def gpu():
    try:
        ng.Renderer(8, 8, [], [])
    except ng.CudaError as e:
        pytest.skip(f"no usable CUDA GPU: {e}")


def material(color):
    return ng.Material(diffuse_color=color, diffuse_albedo=0.9, specular_albedo=0.2, reflectivity=0.0,
                       specular_exponent=40.0)


def hand_built(meshes, instances, matrices):
    """A plain Renderer for [(mesh index, color)] placed by (N, 3, 4) matrices."""
    r = ng.Renderer(W, H, [], LIGHTS, meshes=meshes,
                    instances=[ng.Instance(m, ng.Transform(np.eye(4, dtype=np.float32)), material(c))
                               for m, c in instances])
    r.set_instance_transforms(matrices)
    return r


def center_pixel(image):
    return image[H // 2, W // 2].astype(int)


def is_red(pixel):
    return pixel[0] > pixel[1] + 40 and pixel[0] > pixel[2] + 40


# ------------------------------------------------------------------- Newton --

@pytest.fixture(scope="module")
def nt():
    newton = pytest.importorskip("newton")
    import warp as wp
    wp.config.quiet = True
    return newton, wp


@pytest.fixture(scope="module")
def mj():
    return pytest.importorskip("mujoco")


class TestNewton:
    def model_with(self, nt, add, up_axis=None):
        newton, wp = nt
        builder = newton.ModelBuilder(up_axis=up_axis) if up_axis is not None else newton.ModelBuilder()
        add(builder, wp)
        return builder.finalize()

    def test_every_primitive_is_drawn(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer

        def add(b, wp):
            b.add_ground_plane()
            for i, shape in enumerate(["sphere", "box", "capsule", "cylinder", "cone", "ellipsoid"]):
                body = b.add_body(xform=wp.transform(p=wp.vec3(float(i), 0.0, 1.0), q=wp.quat_identity()))
                {"sphere": lambda: b.add_shape_sphere(body, radius=0.4),
                 "box": lambda: b.add_shape_box(body, hx=0.4, hy=0.3, hz=0.2),
                 "capsule": lambda: b.add_shape_capsule(body, radius=0.2, half_height=0.4),
                 "cylinder": lambda: b.add_shape_cylinder(body, radius=0.3, half_height=0.4),
                 "cone": lambda: b.add_shape_cone(body, radius=0.3, half_height=0.4),
                 "ellipsoid": lambda: b.add_shape_ellipsoid(body, rx=0.4, ry=0.3, rz=0.2)}[shape]()

        view = NewtonRenderer(self.model_with(nt, add), W, H)
        assert view.instance_count == 7  # the six shapes and the ground plane
        assert view.render().std() > 0

    def test_matches_a_hand_built_renderer(self, nt, gpu):
        """A box on a rotated body with an offset shape: pose composition, scale and up axis all matter."""
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt
        angle = 0.7
        body_q = wp.quat(0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2)))

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.5, 0.2, 1.0), q=body_q))
            b.add_shape_box(body, xform=wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity()),
                            hx=0.5, hy=0.3, hz=0.2)

        model = self.model_with(nt, add)
        view = NewtonRenderer(model, W, H, lights=LIGHTS, ambient=0.0)  # no ambient, to match the hand-built one
        camera = view.camera((0.5, -4.0, 2.0), (0.8, 0.2, 1.0))

        # By hand: the shape sits 0.3 along the body's x axis, which the body turned by `angle` about z.
        position = np.array([[0.5 + 0.3 * np.cos(angle), 0.2 + 0.3 * np.sin(angle), 1.0]])
        matrix = shapes.pose_matrices(position, np.array([[0, 0, np.sin(angle / 2), np.cos(angle / 2)]]),
                                      np.array([[0.5, 0.3, 0.2]]))
        by_hand = hand_built([ng.make_box((1, 1, 1))], [(0, tuple(model.shape_color.numpy()[0]))], matrix)
        want = by_hand.render(camera)
        got = view.render(model.state(), camera)
        assert np.abs(got.astype(int) - want.astype(int)).mean() < 0.01
        assert got.std() > 0

    def test_follows_the_state_not_the_model(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)

        model = self.model_with(nt, add)
        view = NewtonRenderer(model, W, H, lights=LIGHTS)
        camera = view.camera((0.0, -5.0, 1.0), (0.0, 0.0, 1.0))
        state = model.state()
        before = view.render(state, camera)

        moved = state.body_q.numpy()
        moved[0, 0] = 1.5  # slide the body along x
        state.body_q.assign(moved)
        after = view.render(state, camera)
        assert not np.array_equal(before, after)

        # The same scene built with the body already at x=1.5 draws the same picture.
        def add_moved(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(1.5, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)

        fresh = NewtonRenderer(self.model_with(nt, add_moved), W, H, lights=LIGHTS)
        assert np.abs(after.astype(int) - fresh.render(None, camera).astype(int)).mean() < 0.01

    def test_shapes_that_are_not_visible_are_skipped(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)
            hidden = newton.ModelBuilder.ShapeConfig(is_visible=False)  # a collision-only shape
            b.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2, cfg=hidden)

        assert NewtonRenderer(self.model_with(nt, add), W, H).instance_count == 1

    def test_ground_plane_is_a_checkerboard(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            b.add_ground_plane()
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.4)

        view = NewtonRenderer(self.model_with(nt, add), W, H, lights=LIGHTS)
        assert view.instance_count == 2  # the sphere and the ground
        image = view.render(None, view.camera((0.0, -5.0, 3.0), (0.0, 2.0, 0.0))).astype(int)
        ground = image[H * 3 // 4:, :, 1]  # the lower quarter of the picture is all ground
        assert ground.max() - ground.min() > 15  # light and dark squares, not one flat colour

    def test_ambient_lights_the_far_side(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)

        model = self.model_with(nt, add)
        behind = [ng.Light((0, 8, 1), intensity=1.0)]  # the camera looks at the side facing away from this
        dark = NewtonRenderer(model, W, H, lights=behind, ambient=0.0)
        lit = NewtonRenderer(model, W, H, lights=behind)  # default ambient
        assert lit.renderer.ambient == pytest.approx(0.2)
        camera = dark.camera((0, -4, 1), (0, 0, 1))
        assert center_pixel(dark.render(None, camera)).max() < 12
        assert center_pixel(lit.render(None, camera)).max() > 40

    def test_sensor_images_name_the_shapes_and_follow_the_state(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)
            b.add_ground_plane()

        model = self.model_with(nt, add)
        view = NewtonRenderer(model, W, H, lights=LIGHTS)
        camera = view.camera((0.0, -5.0, 1.0), (0.0, 0.0, 1.0))
        sphere_id = view.object_id_of_shape[0]
        out = view.render_frame(model.state(), camera, rgb=False, depth=True, object_ids=True)
        assert out["object_ids"][H // 2, W // 2] == sphere_id
        assert out["depth"][H // 2, W // 2] == pytest.approx(4.5, abs=0.05)  # the sphere's near side

        # Move the body, then ask a ray query without drawing: update() puts the shapes where the state says.
        state = model.state()
        moved = state.body_q.numpy()
        moved[0, 1] = 2.0  # slide along y, away from the camera
        state.body_q.assign(moved)
        view.update(state)
        origin, direction = np.array([[0.0, -5.0, 1.0]], dtype=np.float32), np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        hit = view.cast_rays(origin, direction)
        assert hit["object_id"][0] == sphere_id and hit["distance"][0] == pytest.approx(6.5, abs=1e-3)

    def test_up_axis_is_respected(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt

        def add(b, wp):
            body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 1.0, 0.0), q=wp.quat_identity()))
            b.add_shape_sphere(body, radius=0.5)

        view = NewtonRenderer(self.model_with(nt, add, up_axis=newton.Axis.Y), W, H)
        assert np.allclose(view.up, [0, 1, 0])
        assert view.render().std() > 0

    def test_world_offsets_separate_batched_worlds(self, nt, gpu):
        from neraulgraphics.newton import NewtonRenderer
        newton, wp = nt
        one = newton.ModelBuilder()
        body = one.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()))
        one.add_shape_sphere(body, radius=0.5)
        many = newton.ModelBuilder()
        many.replicate(one, world_count=3)
        model = many.finalize()

        stacked = NewtonRenderer(model, W, H, lights=LIGHTS)
        spread = NewtonRenderer(model, W, H, lights=LIGHTS, world_offsets=[[0, 0, 0], [3, 0, 0], [6, 0, 0]])
        assert stacked.instance_count == spread.instance_count == 3
        assert np.allclose(stacked._matrices(None)[:, :, 3], stacked._matrices(None)[0, :, 3])  # all on top of each other
        xs = spread._matrices(None)[:, 0, 3]
        assert np.allclose(np.sort(xs) - xs.min(), [0, 3, 6])


# ------------------------------------------------------------------- MuJoCo --

class TestMuJoCo:
    def model(self, mj, body_xml, assets=""):
        xml = f"<mujoco><asset>{assets}</asset><worldbody>{body_xml}</worldbody></mujoco>"
        m = mj.MjModel.from_xml_string(xml)
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        return m, d

    def test_matches_a_hand_built_renderer(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        # A tilted body holding an offset ellipsoid: MuJoCo hands over rotation matrices, not quaternions.
        m, d = self.model(mj, '<body pos="0.2 0 1" euler="20 35 10"><geom type="ellipsoid" pos="0.3 0 0" '
                              'size="0.5 0.3 0.2" rgba="0.8 0.1 0.1 1"/></body>')
        view = MuJoCoRenderer(m, W, H, lights=LIGHTS, ambient=0.0)  # no ambient, to match the hand-built one
        camera = view.camera((0.5, -4.0, 2.0), (0.4, 0.0, 1.0))
        matrix = shapes.matrices_from_rotations(d.geom_xmat.reshape(-1, 3, 3), d.geom_xpos, np.array([[0.5, 0.3, 0.2]]))
        want = hand_built([ng.make_icosphere(1.0, 3)], [(0, RED)], matrix).render(camera)
        got = view.render(d, camera)
        assert np.abs(got.astype(int) - want.astype(int)).mean() < 0.01
        assert got.std() > 0

    def test_every_primitive_and_a_mesh_are_drawn(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        m, d = self.model(
            mj,
            '<geom type="plane" size="0 0 0.1"/>'
            '<geom type="sphere" size="0.3" pos="0 0 1"/><geom type="box" size="0.2 0.3 0.1" pos="1 0 1"/>'
            '<geom type="capsule" size="0.1 0.3" pos="2 0 1"/><geom type="cylinder" size="0.2 0.3" pos="3 0 1"/>'
            '<geom type="ellipsoid" size="0.3 0.2 0.1" pos="4 0 1"/><geom type="mesh" mesh="tet" pos="5 0 1"/>',
            '<mesh name="tet" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>')
        view = MuJoCoRenderer(m, W, H)
        assert view.instance_count == 7
        assert view.render(d).std() > 0

    def test_hidden_groups_and_clear_geoms_are_skipped(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        m, _ = self.model(mj, '<geom type="sphere" size="0.3"/><geom type="sphere" size="0.3" group="4"/>'
                              '<geom type="sphere" size="0.3" rgba="1 0 0 0"/>')
        assert MuJoCoRenderer(m, W, H).instance_count == 1
        assert MuJoCoRenderer(m, W, H, geom_groups=(0, 4)).instance_count == 2

    def test_colour_comes_from_rgba_or_the_material(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        for body in ('<geom type="sphere" size="0.5" pos="0 0 1" rgba="0.8 0.1 0.1 1"/>',
                     '<geom type="sphere" size="0.5" pos="0 0 1" material="red"/>'):
            m, d = self.model(mj, body, '<material name="red" rgba="0.8 0.1 0.1 1"/>')
            view = MuJoCoRenderer(m, W, H, lights=LIGHTS)
            assert is_red(center_pixel(view.render(d, view.camera((0, -4, 1), (0, 0, 1))))), body

    def test_a_plane_is_a_checkerboard_in_its_own_colour(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        m, d = self.model(mj, '<geom type="plane" size="0 0 0.1" rgba="0.8 0.3 0.3 1"/>'
                              '<geom type="sphere" size="0.3" pos="0 0 0.3"/>')
        view = MuJoCoRenderer(m, W, H, lights=LIGHTS)
        assert view.instance_count == 2
        image = view.render(d, view.camera((0, -4, 2), (0, 1, 0))).astype(int)
        ground = image[H * 3 // 4:, :]
        assert ground[:, :, 1].max() - ground[:, :, 1].min() > 8  # squares of two shades
        assert ground[:, :, 0].mean() > ground[:, :, 2].mean() + 20  # and the geom's reddish colour

    def test_sensor_images_name_the_geoms(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        m, d = self.model(mj, '<geom type="plane" size="0 0 0.1"/><geom type="sphere" size="0.5" pos="0 0 1"/>'
                              '<geom type="box" size="0.3 0.3 0.3" pos="2 0 0.3"/>')
        view = MuJoCoRenderer(m, W, H, lights=LIGHTS)
        camera = view.camera((0, -5, 1), (0, 0, 1))
        out = view.render_frame(d, camera, rgb=False, depth=True, object_ids=True)
        ids = view.object_id_of_geom
        assert out["object_ids"][H // 2, W // 2] == ids[1]
        assert ids[0] in set(out["object_ids"].ravel().tolist())  # the ground
        assert out["depth"][H // 2, W // 2] == pytest.approx(4.5, abs=0.05)
        hit = view.cast_rays(np.array([[2.0, -5.0, 0.3]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32))
        assert hit["object_id"][0] == ids[2] and hit["distance"][0] == pytest.approx(4.7, abs=1e-3)

    def test_follows_the_simulation(self, mj, gpu):
        from neraulgraphics.mujoco import MuJoCoRenderer
        m, d = self.model(mj, '<body pos="0 0 3"><freejoint/><geom type="sphere" size="0.4"/></body>')
        view = MuJoCoRenderer(m, W, H, lights=LIGHTS)
        camera = view.camera((0, -6, 2), (0, 0, 2))
        before = view.render(d, camera)
        for _ in range(100):
            mj.mj_step(m, d)  # the sphere falls
        assert d.geom_xpos[0, 2] < 2.9
        assert not np.array_equal(before, view.render(d, camera))
