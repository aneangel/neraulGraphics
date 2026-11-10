#ifndef NERAULGRAPHICS_BATCH_H
#define NERAULGRAPHICS_BATCH_H

// Many worlds rendered at once: the setting of reinforcement learning, where thousands of copies of
// the same environment run in parallel, each with its own camera, and every one needs an image.
//
// All worlds share the meshes, textures, lights, spheres and floor. Each holds the same list of
// instances (a robot, its objects), placed differently. A camera sees only its own world's
// instances, so worlds can overlap in space without showing through one another, as simulators
// leave them. Every camera is one view: the images come back as one batch, a tensor of
// (views, height, width, ...), in the renderer's own memory, which on a GPU is the GPU's.
//
// Poses and cameras can be given from the GPU too (Memory::Device): the simulator's state goes
// straight to the renderer and the images straight to the policy, with no pass through the CPU.

#include <cstdint>
#include <memory>
#include <vector>
#include <neraulgraphics/renderer.h>

namespace neraulgraphics {

// The images of a batch of views, in the renderer's own buffers: valid until the next call that
// renders. Views are numbered world-major: view v looks at world v / cameras_per_world. Each image
// is laid out like Frame's (rows top to bottom), views one after another. A pointer is null for an
// image that was not asked for.
struct BatchFrame {
    Memory memory;
    int device_id;  // which GPU, when memory is Device; 0 otherwise
    int views;
    int width;
    int height;
    const uint8_t *rgb;
    const float *depth;
    const float *normal;
    const int32_t *object_id;
};

// The same images copied into vectors.
struct HostBatchFrame {
    int views = 0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
    std::vector<float> depth;
    std::vector<float> normal;
    std::vector<int32_t> object_id;
};

// How a simulator's bodies carry the instances. Each instance is attached to a body (or to none) with
// a pose of its own relative to it and a scale, which is how simulators describe a shape.
struct BodyBinding {
    int bodies_per_world = 0;
    std::vector<int> body_of_instance;  // per instance: a body index in 0..bodies_per_world-1, or -1 for fixed in the world
    std::vector<float> local_pose;      // per instance: x, y, z, then a quaternion x, y, z, w
    std::vector<float> scale;           // per instance: x, y, z, applied along the mesh's own axes
};

class BatchRenderer {
public:
    // Every world holds `instances` (placed as they say, to start with) and is seen by
    // cameras_per_world cameras, each width x height. The meshes and textures are built once and
    // shared. Throws std::invalid_argument for what Renderer's constructor rejects, and for a world or
    // camera count below 1.
    BatchRenderer(int width, int height, int world_count, int cameras_per_world, const std::vector<Sphere> &spheres,
        const std::vector<Light> &lights, const Plane &floor, const std::vector<Mesh> &meshes,
        const std::vector<Instance> &instances, const std::vector<Texture> &textures = {},
        Device device = Device::Auto);
    ~BatchRenderer();
    BatchRenderer(BatchRenderer &&) noexcept;
    BatchRenderer &operator=(BatchRenderer &&) noexcept;
    BatchRenderer(const BatchRenderer &) = delete;
    BatchRenderer &operator=(const BatchRenderer &) = delete;

    // Places every instance of every world: world_count * instance_count matrices of 12 floats (the
    // top three rows of the 4x4, row-major), world-major. In host memory each is checked (finite and
    // invertible) and a bad one throws std::invalid_argument, changing nothing. In device memory they
    // are trusted, since checking would need a copy back: a NaN or singular matrix draws garbage.
    void set_instance_transforms(const float *matrices, Memory memory = Memory::Host);

    // Describes how simulator bodies carry the instances, once. Then set_body_poses() places everything
    // from the simulation's state. Throws std::invalid_argument for a body index out of range or a
    // value that is NaN or infinite.
    void bind_bodies(const BodyBinding &binding);

    // Places every instance of every world from body poses: world_count * bodies_per_world poses of
    // 7 floats (x, y, z, then a quaternion x, y, z, w), world-major, as simulators hold them. The
    // instance's matrix is its body's pose, then its own offset, then its scale. On a GPU this is one
    // kernel, so a simulator's state array can be passed as it is.
    void set_body_poses(const float *body_poses, Memory memory = Memory::Host);

    // Renders every view. camera_matrices: view_count matrices of 12 floats, each the top three rows of
    // a camera-to-world transform in the OpenGL convention (x right, y up, looking along -z), so the
    // matrix of a robot's head link works as it is. fov is the vertical field of view in radians,
    // shared by every camera. samples_per_pixel averages that many jittered rays for the picture; the
    // other images are one exact ray through the pixel's middle, as in Renderer::render_frame. There
    // is no accumulation between calls: every call draws every view afresh. frame only varies the
    // sampling noise (the same value draws the same picture), so successive calls can use different
    // noise, or the same.
    //
    // Throws std::invalid_argument for no outputs, an unknown output, a fov outside (0, pi), a
    // sample count below 1, or a negative frame. Cameras in host memory are checked for NaN.
    BatchFrame render(const float *camera_matrices, float fov, int samples_per_pixel, unsigned outputs,
        int frame = 0, Memory camera_memory = Memory::Host);
    HostBatchFrame render_host(const float *camera_matrices, float fov, int samples_per_pixel, unsigned outputs,
        int frame = 0);

    // Casts rays_per_world rays in every world (world_count * rays_per_world rays, world-major, 3 floats
    // each): the same query as Renderer::cast_rays, with each ray seeing only its own world's instances.
    RayHits cast_rays(const float *origins, const float *directions, int rays_per_world, float max_distance,
        Memory input_memory = Memory::Host);
    HostRayHits cast_rays_host(const std::vector<float> &origins, const std::vector<float> &directions,
        int rays_per_world, float max_distance);

    int width() const;
    int height() const;
    int world_count() const;
    int cameras_per_world() const;
    int view_count() const;
    int instances_per_world() const;
    Device device() const;
    Memory memory() const;
    float last_kernel_ms() const;

    // Light that reaches every surface regardless of the lights: see Renderer::set_ambient.
    void set_ambient(float ambient);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Impl &state();
    const Impl &state() const;
};

}  // namespace neraulgraphics

#endif  // NERAULGRAPHICS_BATCH_H
