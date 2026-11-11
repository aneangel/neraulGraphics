// Python bindings: the neraulgraphics._physics extension module, a rigid-body world that gives the same answer on every
// machine.
//
// Contacts and constraints are solved by Jolt Physics built for cross-platform determinism (strict IEEE arithmetic, no fused
// multiply-adds, its own sin, cos and atan2). Nothing else changes the simulation, so a world that is fed the same bodies and
// the same events in the same order, with the same dt, produces the same bits in poses() and velocities() and so the same
// state_hash(), whatever the thread count.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

namespace nb = nanobind;
using namespace nb::literals;
using JPH::Quat;
using JPH::Vec3;

namespace {

namespace Layers {
static constexpr JPH::ObjectLayer NON_MOVING = 0;
static constexpr JPH::ObjectLayer MOVING = 1;
static constexpr JPH::uint NUM_LAYERS = 2;
}  // namespace Layers

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        return a == Layers::MOVING || b == Layers::MOVING;
    }
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return Layers::NUM_LAYERS; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override { return JPH::BroadPhaseLayer(static_cast<JPH::uint8>(layer)); }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override { return layer == JPH::BroadPhaseLayer(0) ? "NON_MOVING" : "MOVING"; }
#endif
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer broad) const override {
        return layer == Layers::MOVING || broad == JPH::BroadPhaseLayer(1);
    }
};

void init_jolt_once() {
    static const bool done = [] {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
        return true;
    }();
    (void)done;
}

using Floats = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

void need(const bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}

Vec3 vec3(const std::array<float, 3> &v) { return Vec3(v[0], v[1], v[2]); }

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T> &&data, std::initializer_list<size_t> shape) {
    auto *owner_vector = new std::vector<T>(std::move(data));
    nb::capsule owner(owner_vector, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
    return nb::ndarray<nb::numpy, T>(owner_vector->data(), shape.size(), shape.begin(), owner);
}


// ------------------------------------------------------------------------------------------------------------- the world --

// Jolt's allocator must be registered before anything of Jolt's is constructed, so this is the World's first member.
struct JoltRuntime {
    JoltRuntime() { init_jolt_once(); }
};

class World {
public:
    World(const std::array<float, 3> &gravity, const int threads, const uint32_t max_bodies, const int velocity_steps, const int position_steps)
        : temp_allocator_(256u * 1024u * 1024u),
          jobs_(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, threads <= 0 ? -1 : threads - 1) {
        system_.Init(max_bodies, 0, std::max(65536u, 32u * max_bodies), std::max(20480u, 16u * max_bodies), layers_, object_vs_broad_, pair_filter_);
        system_.SetGravity(vec3(gravity));
        JPH::PhysicsSettings settings = system_.GetPhysicsSettings();
        settings.mNumVelocitySteps = static_cast<JPH::uint>(velocity_steps);
        settings.mNumPositionSteps = static_cast<JPH::uint>(position_steps);
        settings.mAllowSleeping = false;
        settings.mDeterministicSimulation = true;
        system_.SetPhysicsSettings(settings);
    }

    // -------------------------------------------------------------------------------------------------------- bodies --

    // A static slab whose top is the plane z = 0. It collides but is not one of the numbered bodies, so poses() leaves it out.
    void add_ground(const float friction, const float restitution) {
        JPH::BoxShapeSettings shape(Vec3(2000.0f, 2000.0f, 5.0f), 0.0f);
        JPH::BodyCreationSettings s(shape.Create().Get(), JPH::RVec3(0, 0, -5.0f), Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
        s.mFriction = friction;
        s.mRestitution = restitution;
        JPH::BodyInterface &bi = system_.GetBodyInterfaceNoLock();
        JPH::Body *body = bi.CreateBody(s);
        need(body != nullptr, "the world is full: raise max_bodies");
        bi.AddBody(body->GetID(), JPH::EActivation::DontActivate);
    }

    int add_convex_hull(const Floats &vertices, const std::array<float, 3> &position, const float density, const float friction,
        const float restitution, const float convex_radius, const bool is_static, const float damping) {
        need(vertices.ndim() == 2 && vertices.shape(1) == 3 && vertices.shape(0) >= 4, "vertices must be a (>= 4, 3) array");
        JPH::Array<Vec3> points;
        for (size_t i = 0; i < vertices.shape(0); ++i) points.push_back(Vec3(vertices.data()[3 * i], vertices.data()[3 * i + 1], vertices.data()[3 * i + 2]));
        JPH::ConvexHullShapeSettings shape(points, convex_radius);
        shape.mDensity = density;
        auto made = shape.Create();
        need(!made.HasError(), "the vertices do not make a valid convex hull");
        JPH::BodyCreationSettings s(made.Get(), JPH::RVec3(vec3(position)), Quat::sIdentity(),
            is_static ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic, is_static ? Layers::NON_MOVING : Layers::MOVING);
        s.mFriction = friction;
        s.mRestitution = restitution;
        s.mAllowSleeping = false;
        s.mLinearDamping = s.mAngularDamping = damping;   // Jolt's default is 0.05 a second, which is drag a vacuum does not have
        return add(s);
    }

    int add_sphere(const float radius, const std::array<float, 3> &position, const float density, const float friction, const float restitution, const float damping) {
        JPH::SphereShapeSettings shape(radius);
        shape.mDensity = density;
        JPH::BodyCreationSettings s(shape.Create().Get(), JPH::RVec3(vec3(position)), Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        s.mFriction = friction;
        s.mRestitution = restitution;
        s.mAllowSleeping = false;
        s.mLinearDamping = s.mAngularDamping = damping;
        return add(s);
    }

    void finish_setup() { system_.OptimizeBroadPhase(); }

    // ------------------------------------------------------------------------------------------------------ stepping --

    // One frame: substeps equal solver steps.
    void step(const float dt, const int substeps) {
        need(dt > 0.0f && substeps >= 1, "dt must be positive and substeps at least 1");
        const float h = dt / static_cast<float>(substeps);
        for (int s = 0; s < substeps; ++s) {
            const JPH::EPhysicsUpdateError error = system_.Update(h, 1, &temp_allocator_, &jobs_);
            if (error != JPH::EPhysicsUpdateError::None) throw std::runtime_error("the physics step ran out of room for contacts or body pairs");
        }
    }

    // --------------------------------------------------------------------------------------------------------- state --

    size_t body_count() const { return bodies_.size(); }

    nb::ndarray<nb::numpy, float> poses() const {
        std::vector<float> out(bodies_.size() * 7);
        for (size_t i = 0; i < bodies_.size(); ++i) {
            const Vec3 p = Vec3(bodies_[i]->GetPosition());
            const Quat q = bodies_[i]->GetRotation();
            float *o = &out[7 * i];
            o[0] = p.GetX(); o[1] = p.GetY(); o[2] = p.GetZ();
            o[3] = q.GetX(); o[4] = q.GetY(); o[5] = q.GetZ(); o[6] = q.GetW();
        }
        return to_numpy(std::move(out), {bodies_.size(), size_t(7)});
    }

    // linear velocity of the centre of mass, then angular velocity
    nb::ndarray<nb::numpy, float> velocities() const {
        std::vector<float> out(bodies_.size() * 6);
        for (size_t i = 0; i < bodies_.size(); ++i) {
            const Vec3 v = bodies_[i]->GetLinearVelocity(), w = bodies_[i]->GetAngularVelocity();
            float *o = &out[6 * i];
            o[0] = v.GetX(); o[1] = v.GetY(); o[2] = v.GetZ();
            o[3] = w.GetX(); o[4] = w.GetY(); o[5] = w.GetZ();
        }
        return to_numpy(std::move(out), {bodies_.size(), size_t(6)});
    }

    // Everything back to the given poses and velocities.
    void reset(const Floats &poses, const Floats &velocities) {
        need(poses.ndim() == 2 && poses.shape(0) == bodies_.size() && poses.shape(1) == 7, "poses must be (body_count, 7)");
        need(velocities.ndim() == 2 && velocities.shape(0) == bodies_.size() && velocities.shape(1) == 6, "velocities must be (body_count, 6)");
        JPH::BodyInterface &bi = system_.GetBodyInterfaceNoLock();
        for (size_t i = 0; i < bodies_.size(); ++i) {
            const float *p = &poses.data()[7 * i];
            const float *v = &velocities.data()[6 * i];
            bi.SetPositionAndRotation(bodies_[i]->GetID(), JPH::RVec3(p[0], p[1], p[2]), Quat(p[3], p[4], p[5], p[6]), JPH::EActivation::DontActivate);
            if (!bodies_[i]->IsStatic()) {
                bodies_[i]->SetLinearVelocity(Vec3(v[0], v[1], v[2]));
                bodies_[i]->SetAngularVelocity(Vec3(v[3], v[4], v[5]));
            }
        }
    }

    // FNV-1a over the bits of every body's pose and velocities: the same simulation gives the same number everywhere.
    uint64_t state_hash() const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](const float x) {
            uint32_t bits;
            std::memcpy(&bits, &x, 4);
            for (int k = 0; k < 4; ++k) {
                h ^= (bits >> (8 * k)) & 0xffu;
                h *= 1099511628211ULL;
            }
        };
        for (JPH::Body *body : bodies_) {
            const Vec3 p = Vec3(body->GetPosition());
            const Quat q = body->GetRotation();
            const Vec3 v = body->GetLinearVelocity(), w = body->GetAngularVelocity();
            for (const float x : {p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW(), v.GetX(), v.GetY(), v.GetZ(), w.GetX(), w.GetY(), w.GetZ()}) mix(x);
        }
        return h;
    }

    // Puts body index at origin, unrotated, with a linear and an angular velocity (zero if not given).
    void fire(const int index, const std::array<float, 3> &origin, const std::array<float, 3> &velocity, const std::array<float, 3> &angular) {
        need(index >= 0 && static_cast<size_t>(index) < bodies_.size(), "no such body");
        JPH::BodyInterface &bi = system_.GetBodyInterfaceNoLock();
        JPH::Body *body = bodies_[index];
        bi.SetPositionAndRotation(body->GetID(), JPH::RVec3(vec3(origin)), Quat::sIdentity(), JPH::EActivation::DontActivate);
        body->SetLinearVelocity(vec3(velocity));
        body->SetAngularVelocity(vec3(angular));
    }

private:
    int add(const JPH::BodyCreationSettings &settings) {
        JPH::BodyInterface &bi = system_.GetBodyInterfaceNoLock();
        JPH::Body *body = bi.CreateBody(settings);
        need(body != nullptr, "the world is full: raise max_bodies");
        bi.AddBody(body->GetID(), JPH::EActivation::Activate);
        bodies_.push_back(body);
        return static_cast<int>(bodies_.size()) - 1;
    }

    JoltRuntime runtime_;
    BPLayerInterfaceImpl layers_;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broad_;
    ObjectLayerPairFilterImpl pair_filter_;
    JPH::PhysicsSystem system_;
    JPH::TempAllocatorImpl temp_allocator_;
    JPH::JobSystemThreadPool jobs_;
    std::vector<JPH::Body *> bodies_;
};

}  // namespace

NB_MODULE(_physics, m) {
    m.doc() = "A rigid-body world that gives bit-identical results on every machine (Jolt Physics, cross-platform deterministic build).";

    nb::class_<World>(m, "World",
        "Rigid bodies, simulated deterministically. Z is up. Create the bodies (ground first), call finish_setup(), "
        "then call step() with a fixed dt.")
        .def(nb::init<const std::array<float, 3> &, int, uint32_t, int, int>(), "gravity"_a = std::array<float, 3>{0.0f, 0.0f, -9.81f},
            "threads"_a = 0, "max_bodies"_a = 16384, "velocity_steps"_a = 10, "position_steps"_a = 2,
            "threads is how many threads the solver may use (0: all cores); the result does not depend on it.")
        .def("add_ground", &World::add_ground, "friction"_a = 0.7f, "restitution"_a = 0.0f, "A static slab whose top is z = 0 (not one of the numbered bodies).")
        .def("add_convex_hull", &World::add_convex_hull, "vertices"_a, "position"_a, "density"_a = 2400.0f, "friction"_a = 0.7f,
            "restitution"_a = 0.0f, "convex_radius"_a = 0.02f, "static"_a = false, "damping"_a = 0.0f,
            "A convex body from its vertices (relative to position, an (n, 3) float array). damping is the fraction of velocity "
            "lost each second to drag (0: none, as in a vacuum). Returns the body index.")
        .def("add_sphere", &World::add_sphere, "radius"_a, "position"_a, "density"_a = 1000.0f, "friction"_a = 0.5f, "restitution"_a = 0.05f,
            "damping"_a = 0.0f)
        .def("finish_setup", &World::finish_setup, "Call once after the last body is added.")
        .def("step", &World::step, "dt"_a, "substeps"_a = 1, nb::call_guard<nb::gil_scoped_release>(),
            "Advances dt seconds in substeps equal solver steps.")
        .def("poses", &World::poses, "(body_count, 7) float32: position, then quaternion x y z w, of every body's origin.")
        .def("velocities", &World::velocities, "(body_count, 6) float32: linear velocity of the centre of mass, then angular velocity.")
        .def("reset", &World::reset, "poses"_a, "velocities"_a)
        .def("state_hash", &World::state_hash, "A 64-bit hash of every body's exact pose and velocities.")
        .def("fire", &World::fire, "index"_a, "origin"_a, "velocity"_a, "angular"_a = std::array<float, 3>{0.0f, 0.0f, 0.0f},
            "Puts body index at origin, unrotated, with a linear velocity and an optional angular velocity.")
        .def_prop_ro("body_count", &World::body_count);
}
