"""Records the poses every renderer is fed, so the benchmark compares renderers and not physics.

    python bench/gen_poses.py --nbody 8 --out bench/data/poses_8.npz

Steps seeded, randomly perturbed copies of the scene's falling bodies in MuJoCo Warp, then runs MuJoCo Warp's own kinematics
on the mesh version of the same model to get the world pose of every geom in every frame. Those arrays (and the camera) are
all the runners read: each replays them, cycling, so each frame moves every body of every world.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene import physics_xml, render_xml  # noqa: E402
from timing import poses_hash  # noqa: E402

wp.config.quiet = True


def random_start(rng, base, nworld, nbody):
    """base (nq,) tiled to nworld starting poses, each body nudged in x, y, z and spun to a random orientation."""
    q = np.tile(base, (nworld, 1)).reshape(nworld, nbody, 7)
    q[..., :2] += rng.uniform(-0.04, 0.04, (nworld, nbody, 2))
    q[..., 2] += rng.uniform(0.0, 0.08, (nworld, nbody))
    r = rng.normal(size=(nworld, nbody, 4))
    q[..., 3:] = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return q.reshape(nworld, nbody * 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbody", type=int, default=8)
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--steps", type=int, default=4, help="physics steps between recorded frames")
    p.add_argument("--chunk", type=int, default=1024, help="worlds simulated at a time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    mp, mr = mujoco.MjModel.from_xml_string(physics_xml(args.nbody)), mujoco.MjModel.from_xml_string(render_xml(args.nbody))
    assert mp.nq == mr.nq and mp.nbody == mr.nbody, "physics and render models must have the same bodies"
    model_p, model_r = mjw.put_model(mp), mjw.put_model(mr)
    pos, mat, cam = [], [], None
    for c0 in range(0, args.worlds, args.chunk):
        n = min(args.chunk, args.worlds - c0)
        rng = np.random.default_rng(args.seed + c0)
        dp, dr = mjw.make_data(mp, nworld=n, nconmax=args.nbody * 8, njmax=args.nbody * 40), mjw.make_data(mr, nworld=n)
        dp.qpos.assign(random_start(rng, mp.qpos0, n, args.nbody).astype(np.float32))
        cp, cm = [], []
        for _ in range(args.frames):
            for _ in range(args.steps):
                mjw.step(model_p, dp)
            dr.qpos.assign(dp.qpos.numpy())
            mjw.kinematics(model_r, dr)
            if cam is None:
                mjw.camlight(model_r, dr)
                cam = (dr.cam_xpos.numpy()[0, 0].copy(), dr.cam_xmat.numpy()[0, 0].copy())
            cp.append(dr.geom_xpos.numpy().copy())
            cm.append(dr.geom_xmat.numpy().copy())
        pos.append(np.stack(cp))
        mat.append(np.stack(cm))
        print(f"worlds {c0}..{c0 + n} done", flush=True)
    pos, mat = np.concatenate(pos, axis=1).astype(np.float32), np.concatenate(mat, axis=1).astype(np.float32)
    assert np.isfinite(pos).all() and np.isfinite(mat).all(), "a world blew up"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, geom_xpos=pos, geom_xmat=mat, cam_xpos=cam[0], cam_xmat=cam[1], xml=render_xml(args.nbody), nbody=args.nbody,
             fovy=float(mr.cam_fovy[0]), poses_hash=poses_hash(pos, mat))
    print(f"wrote {args.out}: frames {pos.shape[0]}, worlds {pos.shape[1]}, geoms {pos.shape[2]}, hash {poses_hash(pos, mat)}")


if __name__ == "__main__":
    main()
