"""Times neraulgraphics' batch renderer on the recorded poses (run in the ours environment).

    python bench/run_ours.py --poses bench/data/poses_8.npz --out bench/results/r1_ours_8.json

Per frame it hands the frame's geom poses to MJWarpBatchRenderer (which composes them into instance matrices on the GPU and
rebuilds the instance trees) and renders in the classic mode: one sample per pixel, one point light with no radius (a hard
shadow), materials with no reflectivity, RGB only into the renderer's own buffer. That is the same work, ray for ray, as the
settings run_mjwarp.py uses. The camera is a device array, as in MuJoCo Warp where it is part of the model.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import neraulgraphics as ng
import numpy as np
import warp as wp
from neraulgraphics.mujoco import MJWarpBatchRenderer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene import LIGHT_POSITION  # noqa: E402
from timing import environment, gpu_state, measure, poses_hash  # noqa: E402

wp.config.quiet = True


def camera_matrix(z):
    """Camera-to-world 3 x 4 in the OpenGL convention: MuJoCo's camera looks along -z with y up, so its axes are used as they are."""
    return np.concatenate([z["cam_xmat"], z["cam_xpos"][:, None]], axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--worlds", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--res", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--verify", help="also write depth and colour of a few worlds here, for the equivalence check")
    args = p.parse_args()

    z = np.load(args.poses)
    pos, mat = z["geom_xpos"], z["geom_xmat"]
    mjm = mujoco.MjModel.from_xml_string(str(z["xml"]))
    fov = float(np.radians(z["fovy"]))
    cam = camera_matrix(z)
    lights = [ng.Light(LIGHT_POSITION, intensity=1.0, radius=0.0)]
    results = []
    for nworld in args.worlds:
        px = [wp.array(np.ascontiguousarray(pos[f, :nworld]), dtype=wp.vec3) for f in range(pos.shape[0])]
        pm = [wp.array(np.ascontiguousarray(mat[f, :nworld]), dtype=wp.mat33) for f in range(mat.shape[0])]
        cams = wp.array(np.tile(cam, (nworld, 1, 1)), dtype=wp.float32)
        data = SimpleNamespace(geom_xpos=px[0], geom_xmat=pm[0])
        n = len(px)
        for res in args.res:
            view = MJWarpBatchRenderer(mjm, nworld, res, res, lights=lights, fov=fov)

            def frame(i):
                data.geom_xpos, data.geom_xmat = px[i % n], pm[i % n]
                view.render(data, cams, rgb=True, zero_copy=True, samples_per_pixel=1)

            before = gpu_state()
            r = measure(frame, wp.synchronize, reps=args.reps)
            results.append({"nbody": int(z["nbody"]), "nworld": nworld, "res": res, "boundary": "native", **r, "gpu_before": before, "gpu_after": gpu_state()})
            med = float(np.median(r["reps_ms"]))
            print(f"ours   nbody={int(z['nbody'])} nworld={nworld:5d} res={res:3d} native   {med:8.4f} ms/frame  ({nworld / med * 1e3:11.0f} img/s)", flush=True)
            del view
            gc.collect()
        del px, pm, cams, data
    if args.verify:
        nw, res = 16, 128
        px = [wp.array(np.ascontiguousarray(pos[f, :nw]), dtype=wp.vec3) for f in range(pos.shape[0])]
        pm = [wp.array(np.ascontiguousarray(mat[f, :nw]), dtype=wp.mat33) for f in range(mat.shape[0])]
        view = MJWarpBatchRenderer(mjm, nw, res, res, lights=lights, fov=fov)
        cams, shots = np.tile(cam, (nw, 1, 1)), {}
        for f in (0, len(px) // 2):
            out = view.render(SimpleNamespace(geom_xpos=px[f], geom_xmat=pm[f]), cams, rgb=True, depth=True, samples_per_pixel=1)
            shots[f"rgb_{f}"], shots[f"depth_{f}"] = np.asarray(out["rgb"]), np.asarray(out["depth"])
        np.savez(args.verify, **shots)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"system": "ours", "poses_hash": str(z["poses_hash"]), "environment": environment(["neraulgraphics", "mujoco", "warp-lang"]),
               "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
