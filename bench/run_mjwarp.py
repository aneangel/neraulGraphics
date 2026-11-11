"""Times MuJoCo Warp's batch renderer on the recorded poses (run in the competitors environment).

    python bench/run_mjwarp.py --poses bench/data/poses_8.npz --out bench/results/r1_mjwarp_8.json

Per frame it points the data at that frame's poses, refits the BVH and renders: shadows on, one sample per pixel, no textures,
RGB only, the settings that match what neraulgraphics' classic path does (one primary ray and one shadow ray per pixel,
no bounces). Two timing boundaries are recorded: `native` ends with the packed image MuJoCo Warp renders into, `unpacked`
also converts it to a float RGB array, the form a policy would read.

Three more variants give MuJoCo Warp its best chance: `cheap` switches off the shading terms its docs say cost speed (specular,
emission, per-light ambient, backface culling, vertex normals), `bd256` raises the render block size to 256, `cheap_bd256`
does both. They do less work than neraulgraphics does, so beating them is a conservative claim. Block sizes of 512 and up
cannot launch this kernel at all.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from timing import environment, gpu_state, measure, poses_hash  # noqa: E402

wp.config.quiet = True

CHEAP = dict(enable_specular=False, enable_emission=False, enable_per_light_ambient=False, enable_backface_culling=False,
             enable_vertex_normals=False)
VARIANTS = {"cheap": (CHEAP, None), "bd256": ({}, 256), "cheap_bd256": (CHEAP, 256)}
DEFAULT_BLOCK = 64
DEPTH_RANGE = 10.0    # get_depth returns clamp(depth / range, 0, 1), so this is the farthest depth (m) it can report


def frames_on_device(pos, mat, nworld):
    """Every recorded frame of the first nworld worlds as device arrays, so a frame's poses are handed over with no copy."""
    px = [wp.array(np.ascontiguousarray(pos[f, :nworld]), dtype=wp.vec3) for f in range(pos.shape[0])]
    pm = [wp.array(np.ascontiguousarray(mat[f, :nworld]), dtype=wp.mat33) for f in range(mat.shape[0])]
    return px, pm


def context(mjm, nworld, res, depth=False, **extra):
    return mjw.create_render_context(mjm, nworld=nworld, cam_res=(res, res), render_rgb=True, render_depth=depth, render_seg=False,
                                     use_textures=False, use_shadows=True, samples_per_pixel=1, enabled_geom_groups=[0, 1, 2], **extra)


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
    m = mjw.put_model(mjm)
    results = []
    for nworld in args.worlds:
        px, pm = frames_on_device(pos, mat, nworld)
        d = mjw.make_data(mjm, nworld=nworld)
        mjw.kinematics(m, d)
        mjw.camlight(m, d)
        assert np.allclose(d.cam_xpos.numpy()[0, 0], z["cam_xpos"], atol=1e-5), "camera differs from the recorded one"
        for res in args.res:
            rc = context(mjm, nworld, res)
            out = wp.zeros((nworld, res, res), dtype=wp.vec3)
            n = len(px)

            def native(i):
                d.geom_xpos, d.geom_xmat = px[i % n], pm[i % n]
                mjw.refit_bvh(m, d, rc)
                mjw.render(m, d, rc)

            def unpacked(i):
                native(i)
                mjw.get_rgb(rc, 0, out)

            cases = [("native", native, DEFAULT_BLOCK), ("unpacked", unpacked, DEFAULT_BLOCK)]
            for name, (extra, block) in VARIANTS.items():
                rcv = context(mjm, nworld, res, **extra)

                def variant(i, rcv=rcv):
                    d.geom_xpos, d.geom_xmat = px[i % n], pm[i % n]
                    mjw.refit_bvh(m, d, rcv)
                    mjw.render(m, d, rcv)

                cases.append((name, variant, block or DEFAULT_BLOCK))
            for boundary, fn, block in cases:
                m.block_dim.render = block
                before = gpu_state()
                r = measure(fn, wp.synchronize, reps=args.reps)
                results.append({"nbody": int(z["nbody"]), "nworld": nworld, "res": res, "boundary": boundary, **r, "gpu_before": before, "gpu_after": gpu_state()})
                med = float(np.median(r["reps_ms"]))
                print(f"mjwarp nbody={int(z['nbody'])} nworld={nworld:5d} res={res:3d} {boundary:11s} {med:8.4f} ms/frame  ({nworld / med * 1e3:11.0f} img/s)", flush=True)
            m.block_dim.render = DEFAULT_BLOCK
            del rc, out, cases
            gc.collect()
        del px, pm, d
    if args.verify:
        nw, res = 16, 128
        px, pm = frames_on_device(pos, mat, nw)
        d = mjw.make_data(mjm, nworld=nw)
        mjw.kinematics(m, d)
        mjw.camlight(m, d)
        rc = context(mjm, nw, res, depth=True)
        rgb, depth, shots = wp.zeros((nw, res, res), dtype=wp.vec3), wp.zeros((nw, res, res), dtype=wp.float32), {}
        for f in (0, len(px) // 2):
            d.geom_xpos, d.geom_xmat = px[f], pm[f]
            mjw.refit_bvh(m, d, rc)
            mjw.render(m, d, rc)
            mjw.get_rgb(rc, 0, rgb)
            mjw.get_depth(rc, 0, DEPTH_RANGE, depth)
            shots[f"rgb_{f}"], shots[f"depth_{f}"] = rgb.numpy().copy(), depth.numpy() * DEPTH_RANGE
        np.savez(args.verify, **shots)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"system": "mjwarp", "poses_hash": str(z["poses_hash"]), "environment": environment(["mujoco", "mujoco-warp", "warp-lang"]),
               "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
