"""Does the Vulkan ray-query path agree with the CPU reference?

    python tools/vulkan/compare.py --tool build/vk_rayquery --spv build/rayquery.spv

Makes a seeded random scene of rotated, scaled boxes and a batch of rays, has every Vulkan device trace them (vk_rayquery) and the
CPU backend trace them (Renderer.cast_rays), and compares what each ray hit and how far away. Exits non-zero if any device disagrees on more than 1 ray in 100,000 (the few that remain graze a silhouette edge).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import neraulgraphics as ng


def make_scene(seed, boxes, rays):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(boxes, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    rot = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(boxes, 3, 3)
    scale = rng.uniform(0.4, 2.2, boxes)
    matrices = np.concatenate([rot * scale[:, None, None], rng.uniform(-25, 25, (boxes, 3, 1))], axis=2).astype(np.float32)   # (N, 3, 4)
    origins = rng.uniform(-40, 40, (rays, 3)).astype(np.float32)
    targets = rng.uniform(-28, 28, (rays, 3)).astype(np.float32)
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return matrices, origins, directions.astype(np.float32)


def cpu_reference(matrices, origins, directions):
    box = ng.make_box((0.5, 0.5, 0.5))
    renderer = ng.Renderer(64, 64, [], [], meshes=[box], instances=[ng.Instance(0) for _ in range(len(matrices))], device="cpu")
    renderer.set_instance_transforms(matrices)
    hits = renderer.cast_rays(origins, directions, 1.0e30)
    d = np.asarray(hits["distance"], np.float64)
    ids = np.asarray(hits["object_id"], np.int64) - ng.OBJECT_INSTANCE        # the reference numbers instance i as OBJECT_INSTANCE + i
    miss = ~np.isfinite(d)
    return np.where(miss, -1.0, d), ids, miss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--spv", required=True)
    parser.add_argument("--boxes", type=int, default=2000)
    parser.add_argument("--rays", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    matrices, origins, directions = make_scene(args.seed, args.boxes, args.rays)
    ref_d, ref_id, ref_miss = cpu_reference(matrices, origins, directions)
    print(f"{args.boxes} boxes, {args.rays} rays: the CPU reference hits {int((~ref_miss).sum())} of them")

    listing = subprocess.run([args.tool, args.spv, "/dev/null", "/dev/null", "list"], capture_output=True, text=True, check=True).stdout.split("\n")
    devices = [line.split(" ", 1) for line in listing if line.strip()]
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp) / "scene.bin"
        with open(scene, "wb") as f:
            f.write(np.array([args.boxes, args.rays], np.int32).tobytes())
            f.write(matrices.reshape(-1).tobytes())
            f.write(np.concatenate([origins, directions], axis=1).reshape(-1).tobytes())
        for index, name in devices:
            out = Path(tmp) / f"hits_{index}.bin"
            run = subprocess.run([args.tool, args.spv, str(scene), str(out), index], capture_output=True, text=True)
            if run.returncode != 0:
                print(f"[{index}] {name}: FAILED to run\n{run.stderr}")
                failed = True
                continue
            raw = np.fromfile(out, np.dtype([("t", "<f4"), ("id", "<i4")]))
            t, ids = raw["t"].astype(np.float64), raw["id"].astype(np.int64)
            miss = ids < 0
            both = ~miss & ~ref_miss
            same_id = (ids == ref_id) & both
            # A ray that starts on a surface (within the ray-start epsilon of 1e-4 m) is a convention, not a measurement: the CPU
            # counts the self-hit and a Vulkan ray query starting at tMin does not. They are set aside; every other distance must
            # agree to 0.1 % or to a tenth of a millimetre.
            on_surface = ref_miss.copy()
            on_surface = (ref_d >= 0) & (ref_d < 2.0e-4)
            hit_agree = (miss == ref_miss)[~on_surface].mean()
            both = both & ~on_surface
            rel = np.abs(t - ref_d) / np.maximum(ref_d, 1e-6)
            near = both & ((rel < 1e-3) | (np.abs(t - ref_d) < 1.0e-4))
            rate = [l for l in run.stderr.splitlines() if "Mrays" in l]
            print(f"[{index}] {name}: hit/miss agrees on {hit_agree * 100:.4f} % of rays ({int(on_surface.sum())} that start on a surface set aside); distance "
                  f"agrees on {near.sum() / max(both.sum(), 1) * 100:.4f} % of common hits; same instance on {same_id[~on_surface].sum() / max(both.sum(), 1) * 100:.4f} %"
                  f"  |  {rate[0].split(': ')[1] if rate else ''}")
            # A few rays in a million graze a silhouette edge, where a hardware intersection test and the CPU's round differently.
            n = max(both.sum(), 1)
            if not (hit_agree >= 0.99999 and near.sum() / n >= 0.99999 and same_id[~on_surface].sum() / n >= 0.99999):
                failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
