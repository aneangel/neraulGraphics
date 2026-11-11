"""Turns the runners' result files into comparisons, and refuses to call a winner unless the comparison was fair.

    python bench/analyze.py bench/results/run1

Reads r<round>_<system>_<nbody>.json and verify_<system>_<nbody>.npz from a directory. For each scene size it first checks
the preconditions: both systems replayed identical poses (same hash), and they drew the same picture (silhouettes and depth
agree). If either fails, no speed claim is made for that scene. Otherwise every (worlds, resolution) case gets the median
milliseconds per frame of each system with a bootstrap 95% confidence interval, and the speedup of neraulgraphics with its own
interval. MuJoCo Warp is measured in its default configuration and in tuned ones (see run_mjwarp.py); the verdict is made
against whichever of them is fastest for that case, which is a conservative claim because the tuned ones do less work.
A case is called only if the interval excludes 1 AND every round on its own points the same way.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_IOU = 0.995          # silhouettes of the two renderers must overlap this much, in every world of every checked frame
MAX_DEPTH_P99_MM = 5.0   # and their depths agree to this on the 99th percentile of shared pixels
BOOTSTRAPS = 5000


def equivalence(directory, nbody):
    """(passed, text) for the picture-agreement check of one scene size."""
    a, b = directory / f"verify_mjwarp_{nbody}.npz", directory / f"verify_ours_{nbody}.npz"
    if not (a.exists() and b.exists()):
        return False, "no verification images"
    a, b = np.load(a), np.load(b)
    worst_iou, p99s, outliers, shared = 1.0, [], 0, 0
    for key in (k for k in a.files if k.startswith("depth_")):
        dm, do = a[key], b[key]
        mm, mo = dm > 0, np.isfinite(do)
        worst_iou = min(worst_iou, float(((mm & mo).sum((1, 2)) / np.maximum((mm | mo).sum((1, 2)), 1)).min()))
        both = mm & mo
        err = np.abs(dm - do)[both] * 1e3
        p99s.append(float(np.percentile(err, 99)))
        outliers += int((err > 1.0).sum())
        shared += int(both.sum())
    ok = worst_iou >= MIN_IOU and max(p99s) <= MAX_DEPTH_P99_MM
    return ok, (f"silhouette IoU >= {worst_iou:.5f}, depth p99 diff {max(p99s):.4f} mm, "
                f"{outliers} of {shared} shared pixels differ by more than 1 mm")


def boot_median(x, rng, n=BOOTSTRAPS):
    x = np.asarray(x)
    return np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)


def summarize(x):
    x = np.asarray(x)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "iqr": float(q3 - q1), "min": float(x.min()), "cv_pct": float(x.std() / x.mean() * 100)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--json", type=Path, help="also write the table here")
    args = p.parse_args()

    runs = defaultdict(lambda: defaultdict(list))    # (nbody, nworld, res, boundary, system) -> {round: [reps_ms]}
    hashes, envs, clocks = defaultdict(set), {}, defaultdict(list)
    for path in sorted(args.directory.glob("r*_*.json")):
        rnd, system, nbody = path.stem.split("_")
        data = json.loads(path.read_text())
        hashes[int(nbody)].add(data["poses_hash"])
        envs[data["system"]] = data["environment"]
        for r in data["results"]:
            runs[(r["nbody"], r["nworld"], r["res"], r["boundary"], data["system"])][rnd] += r["reps_ms"]
            if r["gpu_after"]:
                clocks[data["system"]].append(r["gpu_after"]["clocks.sm"] / r["gpu_after"]["clocks.max.sm"])
    if not runs:
        sys.exit("no result files found")

    print("Environment")
    for system, e in envs.items():
        print(f"  {system}: {e['gpu']}, python {e['python']}, " + ", ".join(f"{k}=={v}" for k, v in e["packages"].items()))
    for system, c in clocks.items():
        print(f"  {system}: GPU clock at end of each case, median {np.median(c) * 100:.0f}% of max (min {np.min(c) * 100:.0f}%)")

    rng = np.random.default_rng(0)
    rows = []
    for nbody in sorted(hashes):
        print(f"\n=== {nbody} bodies per world")
        same = len(hashes[nbody]) == 1
        ok, text = equivalence(args.directory, nbody)
        print(f"  identical poses replayed by every run: {'yes' if same else 'NO'} ({', '.join(sorted(hashes[nbody]))})")
        print(f"  same picture drawn: {'yes' if ok else 'NO'} ({text})")
        valid = same and ok
        if not valid:
            print("  -> preconditions failed: no speed claim for this scene")
        print(f"  {'worlds':>6} {'res':>4} {'MJWarp default':>16} {'MJWarp best':>24} {'ours ms [95% CI]':>24} {'vs default':>11} {'vs best MJWarp [95% CI]':>25}  verdict")
        keys = sorted({k[1:3] for k in runs if k[0] == nbody})
        tuned = ("native", "cheap", "bd256", "cheap_bd256")
        flat = lambda d: np.concatenate([np.asarray(v) for v in d.values()])
        ci = lambda x: np.percentile(x, [2.5, 97.5])
        for nworld, res in keys:
            ours = runs[(nbody, nworld, res, "native", "ours")]
            mj = {b: runs[(nbody, nworld, res, b, "mjwarp")] for b in ("unpacked",) + tuned if (nbody, nworld, res, b, "mjwarp") in runs}
            if not ours or "native" not in mj:
                continue
            best = min((b for b in tuned if b in mj), key=lambda b: np.median(flat(mj[b])))
            bo, bd, bb = boot_median(flat(ours), rng), boot_median(flat(mj["native"]), rng), boot_median(flat(mj[best]), rng)
            sp_d, sp_b = bd / bo, bb / bo
            per_round = [np.median(mj[best][r]) / np.median(ours[r]) for r in ours if r in mj[best]]
            lo, hi = ci(sp_b)
            consistent = all(x > 1 for x in per_round) if lo > 1 else all(x < 1 for x in per_round) if hi < 1 else True
            verdict = ("ours faster" if lo > 1 else "ours slower" if hi < 1 else "no significant difference") if valid else "NO CLAIM"
            if valid and not consistent:
                verdict += " (rounds disagree: no claim)"
            so = summarize(flat(ours))
            rows.append({"nbody": nbody, "nworld": nworld, "res": res, "ours": so, "mjwarp": {b: summarize(flat(v)) for b, v in mj.items()}, "best_mjwarp": best,
                         "speedup_vs_default": [float(np.median(sp_d))] + [float(v) for v in ci(sp_d)], "speedup_vs_best": [float(np.median(sp_b)), float(lo), float(hi)],
                         "per_round_speedup_vs_best": [float(x) for x in per_round], "verdict": verdict})
            md, mb = summarize(flat(mj["native"]))["median"], summarize(flat(mj[best]))["median"]
            print(f"  {nworld:6d} {res:4d} {md:12.4f} ms {mb:10.4f} [{np.percentile(bb, 2.5):7.4f},{np.percentile(bb, 97.5):7.4f}]"
                  f" {so['median']:10.4f} [{np.percentile(bo, 2.5):7.4f},{np.percentile(bo, 97.5):7.4f}] {np.median(sp_d):6.2f}x"
                  f" {np.median(sp_b):7.2f}x [{lo:4.2f},{hi:4.2f}] r{min(per_round):.2f}-{max(per_round):.2f} {best:11s} {verdict}")
        print("  speedup = MJWarp ms / ours ms, so above 1 means neraulgraphics is faster. 'MJWarp default' is the matched-work configuration;")
        print("  'MJWarp best' is its fastest of the default and tuned variants for that case (tuned ones skip shading that neraulgraphics does).")
        print("  The bracketed intervals resample back-to-back repetitions and are optimistic; the r<min>-<max> range is the spread between")
        print("  independent rounds against the best MJWarp variant, the better measure of repeatability.")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
