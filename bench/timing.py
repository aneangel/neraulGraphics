"""Timing and bookkeeping shared by the benchmark runners.

A measurement is `reps` repetitions of a block of frames, each block timed from a device sync to a device sync, so a result
is a list of milliseconds-per-frame samples that analyze.py can put confidence intervals on. Nothing here depends on any
renderer.
"""

import hashlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata

import numpy as np


def gpu_state():
    """Clocks, temperature, power and utilisation of GPU 0, so a drifting or throttled run can be spotted afterwards."""
    q = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return dict(zip(q.split(","), [float(x) for x in out.strip().split(",")]))
    except Exception:
        return {}


def environment(packages):
    """What ran: GPU, driver, Python, and the exact version of each package."""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"time": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: metadata.version(p) for p in packages}}


def poses_hash(*arrays):
    """Fingerprint of the poses a runner replays, so two result files can prove they were fed identical input."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def measure(frame, sync, reps=30, rep_seconds=0.05, warmup_seconds=1.5):
    """Calls frame(i) for i = 0, 1, ... and returns {'reps_ms': [...], 'frames_per_rep': K, 'warmup_frames': n}.

    Warmup runs for at least warmup_seconds so clocks, caches and lazily compiled kernels settle. The block length K is then
    chosen so one repetition lasts about rep_seconds, long enough that the sync at each end is a small part of it.
    """
    i = 0
    sync()
    start = time.perf_counter()
    while time.perf_counter() - start < warmup_seconds or i < 10:
        frame(i)
        i += 1
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        frame(i)
        i += 1
    sync()
    per_frame = (time.perf_counter() - t0) / 10
    k = int(min(2000, max(1, np.ceil(rep_seconds / per_frame))))
    samples = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for _ in range(k):
            frame(i)
            i += 1
        sync()
        samples.append((time.perf_counter() - t0) / k * 1e3)
    return {"reps_ms": samples, "frames_per_rep": k, "warmup_frames": i - 10 - k * reps}
