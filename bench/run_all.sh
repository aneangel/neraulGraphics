#!/usr/bin/env bash
# The whole comparison against MuJoCo Warp, end to end.
#
#   bench/run_all.sh COMPETITORS_PYTHON OURS_PYTHON [ROUNDS] [OUT_DIR]
#
# Each python is the interpreter of an environment built from bench/envs/*.lock.txt (see bench/README.md). Poses are recorded
# once per scene size, then each round runs both systems with the order alternating, so slow drift (GPU temperature, clocks,
# background load) lands on both systems and not on one. analyze.py then reads the results.
set -euo pipefail
COMP=${1:?competitors python}; OURS=${2:?ours python}; ROUNDS=${3:-5}; OUT=${4:-bench/results/run1}
HERE=$(cd "$(dirname "$0")" && pwd); DATA="$HERE/data"; mkdir -p "$DATA" "$OUT"
QUIET='^Module|^Warp|CUDA|Devices|"cpu"|"cuda|Kernel cache|/.cache'

for NBODY in 8 32; do
  [ -f "$DATA/poses_$NBODY.npz" ] || "$COMP" -I "$HERE/gen_poses.py" --nbody "$NBODY" --out "$DATA/poses_$NBODY.npz" 2>&1 | grep -Ev "$QUIET"
done

for ROUND in $(seq 1 "$ROUNDS"); do
  if [ $((ROUND % 2)) -eq 1 ]; then ORDER="mjwarp ours"; else ORDER="ours mjwarp"; fi
  for SYSTEM in $ORDER; do
    for NBODY in 8 32; do
      PY=$COMP; [ "$SYSTEM" = ours ] && PY=$OURS
      EXTRA=(); [ "$ROUND" -eq 1 ] && EXTRA=(--verify "$OUT/verify_${SYSTEM}_${NBODY}.npz")
      echo "--- round $ROUND $SYSTEM nbody=$NBODY"
      "$PY" -I "$HERE/run_$SYSTEM.py" --poses "$DATA/poses_$NBODY.npz" --out "$OUT/r${ROUND}_${SYSTEM}_${NBODY}.json" "${EXTRA[@]}" 2>&1 | grep -Ev "$QUIET"
    done
  done
done
python3 "$HERE/analyze.py" "$OUT" --json "$OUT/summary.json" | tee "$OUT/summary.txt"
