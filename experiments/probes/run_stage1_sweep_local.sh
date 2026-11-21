#!/usr/bin/env bash
# FEY-L1-S1 on a Mac: run stage1_sweep.py for every equation of a level,
# JOBS equations at a time (one thread each). Skips equations already done.
# Nothing is deleted: models and histories stay under results/ (gitignored).
#   caffeinate -i bash experiments/probes/run_stage1_sweep_local.sh
# Optional environment: LEVEL (L1), RUN (fey_l1_s1), SEED (42), JOBS (8).
set -euo pipefail
export LEVEL="${LEVEL:-L1}"
export RUN="${RUN:-fey_l1_s1}"
export SEED="${SEED:-42}"
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1
export ROOT="results/feynman_benchmark/${RUN}/seed${SEED}"
export LOGS="experiments/outputs/${RUN}_logs"
mkdir -p "$ROOT" "$LOGS"

one() {
    # One equation; the equation id is $1. (BSD xargs -I limits the
    # command length, so the id is passed as an argument instead.)
    uv run python experiments/probes/stage1_sweep.py \
        --manifest "data/feynman_redundant/${LEVEL}/manifest.json" \
        --data-dir "data/feynman_redundant/${LEVEL}" \
        --results-dir "$ROOT/$1" --equations "$1" --seed "$SEED" > "$LOGS/$1.log" 2>&1 \
        && echo "done $1 $(date +%H:%M)" || echo "FAILED $1 $(date +%H:%M)"
}
export -f one

uv run python -c "import json; [print(e['equation_id']) for e in json.load(open('data/feynman_redundant/${LEVEL}/manifest.json'))]" |
while read -r EQ; do
    [ -f "$ROOT/$EQ/stage1_sweep.json" ] || echo "$EQ"
done |
xargs -P "$JOBS" -n 1 bash -c 'one "$1"' _
echo "all done: $(ls "$ROOT" | wc -l) equations in $ROOT"
