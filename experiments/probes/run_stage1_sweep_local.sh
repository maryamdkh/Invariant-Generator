#!/usr/bin/env bash
# FEY-L1-S1 on a Mac: run stage1_sweep.py for every equation of a level,
# JOBS equations at a time (one thread each). Skips equations already done.
# Nothing is deleted: models and histories stay under results/ (gitignored).
#   caffeinate -i bash experiments/probes/run_stage1_sweep_local.sh
# Optional environment: LEVEL (L1), RUN (fey_l1_s1), SEED (42), JOBS (8).
set -euo pipefail
LEVEL="${LEVEL:-L1}"
RUN="${RUN:-fey_l1_s1}"
SEED="${SEED:-42}"
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1
ROOT="results/feynman_benchmark/${RUN}/seed${SEED}"
LOGS="experiments/outputs/${RUN}_logs"
mkdir -p "$ROOT" "$LOGS"

uv run python -c "import json; [print(e['equation_id']) for e in json.load(open('data/feynman_redundant/${LEVEL}/manifest.json'))]" |
while read -r EQ; do
    [ -f "$ROOT/$EQ/stage1_sweep.json" ] || echo "$EQ"
done |
xargs -P "$JOBS" -I{} sh -c "
    uv run python experiments/probes/stage1_sweep.py \
        --manifest data/feynman_redundant/${LEVEL}/manifest.json \
        --data-dir data/feynman_redundant/${LEVEL} \
        --results-dir $ROOT/{} --equations {} --seed ${SEED} > $LOGS/{}.log 2>&1;
    echo done {} \$(date +%H:%M)"
echo "all done: $(ls "$ROOT" | wc -l) equations in $ROOT"
