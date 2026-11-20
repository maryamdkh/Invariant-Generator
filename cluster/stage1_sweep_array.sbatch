#!/usr/bin/env bash
# FEY-L1-S1: one array task = one equation; full Stage 1 sweep (every n,
# targets log y and y), no PySR. See experiments/probes/stage1_sweep.py.
# Submit from the repository root:
#   python cluster/make_tasks.py --manifest data/feynman_redundant/L1/manifest.json \
#       --seeds 42 --output cluster/tasks_l1.txt
#   sbatch -A naiss2026-4-599-gpu -p gpu --gpus=1 \
#       --array=1-$(wc -l < cluster/tasks_l1.txt)%40 cluster/stage1_sweep_array.sbatch
# Optional environment: CONFIG, LEVEL (L1), RUN (fey_l1_s1), TASKS.
#SBATCH -J stage1-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 02:00:00
#SBATCH --mem=8G
#SBATCH -o cluster/logs/%x-%A_%a.out
set -euo pipefail

CONFIG="${CONFIG:-configs/feynman_redundant_linear.toml}"
LEVEL="${LEVEL:-L1}"
RUN="${RUN:-fey_l1_s1}"
TASKS="${TASKS:-cluster/tasks_l1.txt}"
ARCH="$(uname -m)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-$ARCH"
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export OMP_NUM_THREADS=1

read -r EQUATION SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
OUT="results/feynman_benchmark/${RUN}/seed${SEED}/${EQUATION}"
mkdir -p "$OUT"
echo "task ${SLURM_ARRAY_TASK_ID}: ${EQUATION} seed ${SEED} -> ${OUT} ($(hostname))"

"$PYTHON" experiments/probes/stage1_sweep.py \
    --config "$CONFIG" \
    --manifest "data/feynman_redundant/${LEVEL}/manifest.json" \
    --data-dir "data/feynman_redundant/${LEVEL}" \
    --results-dir "$OUT" \
    --equations "$EQUATION" \
    --seed "$SEED"

# Keep only the compact sweep summary (models and histories are not needed).
find "$OUT" -mindepth 1 -maxdepth 1 ! -name stage1_sweep.json -exec rm -rf {} +
echo "kept: $(ls "$OUT")"
