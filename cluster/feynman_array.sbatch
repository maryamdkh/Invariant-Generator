#!/usr/bin/env bash
# One array task = one (equation, seed) from cluster/tasks.txt, all arms.
# Submit from the repository root, e.g.
#   sbatch -A <project> --array=1-$(wc -l < cluster/tasks.txt) cluster/feynman_array.sbatch
# Optional environment: CONFIG (default configs/feynman_benchmark_v2.toml),
# RUN (results subfolder name), ARMS, TIMEOUT_MINUTES, and for other
# benchmarks MANIFEST, DATA_DIR and TASKS (e.g. LLM-SRBench, see README).
#SBATCH -J feynman-bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o cluster/logs/%x-%A_%a.out
set -euo pipefail

CONFIG="${CONFIG:-configs/feynman_benchmark_v2.toml}"
RUN="${RUN:-v2}"
ARMS="${ARMS:-pipeline,portfolio,baseline}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-15}"
MANIFEST="${MANIFEST:-data/feynman/manifest.json}"
DATA_DIR="${DATA_DIR:-data/feynman}"
TASKS="${TASKS:-cluster/tasks.txt}"
# Login and GPU nodes can differ in CPU architecture (x86_64 vs aarch64),
# so keep one environment and Julia depot per architecture.
ARCH="$(uname -m)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-$ARCH"
export JULIA_DEPOT_PATH="$PWD/.julia_depot-$ARCH"
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
# Julia (PySR) needs a newer libstdc++ than many clusters ship in /lib64
# (GLIBCXX_3.4.30 not found); use the copy bundled with Julia.
export LD_LIBRARY_PATH="$UV_PROJECT_ENVIRONMENT/julia_env/pyjuliapkg/install/lib/julia:${LD_LIBRARY_PATH:-}"
# Torch and PySR share the allocation; keep BLAS from oversubscribing.
export OMP_NUM_THREADS=1

read -r EQUATION SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
OUT="results/feynman_benchmark/${RUN}/seed${SEED}/${EQUATION}"
mkdir -p "$OUT"
echo "task ${SLURM_ARRAY_TASK_ID}: ${EQUATION} seed ${SEED} -> ${OUT} ($(hostname))"

"$PYTHON" scripts/run_feynman_benchmark.py \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --data-dir "$DATA_DIR" \
    --results-dir "$OUT" \
    --equations "$EQUATION" \
    --seed "$SEED" \
    --arms "$ARMS" \
    --timeout-minutes "$TIMEOUT_MINUTES" \
    --numprocs "${SLURM_CPUS_PER_TASK:-16}"
