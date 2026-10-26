#!/usr/bin/env bash
# One array task = one (equation, seed) from cluster/tasks.txt, all arms.
# Submit from the repository root, e.g.
#   sbatch -A <project> --array=1-$(wc -l < cluster/tasks.txt) cluster/feynman_array.sbatch
# Optional environment: CONFIG (default configs/feynman_benchmark_v2.toml),
# RUN (results subfolder name), ARMS, TIMEOUT_MINUTES, TARGET_NOISE (e.g.
# 0.01 for 1% target noise; default 0), and for other
# benchmarks MANIFEST, DATA_DIR and TASKS (e.g. LLM-SRBench, see README).
# KEEP_ARTIFACTS (file quota): pack (default) keeps benchmark_summary.json
# plus one artifacts.tar.gz per task, without models (*.pt) and PySR
# checkpoints (*.pkl, *.bak); summary keeps only benchmark_summary.json;
# full keeps everything.
#SBATCH -J feynman-bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH --mem=64G
#SBATCH -o cluster/logs/%x-%A_%a.out
set -euo pipefail

CONFIG="${CONFIG:-configs/feynman_benchmark_v2.toml}"
RUN="${RUN:-v2}"
ARMS="${ARMS:-pipeline,portfolio,baseline}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-15}"
MANIFEST="${MANIFEST:-data/feynman/manifest.json}"
DATA_DIR="${DATA_DIR:-data/feynman}"
TASKS="${TASKS:-cluster/tasks.txt}"
TARGET_NOISE="${TARGET_NOISE:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-pack}"
# Login and GPU nodes can differ in CPU architecture (x86_64 vs aarch64),
# so keep one environment and Julia depot per architecture.
ARCH="$(uname -m)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-$ARCH"
export JULIA_DEPOT_PATH="$PWD/.julia_depot-$ARCH"
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
# Julia (PySR) needs a newer libstdc++ than many clusters ship in /lib64
# (GLIBCXX_3.4.30 not found). Preload only the copy bundled with Julia;
# putting its whole lib directory on LD_LIBRARY_PATH breaks Julia's own
# loader (libLLVM not found).
JULIA_LIB="$UV_PROJECT_ENVIRONMENT/julia_env/pyjuliapkg/install/lib/julia"
preload_julia_libstdcxx() {
    if [ -f "$JULIA_LIB/libstdc++.so.6" ]; then
        export LD_PRELOAD="$JULIA_LIB/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
}
preload_julia_libstdcxx
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
    --target-noise "$TARGET_NOISE" \
    --numprocs "${SLURM_CPUS_PER_TASK:-16}"

# One task otherwise leaves ~100 files (models, PySR checkpoints, splits).
case "$KEEP_ARTIFACTS" in
    full) ;;
    pack|summary)
        if [ "$KEEP_ARTIFACTS" = pack ]; then
            tar czf "$OUT.artifacts.tar.gz" -C "$OUT" \
                --exclude ./benchmark_summary.json --exclude '*.pt' --exclude '*.pkl' --exclude '*.bak' .
            mv "$OUT.artifacts.tar.gz" "$OUT/artifacts.tar.gz"
        fi
        find "$OUT" -mindepth 1 -maxdepth 1 ! -name benchmark_summary.json ! -name artifacts.tar.gz \
            -exec rm -rf {} +
        ;;
    *) echo "Unknown KEEP_ARTIFACTS=$KEEP_ARTIFACTS (pack | summary | full)" >&2; exit 1 ;;
esac
echo "kept: $(ls "$OUT")"
