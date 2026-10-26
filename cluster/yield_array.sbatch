#!/usr/bin/env bash
# One array task = one (yield config, seed) from cluster/tasks_yield.txt, all arms
# (scripts/run_yield_benchmark.py). Task lines: "<config path> <seed>", e.g.
#   configs/hill_ladder/rh3.toml 42
# Submit from the repository root, e.g.
#   RUN=hill_ladder ARMS=pipeline,baseline sbatch -A <project> -p gpu --gpus=1 \
#     --array=1-$(wc -l < cluster/tasks_yield.txt) cluster/yield_array.sbatch
# Environment: RUN (results subfolder), ARMS (pipeline, baseline, distillation,
# sparse_linear, library_pysr), TIMEOUT_MINUTES (PySR limit per arm, same for
# all arms), TASKS, KEEP_ARTIFACTS (pack | summary | full; see
# feynman_array.sbatch).
#SBATCH -J yield-bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH -o cluster/logs/%x-%A_%a.out
set -euo pipefail

RUN="${RUN:-hill_ladder}"
ARMS="${ARMS:-pipeline,baseline}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-15}"
TASKS="${TASKS:-cluster/tasks_yield.txt}"
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

read -r CONFIG SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
NAME="$(basename "$CONFIG" .toml)"
OUT="results/yield_benchmark/${RUN}/seed${SEED}/${NAME}"
mkdir -p "$OUT"
echo "task ${SLURM_ARRAY_TASK_ID}: ${CONFIG} seed ${SEED} -> ${OUT} ($(hostname), $(git rev-parse --short HEAD))"

"$PYTHON" scripts/run_yield_benchmark.py \
    --config "$CONFIG" \
    --results-dir "$OUT" \
    --arms "$ARMS" \
    --seed "$SEED" \
    --timeout-minutes "$TIMEOUT_MINUTES" \
    --numprocs "${SLURM_CPUS_PER_TASK:-16}" \
    --device cpu

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
