# Environment for running experiments by hand inside an interactive job.
# From the repository root, on a compute node (not the login node):
#   source cluster/interactive_env.sh
# Then run e.g. "$PYTHON" scripts/run_yield_benchmark.py ...
# Same settings as the batch scripts (per-architecture venv and Julia depot,
# Julia's bundled libstdc++ preloaded).

ARCH="$(uname -m)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-$ARCH"
export JULIA_DEPOT_PATH="$PWD/.julia_depot-$ARCH"
export PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
JULIA_LIB="$UV_PROJECT_ENVIRONMENT/julia_env/pyjuliapkg/install/lib/julia"
if [ -f "$JULIA_LIB/libstdc++.so.6" ]; then
    export LD_PRELOAD="$JULIA_LIB/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# Torch and PySR share the allocation; keep BLAS from oversubscribing.
export OMP_NUM_THREADS=1

if [ ! -x "$PYTHON" ]; then
    echo "No environment for $ARCH at $UV_PROJECT_ENVIRONMENT (run cluster/setup_env.sh on this node type)." >&2
else
    echo "Using $PYTHON ($ARCH)"
fi
