#!/usr/bin/env bash
# One-time setup, once per node architecture. Run from the repository root:
#   bash cluster/setup_env.sh            (login node)
# and inside an interactive job on the nodes the array will use, e.g.
#   srun -A <project> -p gpu --gpus=1 -t 01:00:00 --pty bash cluster/setup_env.sh
set -euo pipefail

# Keep Julia (used by PySR) inside the project, not in a small home quota.
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

# uv is a native binary too: keep one per architecture.
UV_DIR="$HOME/.local/uv-$ARCH"
if ! "$UV_DIR/uv" --version >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh
fi

"$UV_DIR/uv" sync --extra symbolic --extra benchmarks

# The first import downloads Julia into the environment; it may fail on the
# old system libstdc++ before the bundled copy exists, so preload and retry.
"$PYTHON" -c "import juliacall" || true
preload_julia_libstdcxx
if [ ! -f "$JULIA_LIB/libstdc++.so.6" ]; then
    echo "Note: no bundled libstdc++.so.6 in $JULIA_LIB; relying on the system one."
fi

# First PySR fit downloads Julia and precompiles SymbolicRegression.jl.
"$PYTHON" - <<'EOF'
import numpy as np
from pysr import PySRRegressor
X = np.random.default_rng(0).uniform(1, 2, size=(50, 2))
PySRRegressor(niterations=2, progress=False).fit(X, X[:, 0] * X[:, 1])
print("PySR ready")
EOF

# Feynman data: formulas are in the repo; samples are regenerated (noiseless).
"$PYTHON" scripts/prepare_feynman_benchmark.py --selection all
echo "Setup done for $ARCH: $UV_PROJECT_ENVIRONMENT"
