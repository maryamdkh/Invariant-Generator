#!/usr/bin/env bash
# One-time setup on the cluster LOGIN node (compute nodes may have no internet).
# Run from the repository root:  bash cluster/setup_env.sh
set -euo pipefail

# Keep Julia (used by PySR) inside the project, not in a small home quota.
export JULIA_DEPOT_PATH="$PWD/.julia_depot"
# Julia (PySR) needs a newer libstdc++ than many clusters ship in /lib64
# (GLIBCXX_3.4.30 not found); use the copy bundled with Julia.
export LD_LIBRARY_PATH="$PWD/.venv/julia_env/pyjuliapkg/install/lib/julia:${LD_LIBRARY_PATH:-}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra symbolic

# First PySR fit downloads Julia and precompiles SymbolicRegression.jl.
.venv/bin/python - <<'EOF'
import numpy as np
from pysr import PySRRegressor
X = np.random.default_rng(0).uniform(1, 2, size=(50, 2))
PySRRegressor(niterations=2, progress=False).fit(X, X[:, 0] * X[:, 1])
print("PySR ready")
EOF

# Feynman data: formulas are in the repo; samples are regenerated (noiseless).
.venv/bin/python scripts/prepare_feynman_benchmark.py --selection all
echo "Setup done."
