"""Fit sigma^T M sigma = 1 by linear least squares on the 21 quadratic monomials.

Recovers the true (Mandel) tensor M of each quadratic yield dataset; this is
the ground truth for structure recovery and the SINDy-style baseline (a
linear fit on quadratic monomials) for the Hill ladder.

    uv run python experiments/probes/hill_quadratic_fit.py
Writes experiments/outputs/hill_quadratic_fit.json.
"""
import json
from itertools import combinations_with_replacement
from pathlib import Path

import h5py
import numpy as np

DATASETS = ["vonMises25", "vonMises100", "hill", "rotatedhill"]
PAIRS = list(combinations_with_replacement(range(6), 2))
out = {}
for name in DATASETS:
    s = np.asarray(h5py.File(f"data/{name}.hdf", "r")["stress"][()], float)  # Mandel
    X = np.column_stack([s[:, i] * s[:, j] for i, j in PAIRS])
    c, *_ = np.linalg.lstsq(X, np.ones(len(s)), rcond=None)
    M = np.zeros((6, 6))
    for (i, j), v in zip(PAIRS, c):
        M[i, j] = M[j, i] = v if i == j else v / 2
    eig = np.linalg.eigvalsh(M)
    out[name] = {
        "points": len(s),
        "max_abs_residual": float(np.abs(X @ c - 1).max()),
        "nonzero_terms": int((np.abs(c) > 1e-8 * np.abs(c).max()).sum()),
        "eigenvalues": eig.tolist(),
        "M_mandel": M.tolist(),
        "all_components_nonnegative": bool((s >= 0).all()),
    }
    print(f"{name:12s} residual {out[name]['max_abs_residual']:.1e}  terms {out[name]['nonzero_terms']}/21"
          f"  eig/max {np.round(eig / eig.max(), 4)}")
Path("experiments/outputs/hill_quadratic_fit.json").write_text(json.dumps(out, indent=2) + "\n")
