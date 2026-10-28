"""Best polynomial approximation of Yld2004 (a = 8) by degree: least squares
of y^m on all monomials of degree <= m (m = 2, 3, 4, 6, 8), scored on fresh
surface points. Shows how far each feature degree can get.

    uv run python experiments/probes/yld_polynomial_floor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.yield_benchmark import _polynomial_library, _relative_l2, fresh_points, load_truth

config = load_config("configs/yld/yld2004_2090T3.toml")
data = prepare_training_data(config)
X_fresh, y_fresh = fresh_points(config, *load_truth(config))
results = {}
for m in (2, 3, 4, 6, 8):
    Theta, _ = _polynomial_library(data.X_train, m)
    scale = np.linalg.norm(Theta, axis=0)
    xi, *_ = np.linalg.lstsq(Theta / scale, data.y_train**m, rcond=None)
    prediction = np.clip(_polynomial_library(X_fresh, m)[0] @ (xi / scale), 0, None) ** (1 / m)
    results[m] = _relative_l2(prediction, y_fresh)
    print(f"y^{m} = degree-{m} polynomial ({Theta.shape[1]} terms): fresh relative L2 {results[m]:.2e}")
Path("experiments/outputs").mkdir(exist_ok=True)
Path("experiments/outputs/yld_polynomial_floor.json").write_text(json.dumps(results, indent=2) + "\n")
