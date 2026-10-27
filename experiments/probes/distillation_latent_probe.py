"""Distillation (C) network only: how many latent components survive the L1,
and how well does one kept component explain y? No PySR.

    uv run python experiments/probes/distillation_latent_probe.py <l1_weight> configs/hill_ladder/vm25.toml ...
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.distillation import DistillationSettings, latent, train_distillation_net

L1 = float(sys.argv[1])
results = {}
for path in sys.argv[2:]:
    config = load_config(path)
    data = prepare_training_data(config)
    active = [j for j in range(data.X_train.shape[1]) if np.any(data.X_train[:, j] != 0)]
    X = data.X_train[:, active]
    started = time.perf_counter()
    net, info = train_distillation_net(X, data.y_train, DistillationSettings(seed=config.train.seed, l1_weight=L1))
    Z = latent(net, X)
    importance = Z.std(axis=0) * net.latent_weight_norms().detach().double().numpy()
    importance = importance / importance.max()
    top = int(np.argmax(importance))
    # Is y a function of the top component alone? Polynomial fit of y on z_top.
    coeffs = np.polyfit(Z[:, top], data.y_train, 6)
    residual = np.linalg.norm(np.polyval(coeffs, Z[:, top]) - data.y_train) / np.linalg.norm(data.y_train)
    results[path] = {
        "train_nrmse": info["train_nrmse"], "seconds": time.perf_counter() - started,
        "importance_over_max": np.round(np.sort(importance)[::-1], 4).tolist(),
        "y_from_top_component_rel_l2": float(residual),
    }
    print(path, json.dumps(results[path]), flush=True)
