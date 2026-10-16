"""Stage 1 n=1 conditioning probe: normalization mode, epochs, seeds."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from invariant_generator.config import load_config
from invariant_generator.adaptive import config_for_adaptive_n, evaluate_checkpoint_on_train_and_test
from invariant_generator.train import train_from_config
from invariant_generator.sparsify import load_model_for_checkpoint
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.input_features import input_feature_names
import h5py

SP = Path(sys.argv[1])
with h5py.File("data/feynman_gaussian/i_6_20a_sum.h5") as f: X = f["X"][:]; y = f["y"][:]

def g_rmse(J, deg=12):
    z = (J - J.mean()) / J.std(); P = np.polynomial.chebyshev.chebvander(z / np.abs(z).max(), deg)
    c, *_ = np.linalg.lstsq(P, y, rcond=None); return float(np.sqrt(((P @ c - y) ** 2).mean()))

variants = [(f"standard 3000 seed{s}", "standard", 3000, s) for s in range(42, 50)] + [(f"scale_only 3000 seed{s}", "scale_only", 3000, s) for s in (7, 47, 48, 49)]
_unused = [
    ("scale_only 3000", "scale_only", 3000, 42),
    ("standard 3000", "standard", 3000, 42),
    ("standard 3000 seed7", "standard", 3000, 7),
    ("scale_only 8000", "scale_only", 8000, 42),
]
for label, mode, epochs, seed in variants:
    cfg = config_for_adaptive_n(load_config("configs/adaptive_encoder_feynman_gaussian_sum_v2.toml"), 1)
    cfg.train.results_dir = SP / "stage1_probe"; cfg.train.run_id = label.replace(" ", "_")
    cfg.normalization.mode = mode; cfg.train.epochs = epochs; cfg.train.seed = seed
    cfg.train.device = "cpu"
    t0 = time.time(); res = train_from_config(cfg)
    tr, va = evaluate_checkpoint_on_train_and_test(cfg, res.best_checkpoint)
    model = load_model_for_checkpoint(cfg, res.best_checkpoint)
    rep = encoder_formula_report(model, input_feature_names(cfg))
    beta = np.asarray(rep["raw_coefficients"][0]); b0 = rep["raw_intercept"][0]
    feats = np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2, X[:, 0]*X[:, 1], X[:, 1]**2, X[:, 0]/X[:, 1], X[:, 1]/X[:, 0]])
    J = feats @ beta + b0
    print(f"RESULT {label:22s} val_nrmse={va['nrmse']:.4f} g(J)_floor={g_rmse(J):.1e} "
          f"time={time.time()-t0:.0f}s raw={np.round(beta, 3).tolist()}", flush=True)
