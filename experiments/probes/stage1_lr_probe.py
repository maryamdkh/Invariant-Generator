import sys, time
from pathlib import Path
import numpy as np, h5py
from invariant_generator.config import load_config
from invariant_generator.adaptive import config_for_adaptive_n, evaluate_checkpoint_on_train_and_test
from invariant_generator.train import train_from_config
from invariant_generator.sparsify import load_model_for_checkpoint
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.input_features import input_feature_names
SP = Path(sys.argv[1])
with h5py.File("data/feynman_gaussian/i_6_20a_sum.h5") as f: X = f["X"][:]; y = f["y"][:]
feats = np.column_stack([X[:,0], X[:,1], X[:,0]**2, X[:,0]*X[:,1], X[:,1]**2, X[:,0]/X[:,1], X[:,1]/X[:,0]])
def g_rmse(J, deg=12):
    z=(J-J.mean())/J.std(); P=np.polynomial.chebyshev.chebvander(z/np.abs(z).max(),deg)
    c,*_=np.linalg.lstsq(P,y,rcond=None); return float(np.sqrt(((P@c-y)**2).mean()))
for scale in [float(s) for s in sys.argv[2].split(",")]:
    for seed in range(42, 47):
        cfg = config_for_adaptive_n(load_config("configs/adaptive_encoder_feynman_gaussian_sum_v2.toml"), 1)
        cfg.train.results_dir = SP / "lr_probe"; cfg.train.run_id = f"s{scale}_seed{seed}"
        cfg.train.seed = seed; cfg.train.device = "cpu"; cfg.train.encoder_learning_rate_scale = scale
        res = train_from_config(cfg); _, va = evaluate_checkpoint_on_train_and_test(cfg, res.best_checkpoint)
        rep = encoder_formula_report(load_model_for_checkpoint(cfg, res.best_checkpoint), input_feature_names(cfg))
        J = feats @ np.asarray(rep["raw_coefficients"][0])
        print(f"RESULT scale={scale:<4} seed={seed} val_nrmse={va['nrmse']:.4f} g(J)_floor={g_rmse(J):.1e}", flush=True)
