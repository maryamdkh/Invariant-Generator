"""Stage 1 in log space: does the learned encoder row space match the true one?"""
import json, sys, time
from pathlib import Path
import numpy as np, sympy as sp
from invariant_generator.config import Config
from invariant_generator.adaptive import run_adaptive_sweep
from invariant_generator.sparsify import load_model_for_checkpoint
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.input_features import input_feature_names
from invariant_generator.feynman import parse_pmlb_metadata

SP = Path(sys.argv[1])

def base_config(name, variables):
    c = Config()
    c.data.kind = "supervised"; c.data.data_dir = Path("data/feynman").resolve(); c.data.dataset_name = name
    c.data.dataset_key = "X"; c.data.target_key = "y"; c.data.feature_names = variables
    c.augmentation.enabled = False
    c.input_transform.kind = "log"
    c.encoder.enabled = True; c.encoder.init = "random"
    c.normalization.enabled = True; c.normalization.mode = "standard"
    c.model.output_activation = "none"; c.model.standardize_target = True
    c.loss.lambda_structure = 0.0; c.loss.lambda_encoder_l1_ratio = 0.0; c.loss.lambda_encoder_l2 = 0.0
    c.train.results_dir = SP / "log_probe"; c.train.split_dir = SP / "log_probe" / "splits"; c.train.device = "cpu"
    c.train.epochs = 3000; c.train.learning_rate = 1e-3; c.train.encoder_learning_rate_scale = 30.0
    c.train.lr_plateau_min_delta = 0.0; c.train.early_stopping_min_delta = 0.0; c.train.log_every = 25
    c.adaptive.metric = "nrmse"; c.adaptive.nrmse_threshold = 0.01; c.adaptive.restarts = 3
    c.adaptive.n_min = 1; c.adaptive.n_max = len(variables) - 1; c.adaptive.results_subdir = name
    return c

def true_rowspace(pr):
    s = pr.symbols(); syms = [s[v] for v in pr.variables]; e = pr.expression()
    X, _ = pr.sample(400, seed=1)
    G = np.column_stack([np.broadcast_to(np.asarray(sp.lambdify(syms, sp.diff(e, v) * v, "numpy")(*X.T), float), (400,)) for v in syms])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    _, sv, Vt = np.linalg.svd(G, full_matrices=False)
    return Vt[: int((sv > 1e-6 * sv[0]).sum())]

for name in sys.argv[2].split(","):
    pr = parse_pmlb_metadata(f"data/feynman_tables/pmlb_metadata/{name}.yaml")
    cfg = base_config(name, pr.variables); t0 = time.time()
    sweep = run_adaptive_sweep(cfg)
    runs = [(r.n, round(r.test_metrics["nrmse"], 4)) for r in sweep.runs]
    msg = f"RESULT {name} d={len(pr.variables)} selected n={sweep.selected_n} runs={runs} time={time.time()-t0:.0f}s"
    if sweep.selected_n is not None:
        from invariant_generator.adaptive import config_for_adaptive_n
        rc = config_for_adaptive_n(cfg, sweep.selected_n)
        rep = encoder_formula_report(load_model_for_checkpoint(rc, sweep.selected_checkpoint), input_feature_names(rc))
        B = np.asarray(rep["raw_coefficients"])
        V = true_rowspace(pr)
        Q, _ = np.linalg.qr(B.T)  # learned row space basis (d x n)
        # principal angles: largest sin between subspaces when dims match
        if V.shape[0] == B.shape[0]:
            s = np.linalg.svd(V @ Q, compute_uv=False)
            msg += f" | max subspace angle sin={np.sqrt(max(0, 1 - s.min()**2)):.1e}"
        else:
            msg += f" | true n*={V.shape[0]}"
        msg += f" | B={np.round(B, 3).tolist()}"
    print(msg, flush=True)
