"""Old vs new Stage 2 on a tiny rotated-Hill run: outputs must match exactly."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from invariant_generator.config import load_config
from invariant_generator.adaptive import config_for_adaptive_n, run_adaptive_sweep
from invariant_generator import sparsify as new

SP = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("sparsify_old", SP / "orig" / "sparsify.py")
old = importlib.util.module_from_spec(spec); sys.modules["sparsify_old"] = old; spec.loader.exec_module(old)

cfg = load_config("configs/adaptive_encoder_rotated_hill.toml")
cfg.train.results_dir = SP / "hill_results"
cfg.train.device = "cpu"
cfg.train.epochs = 40
cfg.train.log_every = 10
cfg.adaptive.results_subdir = "hill_regression"
cfg.adaptive.n_max = 1
cfg.adaptive.rmse_threshold = 10.0  # accept n=1 so the tiny run proceeds
cfg.sparsification.epochs = 15
cfg.sparsification.masked_refit_epochs = 15
cfg.sparsification.max_active_terms_candidates = [1, 2]
sweep = run_adaptive_sweep(cfg)
stage = config_for_adaptive_n(cfg, 1)
r_old = old.sparsify_encoder_from_checkpoint(stage, checkpoint_path=sweep.selected_checkpoint, run_id="hill_regression/old")
r_new = new.sparsify_encoder_from_checkpoint(stage, checkpoint_path=sweep.selected_checkpoint, run_id="hill_regression/new")
a = json.loads(r_old.summary_path.read_text()); b = json.loads(r_new.summary_path.read_text())
for key in ["final_sparse_S", "mask", "trained_dense_S", "train_metrics", "test_metrics", "selected_max_active_terms_per_row"]:
    same = a[key] == b[key]
    print(f"{key:36s} identical={same}")
    assert same, (a[key], b[key])
print("extra keys in new summary:", sorted(set(b) - set(a)))
print("HILL REGRESSION OK")
