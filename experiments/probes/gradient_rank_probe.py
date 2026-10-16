"""Can a full-width MLP's gradient spectrum (w.r.t. log x) estimate n*?

For each pilot equation: train one MLP on log features with no bottleneck,
compute row-normalized gradients d f / d log x on training points, SVD them,
and compare candidate rank rules with the analysis n*.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from invariant_generator.config import Config
from invariant_generator.train import train_from_config
from invariant_generator.data import prepare_training_data
from invariant_generator.sparsify import load_model_for_checkpoint

SP = Path(sys.argv[1]); target_log = sys.argv[2] == "log"
manifest = {e["equation_id"]: e for e in json.load(open("data/feynman/manifest.json"))}
for name, entry in manifest.items():
    c = Config()
    c.data.kind = "supervised"; c.data.data_dir = Path("data/feynman").resolve(); c.data.dataset_name = name
    c.data.dataset_key = "X"; c.data.target_key = "y"; c.data.feature_names = entry["variables"]
    c.augmentation.enabled = False; c.input_transform.kind = "log"
    c.encoder.enabled = False
    c.normalization.enabled = True; c.normalization.mode = "standard"
    c.model.output_activation = "none"; c.model.standardize_target = True
    c.loss.lambda_structure = 0.0; c.loss.lambda_encoder_l1_ratio = 0.0; c.loss.lambda_encoder_l2 = 0.0
    c.train.results_dir = SP / f"grad_probe_{sys.argv[2]}"; c.train.split_dir = c.train.results_dir / "splits"
    c.train.run_id = name; c.train.device = "cpu"; c.train.epochs = 3000; c.train.log_every = 25
    c.train.lr_plateau_min_delta = 0.0; c.train.early_stopping_min_delta = 0.0; c.train.save_every = 0
    if target_log:
        # Fit log y: write a transformed copy of the dataset.
        import h5py
        src = Path("data/feynman") / f"{name}.h5"; dst_dir = SP / "logy"; dst_dir.mkdir(exist_ok=True)
        with h5py.File(src) as f: X, y = f["X"][:], f["y"][:]
        if np.any(y <= 0): print(f"SKIP {name}: nonpositive y"); continue
        with h5py.File(dst_dir / f"{name}.h5", "w") as f: f.create_dataset("X", data=X); f.create_dataset("y", data=np.log(y))
        c.data.data_dir = dst_dir
    t0 = time.time(); res = train_from_config(c)
    data = prepare_training_data(c); model = load_model_for_checkpoint(c, res.best_checkpoint)
    X = torch.tensor(data.X_train, dtype=torch.float32, requires_grad=True)
    model.eval(); out = model(X).sum(); (g,) = torch.autograd.grad(out, X)
    G = (g * X).detach().numpy().astype(np.float64)  # d f / d log x
    G /= np.linalg.norm(G, axis=1, keepdims=True).clip(1e-300)
    s = np.linalg.svd(G, compute_uv=False); s = s / s[0]
    val = json.load(open(res.history_path))["final_metrics"]["rmse"] / data.y_validation.std()
    gaps = s[:-1] / np.maximum(s[1:], 1e-300)
    n_gap = int(np.argmax(gaps) + 1) if len(s) > 1 else 1
    print(f"RESULT {name:18s} d={len(s)} n*={entry['log_reducibility']} val_nrmse={val:.1e} "
          f"n_gap={n_gap} n_1e-2={int((s > 1e-2).sum())} spectrum={np.array2string(s, precision=3, separator=',')} "
          f"t={time.time()-t0:.0f}s", flush=True)
