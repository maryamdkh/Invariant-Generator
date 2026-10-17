"""Rotated Hill must not change: run a tiny Stage 1 + Stage 2 on two code versions.

Usage (repo root):  uv run python experiments/probes/hill_old_vs_new.py [OLD_REF]
OLD_REF defaults to HEAD, so uncommitted changes are compared with the last
commit. Prints IDENTICAL or the keys that differ (data, Stage 1 weights and
metrics, Stage 2 S/mask/metrics) and exits non-zero on a difference.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]

RUN = r'''
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from invariant_generator.config import load_config
from invariant_generator.adaptive import config_for_adaptive_n, run_adaptive_sweep
from invariant_generator.data import prepare_training_data
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint

out = Path(sys.argv[1])
cfg = load_config("configs/adaptive_encoder_rotated_hill.toml")
cfg.train.results_dir = out
cfg.train.split_dir = out / "splits"
cfg.train.device = "cpu"
cfg.train.epochs = 40
cfg.train.log_every = 10
cfg.adaptive.results_subdir = "hill_tiny"
cfg.adaptive.n_max = 1
cfg.adaptive.rmse_threshold = 10.0  # accept n=1 so the tiny run proceeds
cfg.sparsification.epochs = 15
cfg.sparsification.masked_refit_epochs = 15
cfg.sparsification.max_active_terms_candidates = [1, 2]
d = prepare_training_data(cfg)
# Hill has no validation split (None): hash only real arrays.
arrays = [a for a in (d.X_train, d.X_validation, d.X_test) if a is not None]
data_hash = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()
sweep = run_adaptive_sweep(cfg)
ckpt = torch.load(sweep.selected_checkpoint, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
weights = hashlib.sha256(b"".join(
    v.numpy().tobytes() for _, v in sorted(state.items()) if hasattr(v, "numpy")
)).hexdigest()
sparse = sparsify_encoder_from_checkpoint(config_for_adaptive_n(cfg, 1), checkpoint_path=sweep.selected_checkpoint)
s = json.loads(sparse.summary_path.read_text())
keys = ["final_sparse_S", "mask", "trained_dense_S", "train_metrics", "test_metrics",
        "selected_max_active_terms_per_row"]
print("RESULT " + json.dumps({
    "data": data_hash, "stage1_weights": weights,
    "stage1_runs": [(r.n, r.test_metrics) for r in sweep.runs],
    **{k: s[k] for k in keys},
}, sort_keys=True, default=str))
'''


def run(tree: Path, script: Path, out: Path) -> dict:
    result = subprocess.run(
        ["uv", "run", "--project", str(ROOT), "python", str(script), str(out)],
        cwd=tree, env={**__import__("os").environ, "PYTHONPATH": str(tree / "src")},
        capture_output=True, text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if result.returncode != 0 or not lines:
        sys.exit(f"run failed in {tree}:\n{result.stderr[-3000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def main() -> None:
    old_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "hill_tiny.py"
        script.write_text(RUN)
        old_tree = tmp / "old"
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(old_tree), old_ref], cwd=ROOT, check=True)
        try:
            # Untracked data (data/*.hdf) is not in the worktree: share the current copy.
            (old_tree / "data").rename(tmp / "old_data_tracked")
            (old_tree / "data").symlink_to(ROOT / "data")
            old = run(old_tree, script, tmp / "out_old")
            new = run(ROOT, script, tmp / "out_new")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(old_tree)], cwd=ROOT, check=False)
    differing = [key for key in old if old[key] != new.get(key)]
    if differing:
        print(f"DIFFERENT vs {old_ref}: {differing}")
        sys.exit(1)
    print(f"IDENTICAL to {old_ref}: " + ", ".join(sorted(old)))


if __name__ == "__main__":
    main()
