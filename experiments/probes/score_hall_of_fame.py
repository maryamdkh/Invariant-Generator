"""Score every equation of a yield pipeline's Stage 3 hall of fame on fresh
surface points (same scoring as the pipeline arm), to see which complexity
PySR's model selection should have picked.

    uv run python experiments/probes/score_hall_of_fame.py <config> <results-dir> --n <selected n> [--output json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from invariant_generator.adaptive_symbolic import _symbolic_inputs
from invariant_generator.adaptive import config_for_adaptive_n
from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.input_features import input_feature_names
from invariant_generator.sparsify import load_model_for_checkpoint
from invariant_generator.yield_benchmark import _evaluate_equation, _relative_l2, fresh_points, load_truth

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("results_dir", type=Path)
parser.add_argument("--n", type=int, required=True, help="selected n (benchmark_summary.json)")
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()

config = load_config(args.config)
config.train.device = "cpu"
name = Path(config.data.dataset_name).name
config.train.results_dir = args.results_dir
config.train.split_dir = args.results_dir / "splits"
run = args.results_dir / name
checkpoint_path = next(run.glob("stage2_*/checkpoint_best.pt"))
model = load_model_for_checkpoint(config_for_adaptive_n(config, args.n), checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
data = prepare_training_data(config)
X_fresh, y_fresh = fresh_points(config, *load_truth(config))


def J(X: np.ndarray) -> np.ndarray:
    return _symbolic_inputs(model, X, input_feature_names(config), device=torch.device("cpu"),
                            exact_coefficients=checkpoint.get("exact_raw_coefficients"),
                            transform=config.symbolic.encoded_feature_transform.lower())


J_fresh, J_test = J(X_fresh), J(data.X_test)
rows = []
for row in csv.DictReader((run / "stage3_pysr" / "equations.csv").open()):
    rows.append({
        "complexity": int(row["complexity"]), "loss": float(row["loss"]), "equation": row["equation"],
        "fresh_rel_l2": _relative_l2(_evaluate_equation(row["equation"], J_fresh), y_fresh),
        "test_rel_l2": _relative_l2(_evaluate_equation(row["equation"], J_test), data.y_test),
    })
    print(f"{rows[-1]['complexity']:3d}  loss {rows[-1]['loss']:.3e}  fresh {rows[-1]['fresh_rel_l2']:.3e}"
          f"  test {rows[-1]['test_rel_l2']:.3e}  {row['equation'][:90]}")
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
