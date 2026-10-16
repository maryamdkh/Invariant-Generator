"""Stages 2-3 on a yield benchmark from a finished Stage 1, then compare the
recovered formula with the generating criterion on fresh stress directions.

Usage (after yield_stage1_probe.py):
    python experiments/probes/yield_pipeline_probe.py configs/yield_cb2001_2090T3.toml
"""
import json, sys
from importlib import import_module
import_module("pysr")  # Julia before Torch
import numpy as np, torch
from invariant_generator.adaptive import (
    adaptive_sparsification_run_id, adaptive_stage1_dir, config_for_adaptive_n,
)
from invariant_generator.adaptive_symbolic import train_encoded_symbolic_from_config
from invariant_generator.config import load_config
from invariant_generator.sparsify import load_model_for_checkpoint, sparsify_encoder_from_checkpoint
from invariant_generator.yield_criteria import CRITERIA

config = load_config(sys.argv[1])
stage1 = json.loads((adaptive_stage1_dir(config) / config.adaptive.summary_name).read_text())
n = stage1["selected_n"]
stage_config = config_for_adaptive_n(config, n)
sparse = sparsify_encoder_from_checkpoint(stage_config, checkpoint_path=stage1["selected_checkpoint"])
summary = json.loads(sparse.summary_path.read_text())
print("stage2", summary["method"], "accepted:", summary["snapped"], "test metrics", summary["test_metrics"].get(config.adaptive.metric),
      "candidates", [(c["denominator"], round(c["rounding_residual"], 6), round(c["test_metrics"][config.adaptive.metric], 5)) for c in summary["candidate_results"]])
for formula in summary["formulas"]["formulas"]:
    print("  ", formula["name"], "terms:", formula["active_count"])

symbolic_config = config_for_adaptive_n(config, n)
symbolic_config.train.run_id = adaptive_sparsification_run_id(symbolic_config)
result = train_encoded_symbolic_from_config(symbolic_config, checkpoint_path=sparse.checkpoint_path)
print("best equation in J:", result.best_equation)

# Fresh stress directions: recovered f(J(sigma)) vs the true equivalent stress.
criterion = CRITERIA[config.data.dataset_name]
model = load_model_for_checkpoint(stage_config, sparse.checkpoint_path)
X = np.random.default_rng(7).normal(size=(5000, 6))
with torch.no_grad():
    J = model.invariant_features(torch.as_tensor(X, dtype=torch.float32)).double().numpy()
# Evaluate the selected equation (our own PySR output) with numpy operators.
namespace = {"square": np.square, "cube": lambda v: v**3, "cbrt": np.cbrt, "sqrt": np.sqrt,
             **{f"J{i + 1}": J[:, i] for i in range(J.shape[1])}}
with np.errstate(all="ignore"):
    prediction = eval(result.best_equation, {"__builtins__": {}}, namespace)
truth = criterion.equivalent_stress(X)
print("fresh-direction relative L2:", float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth)))
