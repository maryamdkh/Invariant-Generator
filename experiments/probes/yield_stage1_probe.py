"""Stage 1 on a yield benchmark: is n = 2 selected, and do the learned rows
span the generalized J2^0 and J3^0 of the generating criterion?

Usage: python experiments/probes/yield_stage1_probe.py configs/yield_cb2001_2090T3.toml
"""
import json, sys, time
import numpy as np, torch
from invariant_generator.adaptive import config_for_adaptive_n, run_adaptive_sweep
from invariant_generator.config import load_config
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.input_features import OrthotropicPolynomialPool, input_feature_names
from invariant_generator.sparsify import load_model_for_checkpoint
from invariant_generator.yield_criteria import CRITERIA

config = load_config(sys.argv[1])
criterion = CRITERIA[config.data.dataset_name]
t0 = time.time()
sweep = run_adaptive_sweep(config)
summary = json.loads(sweep.summary_path.read_text())
print("reference nrmse", summary.get("reference", {}).get("test_metrics", {}).get("nrmse"),
      "spectrum", np.round(summary.get("reference", {}).get("gradient_spectrum", [])[:5], 4).tolist())
print("runs", [(r["n"], round(r["test_metrics"]["nrmse"], 5)) for r in summary["runs"]],
      "threshold", summary["threshold"], "selected", sweep.selected_n, f"{time.time() - t0:.0f}s")

# True J2^0, J3^0 as coefficient vectors over the feature pool.
pool = OrthotropicPolynomialPool()
X = np.random.default_rng(0).normal(size=(4000, 6))
F = pool(torch.as_tensor(X)).numpy()
truth = np.vstack([np.linalg.lstsq(F, target, rcond=None)[0] for target in criterion.invariants(X)])
if sweep.selected_n is not None:
    run_config = config_for_adaptive_n(config, sweep.selected_n)
    model = load_model_for_checkpoint(run_config, sweep.selected_checkpoint)
    B = np.asarray(encoder_formula_report(model, input_feature_names(run_config))["raw_coefficients"])
    learned = np.linalg.qr(B.T)[0]
    true_basis = np.linalg.qr(truth.T)[0]
    cosines = np.linalg.svd(true_basis.T @ learned, compute_uv=False)
    print("principal-angle cosines between learned rows and span{J2, J3}:", np.round(cosines, 6))
    for name, row in zip(("J2", "J3"), true_basis.T):
        print(f"  {name} captured by learned span: {np.linalg.norm(learned.T @ row):.6f}")
