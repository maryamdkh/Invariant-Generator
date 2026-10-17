"""Rerun Stages 2-3 for chosen equations from saved Stage 1 results (e.g. a
cluster run copied back), with a different config.

Usage: python experiments/probes/stage23_from_checkpoint.py CONFIG STAGE1_RUN_DIR OUT_DIR EQ[,EQ...]
"""
import json, sys
from importlib import import_module
from pathlib import Path
import_module("pysr")
from invariant_generator.config import load_config
from invariant_generator.feynman_benchmark import problem_config, run_pipeline

config_path, stage1_root, out_dir, equations = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]).resolve(), sys.argv[4].split(",")
base = load_config(config_path)
entries = {e["equation_id"]: e for e in json.loads(Path("data/feynman/manifest.json").read_text())}
for name in equations:
    config = problem_config(base, entries[name], results_dir=out_dir)
    src = next(stage1_root.rglob(f"{name}/stage1/adaptive_stage1_summary.json"))
    summary = json.loads(src.read_text())
    n = summary["selected_n"]
    local_ckpt = next(src.parent.rglob(f"n{n:02d}/**/checkpoint_best.pt"), None) or next(src.parent.rglob("checkpoint_best.pt"))
    # Point the copied summary at the local checkpoint, then let run_pipeline reuse Stage 1.
    best = min((r for r in summary["runs"] if r["n"] == n), key=lambda r: r["test_metrics"]["nrmse"])
    seed = Path(best["checkpoint"]).parent.name
    candidate = next(src.parent.rglob(f"n{n:02d}/{seed}/checkpoint_best.pt"), local_ckpt)
    summary["selected_checkpoint"] = str(candidate)
    target = out_dir / name / "stage1"
    target.mkdir(parents=True, exist_ok=True)
    (target / config.adaptive.summary_name).write_text(json.dumps(summary, indent=2))
    result = run_pipeline(config, reuse_stages=True)
    print(f"RESULT {name}: n={result.get('selected_n')} snapped={result.get('snapped')} groups={result.get('exponents')} "
          f"recovered={result.get('numerical_recovery')} err={result.get('fresh_sample_relative_l2')} pysr_s={result.get('pysr_seconds')}", flush=True)
