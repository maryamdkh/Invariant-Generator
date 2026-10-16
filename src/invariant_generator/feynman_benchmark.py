"""Run the adaptive pipeline and a PySR-alone baseline on Feynman problems.

Both arms use the same data split, PySR settings, time limit and recovery
check. The pipeline arm runs Stage 1 (smallest number of encoded variables,
searched below the input dimension), Stage 2 and Stage 3 on the encoded
variables. When Stage 1 finds no reduction, the pipeline has nothing to add
and the row reports the baseline result for that arm.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from invariant_generator.adaptive import (
    adaptive_sparsification_run_id,
    adaptive_stage1_dir,
    config_for_adaptive_n,
    run_adaptive_sweep,
)
from invariant_generator.adaptive_symbolic import (
    load_known_formula,
    original_input_evaluation,
    train_encoded_symbolic_from_config,
)
from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint
from invariant_generator.symbolic import (
    _best_equation_text,
    _import_pysr_regressor,
    _save_equations_csv,
    pysr_regressor_options,
)
from invariant_generator.utils import save_json


def problem_config(base: Config, entry: dict[str, Any], *, results_dir: Path) -> Config:
    """Per-equation config: dataset, inputs, n search range and stop condition."""
    config = deepcopy(base)
    config.data.dataset_name = entry["equation_id"]
    config.data.feature_names = list(entry["variables"])
    config.train.results_dir = results_dir
    config.train.split_dir = results_dir / "splits"
    config.adaptive.results_subdir = entry["equation_id"]
    config.adaptive.n_min = 1
    # A reduction needs fewer encoded variables than inputs.
    config.adaptive.n_max = max(1, len(entry["variables"]) - 1)
    return config


def _with_stop_condition(config: Config, y: np.ndarray) -> Config:
    """Stop PySR once its MSE is below 1e-12 of the target variance (nrmse 1e-6)."""
    config = deepcopy(config)
    threshold = max(float(np.var(y)) * 1e-12, 1e-300)
    config.symbolic.early_stop_condition = f"stop_if(loss, complexity) = loss < {threshold:.6e}"
    return config


def run_pysr_baseline(config: Config, output_dir: Path) -> dict[str, Any]:
    """PySR on the raw inputs with the same settings as the pipeline's Stage 3."""
    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(config)
    config = _with_stop_condition(config, data.y_train)
    names = list(config.data.feature_names)
    # Neutral names avoid clashes between physics symbols and Julia/SymPy names.
    pysr_names = [f"x{i}" for i in range(len(names))]
    model = PySRRegressor(**pysr_regressor_options(config, output_directory=output_dir))
    started = time.perf_counter()
    model.fit(data.X_train, data.y_train, variable_names=pysr_names)
    elapsed = time.perf_counter() - started
    _save_equations_csv(model, output_dir / "equations.csv")
    known = load_known_formula(config, fallback_X=data.X_test)
    import sympy as sp

    symbols = (
        known[1]["symbols"] if known is not None else {name: sp.Symbol(name) for name in names}
    )
    expression = sp.sympify(model.sympy()).subs(
        {sp.Symbol(pysr_names[i]): symbols[name] for i, name in enumerate(names)},
        simultaneous=True,
    )
    evaluation = original_input_evaluation(
        str(expression), known,
        test_prediction=np.asarray(model.predict(data.X_test), dtype=np.float64),
        y_test=data.y_test,
    )
    validation_error = np.asarray(model.predict(data.X_validation), dtype=np.float64) - data.y_validation
    result = {
        "best_equation_raw_names": _best_equation_text(model),
        "pysr_seconds": elapsed,
        "early_stop_condition": config.symbolic.early_stop_condition,
        "validation_rmse": float(np.sqrt(np.mean(validation_error**2))),
        **evaluation,
    }
    save_json(output_dir / "baseline_evaluation.json", result)
    return result


def run_pipeline(
    config: Config,
    *,
    symbolic_hours: float | None = None,
    stage3_subdir: str | None = None,
    reuse_stages: bool = False,
) -> dict[str, Any]:
    """Stage 1 -> Stage 2 -> Stage 3 for one equation.

    With reuse_stages, saved Stage 1 and Stage 2 results are loaded instead of
    retrained (e.g. for a second Stage 3 with a different PySR budget).
    """
    result: dict[str, Any] = {}
    stage1_summary = adaptive_stage1_dir(config) / config.adaptive.summary_name
    if reuse_stages and stage1_summary.exists():
        saved = json.loads(stage1_summary.read_text(encoding="utf-8"))
        selected_n = saved["selected_n"]
        selected_checkpoint = saved["selected_checkpoint"]
        result["stage1_reused"] = True
    else:
        started = time.perf_counter()
        sweep = run_adaptive_sweep(config)
        result["stage1_seconds"] = time.perf_counter() - started
        result["stage1_runs"] = [
            {"n": run.n, "val_metric": run.test_metrics.get(config.adaptive.metric)}
            for run in sweep.runs
        ]
        selected_n, selected_checkpoint = sweep.selected_n, sweep.selected_checkpoint
    result["selected_n"] = selected_n
    if selected_n is None:
        result["reduction"] = False
        return result
    result["reduction"] = True

    stage_config = config_for_adaptive_n(config, selected_n)
    stage2_summary = (
        config.train.results_dir / adaptive_sparsification_run_id(stage_config)
        / config.sparsification.summary_name
    )
    if reuse_stages and stage2_summary.exists():
        summary = json.loads(stage2_summary.read_text(encoding="utf-8"))
        sparse_checkpoint = Path(summary["checkpoint"])
    else:
        started = time.perf_counter()
        sparse = sparsify_encoder_from_checkpoint(stage_config, checkpoint_path=selected_checkpoint)
        result["stage2_seconds"] = time.perf_counter() - started
        summary = json.loads(sparse.summary_path.read_text(encoding="utf-8"))
        sparse_checkpoint = sparse.checkpoint_path
    result["snapped"] = summary.get("snapped")
    result["exponents"] = summary.get("exact_raw_coefficients")
    result["stage2_val_metric"] = summary["test_metrics"].get(config.adaptive.metric)

    symbolic_config = config_for_adaptive_n(config, selected_n)
    symbolic_config.train.run_id = adaptive_sparsification_run_id(symbolic_config)
    if symbolic_hours is not None:
        symbolic_config.symbolic.timeout_hours = symbolic_hours
    if stage3_subdir is not None:
        symbolic_config.symbolic.output_subdir = stage3_subdir
    data = prepare_training_data(symbolic_config)
    symbolic_config = _with_stop_condition(symbolic_config, data.y_train)
    started = time.perf_counter()
    stage3 = train_encoded_symbolic_from_config(symbolic_config, checkpoint_path=sparse_checkpoint)
    result["pysr_seconds"] = time.perf_counter() - started
    result["best_equation_in_groups"] = stage3.best_equation
    evaluation = json.loads(
        (stage3.output_dir / "original_input_evaluation.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(stage3.metrics_path.read_text(encoding="utf-8"))
    result["validation_rmse"] = None if metrics.get("validation") is None else metrics["validation"]["rmse"]
    result.update(evaluation)
    return result


def run_portfolio(config: Config, *, results_dir: Path, raw_share: float) -> dict[str, Any]:
    """Split the PySR budget between the groups and the raw inputs and keep
    the candidate with the lower validation RMSE (the known formula is not
    used). Without a reduction the whole budget goes to the raw inputs."""
    total_hours = float(config.symbolic.timeout_hours)
    groups = run_pipeline(
        config,
        symbolic_hours=total_hours * (1.0 - raw_share),
        stage3_subdir="stage3_portfolio",
        reuse_stages=True,
    )
    raw_config = deepcopy(config)
    name = config.data.dataset_name
    if not groups.get("reduction"):
        raw = run_pysr_baseline(raw_config, results_dir / name / "portfolio_raw_full")
        return {**raw, "portfolio_choice": "raw", "reduction": False}
    raw_config.symbolic.timeout_hours = total_hours * raw_share
    raw = run_pysr_baseline(raw_config, results_dir / name / "portfolio_raw")
    group_score = groups.get("validation_rmse")
    raw_score = raw.get("validation_rmse")
    choose_groups = raw_score is None or (group_score is not None and group_score <= raw_score)
    chosen = groups if choose_groups else raw
    return {
        **chosen,
        "portfolio_choice": "groups" if choose_groups else "raw",
        "portfolio_groups": {k: groups.get(k) for k in (
            "numerical_recovery", "fresh_sample_relative_l2", "validation_rmse", "pysr_seconds",
            "expanded_equation",
        )},
        "portfolio_raw": {k: raw.get(k) for k in (
            "numerical_recovery", "fresh_sample_relative_l2", "validation_rmse", "pysr_seconds",
            "expanded_equation",
        )},
        "reduction": True,
        "selected_n": groups.get("selected_n"),
        "exponents": groups.get("exponents"),
    }


def run_benchmark(
    base: Config,
    entries: list[dict[str, Any]],
    *,
    results_dir: Path,
    arms: tuple[str, ...] = ("pipeline", "baseline"),
    resume: bool = True,
    portfolio_raw_share: float = 0.5,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"
    rows: dict[str, dict[str, Any]] = {}
    if resume and summary_path.exists():
        rows = {row["equation_id"]: row for row in json.loads(summary_path.read_text(encoding="utf-8"))}
    for entry in entries:
        name = entry["equation_id"]
        row = rows.get(name, {
            "equation_id": name,
            "formula": entry["formula"],
            "n_inputs": len(entry["variables"]),
            "log_reducibility_analysis": entry.get("log_reducibility"),
        })
        config = problem_config(base, entry, results_dir=results_dir)
        for arm in arms:
            if arm in row and "error" not in row[arm]:
                continue
            print(f"[BENCH] {name}: {arm}", flush=True)
            try:
                if arm == "pipeline":
                    row[arm] = run_pipeline(config)
                elif arm == "baseline":
                    row[arm] = run_pysr_baseline(config, results_dir / name / "baseline_pysr")
                elif arm == "portfolio":
                    row[arm] = run_portfolio(
                        config, results_dir=results_dir, raw_share=portfolio_raw_share,
                    )
                else:
                    raise ValueError(f"Unknown arm {arm!r}")
            except Exception as exc:
                row[arm] = {"error": repr(exc), "traceback": traceback.format_exc()}
            rows[name] = row
            save_json(summary_path, list(rows.values()))
    return summary_path
