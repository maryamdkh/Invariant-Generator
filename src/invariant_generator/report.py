from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from invariant_generator.adaptive import adaptive_results_dir
from invariant_generator.config import Config, PROJECT_ROOT


def _source_lines(source: str) -> list[str]:
    return source.strip("\n").splitlines(keepends=True)


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(source),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def create_adaptive_analysis_notebook(
    config: Config,
    *,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Create a compact per-run analysis notebook in the adaptive results folder."""
    run_dir = Path(results_dir) if results_dir is not None else adaptive_results_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = Path(output_path) if output_path is not None else run_dir / "analysis.ipynb"
    config_literal = (
        "None"
        if config_path is None
        else repr(str(Path(config_path).expanduser().resolve()))
    )

    cells = [
        _markdown(
            """
# Adaptive Pipeline Run Analysis

This notebook summarizes one adaptive invariant-discovery run: Stage 1 adaptive encoder sweep, Stage 2 sparsification/refit, Stage 3 PySR, prediction diagnostics, and optional synthetic-benchmark recovery files when they exist.
"""
        ),
        _code(
            f"""
from pathlib import Path
import json
import os
import sys
from pprint import pprint

import numpy as np

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MPLCONFIGDIR = PROJECT_ROOT / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import matplotlib.pyplot as plt

from invariant_generator.adaptive import adaptive_sparsification_run_id
from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model, predict_numpy
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import resolve_device

CONFIG_PATH = {config_literal}
ADAPTIVE_RUN_DIR = Path({str(run_dir.resolve())!r})
STAGE1_DIR = ADAPTIVE_RUN_DIR / "stage1"
STAGE1_SUMMARY = STAGE1_DIR / "adaptive_stage1_summary.json"
STAGE2_DIR = ADAPTIVE_RUN_DIR / "stage2_sparse"
STAGE2_SUMMARY = STAGE2_DIR / "adaptive_stage2_sparsify.json"
SPARSE_CHECKPOINT = STAGE2_DIR / "checkpoint_best.pt"
SYMBOLIC_DIR = ADAPTIVE_RUN_DIR / "stage3_pysr"

print("project:", PROJECT_ROOT)
print("config:", CONFIG_PATH)
print("run dir:", ADAPTIVE_RUN_DIR)
print("stage1 summary exists:", STAGE1_SUMMARY.exists())
print("stage2 summary exists:", STAGE2_SUMMARY.exists())
print("symbolic dir exists:", SYMBOLIC_DIR.exists())

def load_json(path):
    path = Path(path)
    if not path.exists():
        print("missing:", path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
"""
        ),
        _markdown("## Stage 1: Adaptive n Sweep"),
        _code(
            """
stage1 = load_json(STAGE1_SUMMARY)
if stage1 is None:
    print("Stage 1 summary was not found.")
else:
    runs = stage1.get("runs", [])
    metric = stage1.get("metric", "rmse")
    selected_n = stage1.get("selected_n")
    print("metric:", metric)
    print("threshold:", stage1.get("threshold"))
    print("selected n:", selected_n)
    print("selected checkpoint:", stage1.get("selected_checkpoint"))

    if runs:
        n_values = np.array([row["n"] for row in runs], dtype=int)
        train_values = np.array([row["train_metrics"].get(metric, np.nan) for row in runs], dtype=float)
        test_values = np.array([row["test_metrics"].get(metric, np.nan) for row in runs], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(n_values, train_values, marker="o", label=f"train {metric}")
        ax.plot(n_values, test_values, marker="o", label=f"test {metric}")
        if selected_n is not None:
            ax.axvline(selected_n, color="black", linestyle="--", linewidth=1, label="selected n")
        ax.set_xlabel("encoder output dimension n")
        ax.set_ylabel(metric)
        ax.set_title("Adaptive encoder sweep")
        ax.legend()
        fig.tight_layout()
        plt.show()

        for row in runs:
            print(f"n={row['n']:02d} selected={row.get('selected')} train={row['train_metrics'].get(metric):.6g} test={row['test_metrics'].get(metric):.6g}")
"""
        ),
        _markdown("### Stage 1 Training Histories"),
        _code(
            """
if stage1 is not None:
    histories = {}
    for row in stage1.get("runs", []):
        history_path = row.get("history_path") or str(Path(row["experiment_dir"]) / "history.json")
        payload = load_json(history_path)
        history = [] if payload is None else payload.get("history", [])
        if history:
            histories[row["n"]] = history

    if not histories:
        print("No Stage 1 histories found.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for n, history in sorted(histories.items()):
            epochs = np.array([item["epoch"] for item in history], dtype=float)
            for key, ax in [("test_mse", axes[0]), ("test_rmse", axes[1])]:
                values = np.array([item.get(key, np.nan) for item in history], dtype=float)
                if np.isfinite(values).any():
                    ax.plot(epochs, values, label=f"n={n}")
                    ax.set_xlabel("epoch")
                    ax.set_ylabel(key)
                    ax.set_yscale("log")
        axes[0].set_title("Stage 1 test MSE")
        axes[1].set_title("Stage 1 test RMSE")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        plt.show()
"""
        ),
        _markdown("## Stage 2: Sparse Encoder"),
        _code(
            """
stage2 = load_json(STAGE2_SUMMARY)
if stage2 is None:
    print("Stage 2 summary was not found.")
else:
    print("method:", stage2.get("method"))
    print("threshold:", stage2.get("threshold"))
    print("selected max active terms per row:", stage2.get("selected_max_active_terms_per_row", stage2.get("max_active_terms_per_row")))
    print("selected candidate passes:", stage2.get("selected_candidate_passes"))
    print("checkpoint:", stage2.get("checkpoint"))
    print("train metrics:")
    pprint(stage2.get("train_metrics"))
    print("test metrics:")
    pprint(stage2.get("test_metrics"))

    candidate_results = stage2.get("candidate_results", [])
    if candidate_results:
        metric = stage2.get("selection_metric", "rmse")
        caps = [item.get("max_active_terms_per_row") for item in candidate_results]
        train = [item.get("train_metrics", {}).get(metric, np.nan) for item in candidate_results]
        test = [item.get("test_metrics", {}).get(metric, np.nan) for item in candidate_results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(caps, train, marker="o", label=f"train {metric}")
        ax.plot(caps, test, marker="o", label=f"test {metric}")
        ax.set_xlabel("max active terms per row")
        ax.set_ylabel(metric)
        ax.set_title("Stage 2 candidate comparison")
        ax.legend()
        fig.tight_layout()
        plt.show()
"""
        ),
        _markdown("### Stage 2 Loss Histories and Encoder Formulas"),
        _code(
            """
def stage2_history(path_key, inline_key):
    if stage2 is None:
        return []
    path = stage2.get(path_key)
    if path:
        payload = load_json(path)
        if payload is not None:
            return payload.get("history", [])
    return stage2.get(inline_key, [])

if stage2 is not None:
    sparse_history = stage2_history("sparse_history_path", "sparse_history")
    refit_history = stage2_history("refit_history_path", "refit_history")
    if sparse_history or refit_history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for history, label in [(sparse_history, "sparsity"), (refit_history, "masked refit")]:
            if not history:
                continue
            epochs = np.array([item["epoch"] for item in history], dtype=float)
            for key, ax in [("loss_total", axes[0]), ("loss_data", axes[1])]:
                values = np.array([item.get(key, np.nan) for item in history], dtype=float)
                ax.plot(epochs, values, label=label)
                ax.set_xlabel("epoch")
                ax.set_ylabel(key)
                ax.set_yscale("log")
        axes[0].set_title("Stage 2 total loss")
        axes[1].set_title("Stage 2 data loss")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        plt.show()
    else:
        print("No Stage 2 histories found.")

    formulas = stage2.get("formulas", {}).get("formulas", [])
    if formulas:
        print("Final encoded invariant formulas:")
        for item in formulas:
            print(item.get("raw_formula"))
            print("  active:", ", ".join(item.get("active_terms", [])) or "none")
"""
        ),
        _markdown("## Prediction Diagnostics from Sparse Checkpoint"),
        _code(
            """
if CONFIG_PATH is None:
    print("No config path is embedded in this notebook.")
elif not SPARSE_CHECKPOINT.exists():
    print("Sparse checkpoint not found:", SPARSE_CHECKPOINT)
else:
    import torch
    config = load_config(CONFIG_PATH)
    selected_n = None if stage1 is None else stage1.get("selected_n")
    if selected_n is not None:
        config.encoder.enabled = True
        config.encoder.output_dim = int(selected_n)
    config.train.run_id = adaptive_sparsification_run_id(config)
    device = resolve_device(config.train.device)
    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(SPARSE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    metrics = evaluate_model(model, data.X_test, data.y_test, device=device, batch_size=8192)
    print("test metrics:")
    pprint(metrics)

    y_pred = predict_numpy(model, data.X_test, device=device, batch_size=8192)
    y_true = data.y_test
    error = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_true, y_pred, s=16, alpha=0.7)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set_title("Test predictions vs targets")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[1].hist(error, bins=30)
    axes[1].set_title("Prediction error")
    axes[1].set_xlabel("prediction - target")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    plt.show()
"""
        ),
        _markdown("## Stage 3: PySR"),
        _code(
            """
metrics_path = SYMBOLIC_DIR / "metrics.json"
formulas_path = SYMBOLIC_DIR / "encoded_invariant_formulas.json"
best_equation_path = SYMBOLIC_DIR / "best_equation.txt"
equations_path = SYMBOLIC_DIR / "equations.csv"

symbolic_metrics = load_json(metrics_path)
symbolic_formulas = load_json(formulas_path)
if symbolic_metrics is not None:
    print("PySR metrics:")
    pprint(symbolic_metrics)
if best_equation_path.exists():
    best_equation = best_equation_path.read_text(encoding="utf-8").strip()
    print("\\nBest equation:")
    print(best_equation)
else:
    best_equation = None

if symbolic_formulas is not None:
    print("\\nEncoded variables in raw invariant coordinates:")
    for item in symbolic_formulas.get("formulas", []):
        print(item.get("raw_formula"))

if equations_path.exists():
    print("\\nEquation table:", equations_path)
"""
        ),
        _markdown("## Synthetic Benchmark Files, If Present"),
        _code(
            """
case_dir = ADAPTIVE_RUN_DIR
for name in [
    "benchmark_metadata.json",
    "generation_quality.json",
    "formula_dataset_quality.json",
    "recovery_metrics.json",
    "case_summary.json",
]:
    path = case_dir / name
    payload = load_json(path)
    if payload is None:
        continue
    print("\\n==", name, "==")
    if name == "recovery_metrics.json":
        print("classification:", payload.get("classification"))
        print("best equation:", payload.get("best_equation"))
        print("recovery metrics:")
        pprint(payload.get("recovery_metrics"))
    elif name == "formula_dataset_quality.json":
        print("formula:", payload.get("formula_expression"))
        print("complexity:")
        pprint(payload.get("formula_complexity"))
        print("homogeneity:")
        pprint(payload.get("formula_homogeneity"))
        print("dataset surface:")
        pprint(payload.get("dataset_surface"))
    else:
        keys = list(payload.keys())
        print("keys:", keys)
        for key in ["formula_name", "formula_expression", "difficulty", "generation_quality"]:
            if key in payload:
                print(key + ":", payload[key])
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path
