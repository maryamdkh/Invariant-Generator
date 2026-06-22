from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.model import InvariantYieldModel
from invariant_generator.train import TrainResult, train_from_config
from invariant_generator.utils import resolve_device, save_json


@dataclass(slots=True)
class AdaptiveRunSummary:
    n: int
    run_id: str
    experiment_dir: Path
    checkpoint: Path
    history_path: Path
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    selected: bool


@dataclass(slots=True)
class AdaptiveSweepResult:
    summary_path: Path
    plot_path: Path | None
    selected_n: int | None
    selected_checkpoint: Path | None
    runs: list[AdaptiveRunSummary]


def adaptive_metric_threshold(config: Config) -> tuple[str, float]:
    metric = config.adaptive.metric.lower()
    if metric == "mse":
        return metric, float(config.adaptive.mse_threshold)
    if metric == "rmse":
        return metric, float(config.adaptive.rmse_threshold)
    raise ValueError("adaptive.metric must be 'mse' or 'rmse'.")


def adaptive_run_passes(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    *,
    metric: str,
    threshold: float,
    max_generalization_gap: float | None = None,
) -> bool:
    if metric not in train_metrics or metric not in test_metrics:
        raise KeyError(f"Metric {metric!r} is missing from train/test metrics.")
    train_value = float(train_metrics[metric])
    test_value = float(test_metrics[metric])
    if train_value > threshold or test_value > threshold:
        return False
    if max_generalization_gap is not None and abs(test_value - train_value) > max_generalization_gap:
        return False
    return True


def adaptive_n_bounds(config: Config) -> tuple[int, int]:
    input_dim = len(config.invariants.selected)
    n_min = max(1, int(config.adaptive.n_min))
    n_max = int(config.adaptive.n_max) or input_dim
    n_max = min(n_max, input_dim)
    if n_min > n_max:
        raise ValueError(f"adaptive.n_min={n_min} must be <= adaptive.n_max={n_max}.")
    return n_min, n_max


def adaptive_n_values(config: Config) -> list[int]:
    n_min, n_max = adaptive_n_bounds(config)
    return list(range(n_min, n_max + 1))


def adaptive_results_dir(config: Config) -> Path:
    return config.train.results_dir / config.adaptive.results_subdir


def adaptive_stage1_dir(config: Config) -> Path:
    return adaptive_results_dir(config) / config.adaptive.stage1_subdir


def adaptive_sparsification_run_id(config: Config) -> str:
    return str(Path(config.adaptive.results_subdir) / config.sparsification.run_id)


def config_for_adaptive_n(config: Config, n: int) -> Config:
    run_config = deepcopy(config)
    input_dim = len(run_config.invariants.selected)
    run_config.encoder.enabled = True
    run_config.encoder.output_dim = int(n)
    if n < input_dim and run_config.encoder.init == "identity":
        run_config.encoder.init = "random"
    run_config.train.run_id = str(
        Path(run_config.adaptive.results_subdir)
        / run_config.adaptive.stage1_subdir
        / f"{run_config.adaptive.run_id_prefix}{n:02d}"
    )
    return run_config


@torch.no_grad()
def evaluate_checkpoint_on_train_and_test(
    config: Config,
    checkpoint_path: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    device = resolve_device(config.train.device)
    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate_model(
        model,
        data.X_train,
        data.y_train,
        device=device,
        batch_size=8192,
    )
    test_metrics = evaluate_model(
        model,
        data.X_test,
        data.y_test,
        device=device,
        batch_size=8192,
    )
    return train_metrics, test_metrics


def _save_stage1_plot(summary_dir: Path, runs: list[AdaptiveRunSummary], metric: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not runs:
        return None

    ordered_runs = sorted(runs, key=lambda run: run.n)
    n_values = [run.n for run in ordered_runs]
    train_values = [run.train_metrics[metric] for run in ordered_runs]
    test_values = [run.test_metrics[metric] for run in ordered_runs]
    plot_path = summary_dir / f"adaptive_stage1_{metric}_vs_n.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_values, train_values, marker="o", label=f"train {metric}")
    ax.plot(n_values, test_values, marker="o", label=f"test {metric}")
    for run in ordered_runs:
        if run.selected:
            ax.axvline(run.n, color="black", linestyle="--", linewidth=1)
            break
    ax.set_xlabel("encoder output dimension n")
    ax.set_ylabel(metric)
    ax.set_title("Adaptive encoder sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def run_adaptive_sweep(config: Config) -> AdaptiveSweepResult:
    metric, threshold = adaptive_metric_threshold(config)
    summary_dir = adaptive_stage1_dir(config)
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs: list[AdaptiveRunSummary] = []
    selected_n: int | None = None
    selected_checkpoint: Path | None = None
    passing_streak: list[AdaptiveRunSummary] = []
    confirmation_count = max(0, int(config.adaptive.patience)) + 1

    for n in adaptive_n_values(config):
        run_config = config_for_adaptive_n(config, n)
        train_result: TrainResult = train_from_config(run_config)
        train_metrics, test_metrics = evaluate_checkpoint_on_train_and_test(
            run_config,
            train_result.best_checkpoint,
        )
        passes = adaptive_run_passes(
            train_metrics,
            test_metrics,
            metric=metric,
            threshold=threshold,
            max_generalization_gap=config.adaptive.max_generalization_gap,
        )
        run_summary = AdaptiveRunSummary(
            n=n,
            run_id=run_config.train.run_id,
            experiment_dir=train_result.experiment_dir,
            checkpoint=train_result.best_checkpoint,
            history_path=train_result.history_path,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            selected=False,
        )
        runs.append(run_summary)
        if passes:
            passing_streak.append(run_summary)
        else:
            passing_streak.clear()

        if len(passing_streak) >= confirmation_count and selected_n is None:
            chosen = passing_streak[0]
            chosen.selected = True
            selected_n = chosen.n
            selected_checkpoint = chosen.checkpoint
            if config.adaptive.stop_on_first_success:
                break

    plot_path = _save_stage1_plot(summary_dir, runs, metric)
    summary_path = save_json(
        summary_dir / config.adaptive.summary_name,
        {
            "metric": metric,
            "threshold": threshold,
            "max_generalization_gap": config.adaptive.max_generalization_gap,
            "patience": config.adaptive.patience,
            "selected_n": selected_n,
            "selected_checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
            "runs": [
                {
                    "n": run.n,
                    "run_id": run.run_id,
                    "experiment_dir": str(run.experiment_dir),
                    "checkpoint": str(run.checkpoint),
                    "history_path": str(run.history_path),
                    "train_metrics": run.train_metrics,
                    "test_metrics": run.test_metrics,
                    "selected": run.selected,
                }
                for run in runs
            ],
            "plot": None if plot_path is None else str(plot_path),
        },
    )
    return AdaptiveSweepResult(
        summary_path=summary_path,
        plot_path=plot_path,
        selected_n=selected_n,
        selected_checkpoint=selected_checkpoint,
        runs=runs,
    )
