from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.model import InvariantYieldModel
from invariant_generator.input_features import input_feature_names
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
    # One record per restart when adaptive.restarts > 1; the run fields above
    # describe the restart with the best selection-partition metric.
    restarts: list[dict[str, object]] = field(default_factory=list)


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
    if metric == "nrmse":
        return metric, float(config.adaptive.nrmse_threshold)
    raise ValueError("adaptive.metric must be 'mse', 'rmse' or 'nrmse'.")


def with_selection_metric(
    metrics: dict[str, float],
    y: np.ndarray,
    *,
    metric: str,
) -> dict[str, float]:
    """Add nrmse = rmse / std(y) when it is the selection metric."""
    if metric != "nrmse":
        return metrics
    scale = float(np.std(np.asarray(y, dtype=np.float64)))
    if scale <= 0.0:
        raise ValueError("adaptive.metric='nrmse' requires a non-constant target; use 'rmse'.")
    return {**metrics, "nrmse": float(metrics["rmse"]) / scale}


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
    input_dim = len(input_feature_names(config))
    default_max = (
        len(config.data.feature_names)
        if config.input_transform.kind == "pairwise"
        else input_dim
    )
    n_min = max(1, int(config.adaptive.n_min))
    n_max = int(config.adaptive.n_max) or default_max
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
    input_dim = len(input_feature_names(run_config))
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
        data.X_selection,
        data.y_selection,
        device=device,
        batch_size=8192,
    )
    metric = config.adaptive.metric.lower()
    return (
        with_selection_metric(train_metrics, data.y_train, metric=metric),
        with_selection_metric(test_metrics, data.y_selection, metric=metric),
    )


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


@torch.no_grad()
def _normalized_features(model: InvariantYieldModel, X: np.ndarray, device: torch.device) -> torch.Tensor:
    model.eval()
    return model.normalized_invariant_features(torch.as_tensor(X, dtype=torch.float32, device=device))


def reference_gradient_spectrum(
    model: InvariantYieldModel,
    X: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Singular values (divided by the largest) and right singular vectors of
    the row-normalized gradients of a bottleneck-free model with respect to
    its normalized input features.

    If the target depends on the features only through n linear combinations,
    every gradient lies in their n-dimensional span, so the spectrum shows n
    large values and the top vectors span the encoder rows.
    """
    if model.encoder is not None:
        raise ValueError("The reference model must not have an encoder.")
    features = _normalized_features(model, X, device).clone().requires_grad_(True)
    with torch.enable_grad():
        prediction = model.regressor(features)
        if model.target_scaler is not None:
            prediction = model.target_scaler(prediction)
        (gradients,) = torch.autograd.grad(prediction.sum(), features)
    G = gradients.detach().cpu().numpy().astype(np.float64)
    G /= np.linalg.norm(G, axis=1, keepdims=True).clip(1e-300)
    _, singular_values, Vt = np.linalg.svd(G, full_matrices=False)
    return singular_values / singular_values[0], Vt


def _train_reference(config: Config, metric: str) -> dict[str, object]:
    """Full-width model without the bottleneck: the attainable fit and the
    gradient spectrum for this dataset and MLP."""
    reference_config = deepcopy(config)
    reference_config.encoder.enabled = False
    reference_config.train.run_id = str(
        Path(config.adaptive.results_subdir) / config.adaptive.stage1_subdir / "reference"
    )
    result = train_from_config(reference_config)
    train_metrics, test_metrics = evaluate_checkpoint_on_train_and_test(
        reference_config, result.best_checkpoint,
    )
    device = resolve_device(reference_config.train.device)
    model = InvariantYieldModel.from_config(reference_config).to(device)
    checkpoint = torch.load(result.best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    data = prepare_training_data(reference_config)
    spectrum, Vt = reference_gradient_spectrum(model, data.X_train, device=device)
    return {
        "checkpoint": str(result.best_checkpoint),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gradient_spectrum": spectrum.tolist(),
        "right_singular_vectors": Vt.tolist(),
        "metric": metric,
    }


def run_adaptive_sweep(config: Config) -> AdaptiveSweepResult:
    metric, threshold = adaptive_metric_threshold(config)
    summary_dir = adaptive_stage1_dir(config)
    summary_dir.mkdir(parents=True, exist_ok=True)

    acceptance = config.adaptive.acceptance.lower()
    if acceptance not in {"absolute", "relative_to_reference"}:
        raise ValueError("adaptive.acceptance must be 'absolute' or 'relative_to_reference'.")
    reference: dict[str, object] | None = None
    if acceptance == "relative_to_reference":
        reference = _train_reference(config, metric)
        attainable = max(
            float(reference["train_metrics"][metric]),  # type: ignore[index]
            float(reference["test_metrics"][metric]),  # type: ignore[index]
        )
        threshold = max(float(config.adaptive.relative_factor) * attainable, float(config.adaptive.relative_floor))

    runs: list[AdaptiveRunSummary] = []
    selected_n: int | None = None
    selected_checkpoint: Path | None = None
    passing_streak: list[AdaptiveRunSummary] = []
    confirmation_count = max(0, int(config.adaptive.patience)) + 1

    restarts = max(1, int(config.adaptive.restarts))
    for n in adaptive_n_values(config):
        run_config = config_for_adaptive_n(config, n)
        restart_records: list[dict[str, object]] = []
        best: tuple[float, Config, TrainResult, dict[str, float], dict[str, float]] | None = None
        for restart in range(restarts):
            restart_config = run_config
            if restarts > 1:
                # Each restart changes only the initialization/shuffling seed.
                restart_config = deepcopy(run_config)
                restart_config.train.seed = int(run_config.train.seed) + restart
                restart_config.train.run_id = str(
                    Path(run_config.train.run_id) / f"seed{restart_config.train.seed:03d}"
                )
            if reference is not None and config.adaptive.reference_init and restart == 0:
                # Start one restart from the top-n gradient directions.
                restart_config = deepcopy(restart_config)
                restart_config.encoder.init_weights = [
                    list(row) for row in reference["right_singular_vectors"][:n]  # type: ignore[index]
                ]
            result = train_from_config(restart_config)
            restart_train, restart_test = evaluate_checkpoint_on_train_and_test(
                restart_config,
                result.best_checkpoint,
            )
            score = float(restart_test[metric])
            if restarts > 1:
                restart_records.append({
                    "seed": restart_config.train.seed,
                    "checkpoint": str(result.best_checkpoint),
                    "train_metrics": restart_train,
                    "test_metrics": restart_test,
                })
            if best is None or score < best[0]:
                best = (score, restart_config, result, restart_train, restart_test)
        assert best is not None
        _, run_config, train_result, train_metrics, test_metrics = best
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
            restarts=restart_records,
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
            **({} if reference is None else {
                "acceptance": acceptance,
                "reference": {k: v for k, v in reference.items() if k != "right_singular_vectors"},
            }),
            "max_generalization_gap": config.adaptive.max_generalization_gap,
            "patience": config.adaptive.patience,
            "selected_n": selected_n,
            "selected_checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
            "selection_partition": "validation" if config.data.kind == "supervised" else "test",
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
                    **({"restarts": run.restarts} if run.restarts else {}),
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
