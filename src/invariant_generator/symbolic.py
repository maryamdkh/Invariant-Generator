from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.diagnostics import (
    constraint_diagnostics,
    encoder_score_diagnostics,
    invariant_feature_statistics,
)
from invariant_generator.evaluation import regression_metrics
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import resolve_device, save_config_snapshot, save_json


@dataclass(slots=True)
class InvariantSelection:
    names: list[str]
    indices: list[int]
    scores: list[float]
    raw_scores: list[float] = field(default_factory=list)
    scaled_scores: list[float] = field(default_factory=list)
    feature_selection: str = "encoder_norm"


@dataclass(slots=True)
class SymbolicResult:
    output_dir: Path
    config_snapshot_path: Path
    selected_invariants_path: Path
    equations_path: Path
    best_equation_path: Path
    metrics_path: Path
    selected_invariants: list[str]
    best_equation: str


def _import_pysr_regressor() -> type[Any]:
    try:
        module = import_module("pysr")
    except ImportError as exc:
        raise ImportError(
            "PySR is required for symbolic regression. Install the optional "
            "dependency with: uv sync --extra symbolic"
        ) from exc
    return module.PySRRegressor


def select_top_invariants_from_encoder(
    encoder_matrix: torch.Tensor,
    invariant_names: list[str],
    *,
    top_k: int,
    feature_selection: str = "encoder_norm",
    feature_stds: np.ndarray | list[float] | None = None,
) -> InvariantSelection:
    """Rank original invariant columns by raw or scale-aware encoder score."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if encoder_matrix.ndim != 2:
        raise ValueError(f"encoder_matrix must be 2D, got shape {encoder_matrix.shape}.")
    if encoder_matrix.shape[1] != len(invariant_names):
        raise ValueError(
            "encoder_matrix columns must match invariant_names. "
            f"Got {encoder_matrix.shape[1]} columns and {len(invariant_names)} names."
        )

    raw_scores_tensor = torch.linalg.vector_norm(
        encoder_matrix.detach().float().cpu(),
        ord=2,
        dim=0,
    )
    raw_scores_array = raw_scores_tensor.numpy().astype(np.float64)
    if feature_stds is None:
        std_array = np.ones_like(raw_scores_array)
    else:
        std_array = np.asarray(feature_stds, dtype=np.float64)
        if std_array.shape != raw_scores_array.shape:
            raise ValueError(
                "feature_stds must match invariant_names. "
                f"Got {std_array.shape} and {raw_scores_array.shape}."
            )
    scaled_scores_array = raw_scores_array * std_array

    feature_selection = feature_selection.lower()
    if feature_selection == "encoder_norm":
        ranking_scores = raw_scores_array
    elif feature_selection == "scaled_encoder_norm":
        ranking_scores = scaled_scores_array
    else:
        raise ValueError(
            "feature_selection must be 'encoder_norm' or 'scaled_encoder_norm' "
            "when ranking from the encoder."
        )

    ranked = sorted(
        enumerate(float(score) for score in ranking_scores),
        key=lambda item: (-item[1], item[0]),
    )
    selected = ranked[: min(top_k, len(ranked))]

    return InvariantSelection(
        names=[invariant_names[idx] for idx, _ in selected],
        indices=[idx for idx, _ in selected],
        scores=[score for _, score in selected],
        raw_scores=[float(raw_scores_array[idx]) for idx, _ in selected],
        scaled_scores=[float(scaled_scores_array[idx]) for idx, _ in selected],
        feature_selection=feature_selection,
    )


def select_manual_invariants(
    invariant_names: list[str],
    selected_names: list[str],
    *,
    raw_scores: list[float] | None = None,
    scaled_scores: list[float] | None = None,
) -> InvariantSelection:
    """Select user-requested invariant columns in the requested order."""
    if not selected_names:
        raise ValueError("symbolic.selected_invariants must not be empty for manual selection.")

    index_by_name = {name: idx for idx, name in enumerate(invariant_names)}
    unknown = [name for name in selected_names if name not in index_by_name]
    if unknown:
        raise ValueError(f"Unknown symbolic.selected_invariants: {unknown}")

    indices = [index_by_name[name] for name in selected_names]
    raw = raw_scores if raw_scores is not None else [0.0] * len(invariant_names)
    scaled = scaled_scores if scaled_scores is not None else [0.0] * len(invariant_names)
    return InvariantSelection(
        names=list(selected_names),
        indices=indices,
        scores=[float(scaled[idx]) for idx in indices],
        raw_scores=[float(raw[idx]) for idx in indices],
        scaled_scores=[float(scaled[idx]) for idx in indices],
        feature_selection="manual",
    )


@torch.no_grad()
def compute_selected_invariant_features(
    model: InvariantYieldModel,
    X: np.ndarray,
    selected_indices: list[int],
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Compute selected pre-encoder invariant columns for stress samples."""
    if not selected_indices:
        raise ValueError("selected_indices must not be empty.")

    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    index_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=device)
    outputs: list[np.ndarray] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size].to(device)
        features = model.invariant_pool(batch).index_select(dim=1, index=index_tensor)
        outputs.append(features.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def _best_equation_text(pysr_model: Any) -> str:
    if hasattr(pysr_model, "get_best"):
        best = pysr_model.get_best()
        if isinstance(best, dict):
            return str(best.get("equation", best))
        if hasattr(best, "get"):
            return str(best.get("equation", best))
        return str(best)
    if hasattr(pysr_model, "sympy"):
        return str(pysr_model.sympy())
    return str(pysr_model)


def _save_equations_csv(pysr_model: Any, equations_path: Path) -> None:
    equations = getattr(pysr_model, "equations_", None)
    if equations is None:
        equations_path.write_text("", encoding="utf-8")
        return
    if hasattr(equations, "to_csv"):
        equations.to_csv(equations_path, index=False)
        return
    equations_path.write_text(str(equations), encoding="utf-8")


def pysr_sample_weights(target: np.ndarray, *, weight_mode: str, eps: float = 1e-12) -> np.ndarray | None:
    target = np.asarray(target, dtype=np.float64)
    weight_mode = weight_mode.lower()
    if weight_mode in {"none", "off", ""}:
        return None
    if weight_mode == "inverse_target_squared":
        return 1.0 / np.maximum(np.abs(target), eps) ** 2
    raise ValueError(
        "symbolic.weight_mode must be 'inverse_target_squared' or 'none', "
        f"got {weight_mode!r}."
    )


def _pysr_fit_kwargs(config: Config, target: np.ndarray) -> dict[str, Any]:
    if "weight" not in config.symbolic.elementwise_loss:
        return {}
    weights = pysr_sample_weights(target, weight_mode=config.symbolic.weight_mode)
    if weights is None:
        return {}
    return {"weights": weights}


def transform_symbolic_target(target: np.ndarray, *, target_transform: str) -> np.ndarray:
    """Apply the user-selected target transform for symbolic regression."""
    target = np.asarray(target, dtype=np.float64)
    target_transform = target_transform.lower()
    if target_transform in {"identity", "none", ""}:
        return target
    if target_transform == "square":
        return target**2
    raise ValueError(
        "symbolic.target_transform must be 'identity' or 'square', "
        f"got {target_transform!r}."
    )


def train_symbolic_from_config(
    config: Config,
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> SymbolicResult:
    """Run post-training PySR on top-ranked invariant features."""
    feature_selection = config.symbolic.feature_selection.lower()
    if feature_selection not in {"encoder_norm", "scaled_encoder_norm", "manual"}:
        raise ValueError(
            "symbolic.feature_selection must be 'encoder_norm', "
            "'scaled_encoder_norm', or 'manual'."
        )
    if feature_selection != "manual" and not config.encoder.enabled:
        raise ValueError(
            "PySR invariant ranking requires encoder.enabled=true because "
            "selected invariants are ranked from the learned S matrix."
        )

    PySRRegressor = _import_pysr_regressor()
    device = resolve_device(config.train.device)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.experiment_dir / "checkpoint_best.pt"
    )
    output_dir = config.experiment_dir / config.symbolic.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = save_config_snapshot(config, output_dir)

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    invariant_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
    )
    encoder_input_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        normalized=True,
    )
    feature_stds = np.asarray(encoder_input_stats["std"], dtype=np.float64)
    S = model.encoder_matrix()
    encoder_scores = encoder_score_diagnostics(
        model,
        config.invariants.selected,
        feature_std=feature_stds,
    )
    if S is None and feature_selection != "manual":
        raise ValueError("Loaded model does not have an encoder matrix S.")

    if feature_selection == "manual":
        selection = select_manual_invariants(
            config.invariants.selected,
            config.symbolic.selected_invariants,
            raw_scores=encoder_scores.get("raw_l2") if encoder_scores else None,
            scaled_scores=encoder_scores.get("scaled_l2") if encoder_scores else None,
        )
    else:
        assert S is not None
        selection = select_top_invariants_from_encoder(
            S,
            config.invariants.selected,
            top_k=config.symbolic.top_k,
            feature_selection=feature_selection,
            feature_stds=feature_stds,
        )
    print(
        "[INFO] PySR invariant source: "
        f"{selection.feature_selection}"
    )
    print(
        "[INFO] PySR selected invariants: "
        + ", ".join(
            f"{name}(score={score:.6g})"
            for name, score in zip(selection.names, selection.scores)
        )
    )
    X_train = compute_selected_invariant_features(
        model,
        data.X_train,
        selection.indices,
        device=device,
    )
    X_test = compute_selected_invariant_features(
        model,
        data.X_test,
        selection.indices,
        device=device,
    )
    y_train = transform_symbolic_target(
        data.y_train,
        target_transform=config.symbolic.target_transform,
    )
    y_test = transform_symbolic_target(
        data.y_test,
        target_transform=config.symbolic.target_transform,
    )

    pysr_model = PySRRegressor(
        parallelism=config.symbolic.parallelism,
        procs = config.symbolic.numprocs,
        batching = config.symbolic.batching,
        batch_size = config.symbolic.batch_size,
        niterations=config.symbolic.niterations,
        timeout_in_seconds=int(config.symbolic.timeout_hours * 3600),
        warm_start=config.symbolic.warm_start,
        populations=config.symbolic.populations,
        population_size=config.symbolic.population_size,
        ncycles_per_iteration=config.symbolic.ncycles_per_iteration,
        model_selection=config.symbolic.model_selection,
        elementwise_loss=config.symbolic.elementwise_loss,
        binary_operators=config.symbolic.binary_operators,
        unary_operators=config.symbolic.unary_operators,
        maxsize=config.symbolic.maxsize,
        maxdepth=config.symbolic.maxdepth,
        parsimony=config.symbolic.parsimony,
        complexity_of_constants=config.symbolic.complexity_of_constants,
        constraints=config.symbolic.constraints,
        nested_constraints=config.symbolic.nested_constraints,
        precision=config.symbolic.precision,
        progress=config.symbolic.progress,
        output_directory=str(config.symbolic.output_directory),
        run_id=config.symbolic.run_id,
        early_stop_condition=config.symbolic.early_stop_condition,
        random_state=config.symbolic.random_state
    )

 
    pysr_model.fit(
        X_train,
        y_train,
        variable_names=selection.names,
        **_pysr_fit_kwargs(config, y_train),
    )

    train_prediction = np.asarray(pysr_model.predict(X_train), dtype=np.float64)
    test_prediction = np.asarray(pysr_model.predict(X_test), dtype=np.float64)
    train_metrics = regression_metrics(train_prediction, y_train)
    test_metrics = regression_metrics(test_prediction, y_test)

    selected_invariants_path = save_json(
        output_dir / "selected_invariants.json",
        {
            "selected_invariants": selection.names,
            "selected_indices": selection.indices,
            "selected_scores": selection.scores,
            "selected_raw_scores": selection.raw_scores,
            "selected_scaled_scores": selection.scaled_scores,
            "feature_selection": selection.feature_selection,
            "all_invariants": config.invariants.selected,
            "invariant_feature_statistics": invariant_stats,
            "encoder_input_feature_statistics": encoder_input_stats,
            "encoder_score_diagnostics": encoder_scores,
            "invariant_normalization": model.invariant_normalization_state(),
            "constraint_diagnostics": constraint_diagnostics(model, config.constraints),
            "checkpoint": str(checkpoint_path),
            "config": None if config_path is None else str(config_path),
        },
    )

    equations_path = output_dir / "equations.csv"
    _save_equations_csv(pysr_model, equations_path)

    best_equation = _best_equation_text(pysr_model)
    best_equation_path = output_dir / "best_equation.txt"
    best_equation_path.write_text(best_equation + "\n", encoding="utf-8")

    metrics_path = save_json(
        output_dir / "metrics.json",
        {
            "train": train_metrics,
            "test": test_metrics,
            "model_selection": config.symbolic.model_selection,
            "niterations": config.symbolic.niterations,
            "elementwise_loss": config.symbolic.elementwise_loss,
            "weight_mode": config.symbolic.weight_mode,
            "target_transform": config.symbolic.target_transform,
            "constraints": config.symbolic.constraints,
            "nested_constraints": config.symbolic.nested_constraints,
            "constraint_diagnostics": constraint_diagnostics(model, config.constraints),
        },
    )

    return SymbolicResult(
        output_dir=output_dir,
        config_snapshot_path=config_snapshot_path,
        selected_invariants_path=selected_invariants_path,
        equations_path=equations_path,
        best_equation_path=best_equation_path,
        metrics_path=metrics_path,
        selected_invariants=selection.names,
        best_equation=best_equation,
    )
