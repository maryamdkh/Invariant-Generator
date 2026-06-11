from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import regression_metrics
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import resolve_device, save_json


@dataclass(slots=True)
class InvariantSelection:
    names: list[str]
    indices: list[int]
    scores: list[float]


@dataclass(slots=True)
class SymbolicResult:
    output_dir: Path
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
) -> InvariantSelection:
    """Rank original invariant columns by encoder weight norm."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if encoder_matrix.ndim != 2:
        raise ValueError(f"encoder_matrix must be 2D, got shape {encoder_matrix.shape}.")
    if encoder_matrix.shape[1] != len(invariant_names):
        raise ValueError(
            "encoder_matrix columns must match invariant_names. "
            f"Got {encoder_matrix.shape[1]} columns and {len(invariant_names)} names."
        )

    scores_tensor = torch.linalg.vector_norm(
        encoder_matrix.detach().float().cpu(),
        ord=2,
        dim=0,
    )
    scores = [float(score) for score in scores_tensor]
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (-item[1], item[0]),
    )
    selected = ranked[: min(top_k, len(ranked))]

    return InvariantSelection(
        names=[invariant_names[idx] for idx, _ in selected],
        indices=[idx for idx, _ in selected],
        scores=[score for _, score in selected],
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


def train_symbolic_from_config(
    config: Config,
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> SymbolicResult:
    """Run post-training PySR on top-ranked invariant features."""
    if not config.encoder.enabled:
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

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    S = model.encoder_matrix()
    if S is None:
        raise ValueError("Loaded model does not have an encoder matrix S.")

    selection = select_top_invariants_from_encoder(
        S,
        config.invariants.selected,
        top_k=config.symbolic.top_k,
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

    pysr_model = PySRRegressor(
        niterations=config.symbolic.niterations,
        binary_operators=config.symbolic.binary_operators,
        unary_operators=config.symbolic.unary_operators,
        model_selection=config.symbolic.model_selection,
        random_state=config.symbolic.random_state,
    )
    pysr_model.fit(X_train, data.y_train, variable_names=selection.names)

    train_prediction = np.asarray(pysr_model.predict(X_train), dtype=np.float64)
    test_prediction = np.asarray(pysr_model.predict(X_test), dtype=np.float64)
    train_metrics = regression_metrics(train_prediction, data.y_train)
    test_metrics = regression_metrics(test_prediction, data.y_test)

    output_dir = config.experiment_dir / config.symbolic.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_invariants_path = save_json(
        output_dir / "selected_invariants.json",
        {
            "selected_invariants": selection.names,
            "selected_indices": selection.indices,
            "selected_scores": selection.scores,
            "all_invariants": config.invariants.selected,
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
        },
    )

    return SymbolicResult(
        output_dir=output_dir,
        selected_invariants_path=selected_invariants_path,
        equations_path=equations_path,
        best_equation_path=best_equation_path,
        metrics_path=metrics_path,
        selected_invariants=selection.names,
        best_equation=best_equation,
    )
