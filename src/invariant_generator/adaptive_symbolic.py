from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from invariant_generator.adaptive import adaptive_results_dir
from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import regression_metrics
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.model import InvariantYieldModel
from invariant_generator.symbolic import (
    _best_equation_text,
    _import_pysr_regressor,
    _pysr_fit_kwargs,
    _save_equations_csv,
    compute_symbolic_target,
)
from invariant_generator.utils import resolve_device, save_config_snapshot, save_json


@dataclass(slots=True)
class EncodedSymbolicResult:
    output_dir: Path
    config_snapshot_path: Path
    formulas_path: Path
    equations_path: Path
    best_equation_path: Path
    metrics_path: Path
    best_equation: str


@torch.no_grad()
def compute_encoded_invariant_features(
    model: InvariantYieldModel,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    outputs: list[np.ndarray] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size].to(device)
        features = model.invariant_features(batch)
        outputs.append(features.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _feature_statistics(values: np.ndarray, names: list[str]) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "names": list(names),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def train_encoded_symbolic_from_config(
    config: Config,
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> EncodedSymbolicResult:
    PySRRegressor = _import_pysr_regressor()
    device = resolve_device(config.train.device)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.experiment_dir / "checkpoint_best.pt"
    )
    output_dir = adaptive_results_dir(config) / config.symbolic.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = save_config_snapshot(config, output_dir)

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    X_train = compute_encoded_invariant_features(
        model,
        data.X_train,
        device=device,
        batch_size=8192,
    )
    X_test = compute_encoded_invariant_features(
        model,
        data.X_test,
        device=device,
        batch_size=8192,
    )
    variable_names = [f"J{i + 1}" for i in range(X_train.shape[1])]
    y_train = compute_symbolic_target(
        model,
        data.X_train,
        data.y_train,
        target_source=config.symbolic.target_source,
        target_transform=config.symbolic.target_transform,
        device=device,
    )
    y_test = compute_symbolic_target(
        model,
        data.X_test,
        data.y_test,
        target_source=config.symbolic.target_source,
        target_transform=config.symbolic.target_transform,
        device=device,
    )

    formula_report = encoder_formula_report(
        model,
        config.invariants.selected,
        threshold=config.sparsification.threshold,
    )
    formulas_path = save_json(
        output_dir / "encoded_invariant_formulas.json",
        {
            "feature_space": "encoded_invariants",
            "variables": variable_names,
            "source_checkpoint": str(checkpoint_path),
            "config": None if config_path is None else str(config_path),
            **formula_report,
        },
    )
    save_json(
        output_dir / "preflight_diagnostics.json",
        {
            "feature_space": "encoded_invariants",
            "variables": variable_names,
            "feature_statistics": {
                "train": _feature_statistics(X_train, variable_names),
                "test": _feature_statistics(X_test, variable_names),
            },
            "target_source": config.symbolic.target_source,
            "target_transform": config.symbolic.target_transform,
        },
    )

    pysr_model = PySRRegressor(
        parallelism=config.symbolic.parallelism,
        procs=config.symbolic.numprocs,
        batching=config.symbolic.batching,
        batch_size=config.symbolic.batch_size,
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
        output_directory=str(output_dir),
        run_id=config.symbolic.run_id,
        early_stop_condition=config.symbolic.early_stop_condition,
        random_state=config.symbolic.random_state,
    )
    pysr_model.fit(
        X_train,
        y_train,
        variable_names=variable_names,
        **_pysr_fit_kwargs(config, y_train),
    )

    train_prediction = np.asarray(pysr_model.predict(X_train), dtype=np.float64)
    test_prediction = np.asarray(pysr_model.predict(X_test), dtype=np.float64)
    equations_path = output_dir / "equations.csv"
    _save_equations_csv(pysr_model, equations_path)

    best_equation = _best_equation_text(pysr_model)
    best_equation_path = output_dir / "best_equation.txt"
    best_equation_path.write_text(best_equation + "\n", encoding="utf-8")
    metrics_path = save_json(
        output_dir / "metrics.json",
        {
            "feature_space": "encoded_invariants",
            "variables": variable_names,
            "target_source": config.symbolic.target_source,
            "target_transform": config.symbolic.target_transform,
            "train": regression_metrics(train_prediction, y_train),
            "test": regression_metrics(test_prediction, y_test),
            "model_selection": config.symbolic.model_selection,
            "niterations": config.symbolic.niterations,
            "elementwise_loss": config.symbolic.elementwise_loss,
            "weight_mode": config.symbolic.weight_mode,
            "formulas": str(formulas_path),
        },
    )

    return EncodedSymbolicResult(
        output_dir=output_dir,
        config_snapshot_path=config_snapshot_path,
        formulas_path=formulas_path,
        equations_path=equations_path,
        best_equation_path=best_equation_path,
        metrics_path=metrics_path,
        best_equation=best_equation,
    )
