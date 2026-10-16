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
from invariant_generator.input_features import LogFeaturePool, input_feature_names
from invariant_generator.symbolic import (
    _best_equation_text,
    _import_pysr_regressor,
    _pysr_fit_kwargs,
    _save_equations_csv,
    compute_symbolic_target,
    pysr_regressor_options,
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


def _expanded_equation(
    model: InvariantYieldModel,
    pysr_model: Any,
    names: list[str],
    *,
    exact_coefficients: list[list[str]] | None = None,
    transform: str = "identity",
    positive: bool = False,
) -> str:
    """Substitute the learned linear encoder and named input features into PySR."""
    import sympy as sp

    if exact_coefficients is not None:
        coefficients = [[sp.Rational(value) for value in row] for row in exact_coefficients]
        intercepts = [sp.Integer(0)] * len(coefficients)
    else:
        report = encoder_formula_report(model, names, threshold=0.0)
        coefficients = [[sp.Float(w) for w in row] for row in report["raw_coefficients"]]
        intercepts = [sp.Float(b) for b in report["raw_intercept"]]
    raw_symbols = {
        name: sp.Symbol(name, positive=True if positive else None)
        for name in model.invariant_pool.input_names
    }
    feature_expressions = [sp.sympify(name, locals=raw_symbols) for name in names]
    substitutions = {}
    for i, row in enumerate(coefficients):
        linear = intercepts[i] + sum(
            (weight * feature_expressions[j] for j, weight in enumerate(row) if weight != 0),
            sp.Integer(0),
        )
        # exp(sum_j B_j log x_j) -> prod_j x_j^B_j
        value = sp.expand_power_exp(sp.exp(linear)) if transform == "exp" else linear
        substitutions[sp.Symbol(f"J{i + 1}")] = value
    return str(sp.sympify(pysr_model.sympy()).subs(substitutions, simultaneous=True))


def _symbolic_inputs(
    model: InvariantYieldModel,
    X: np.ndarray,
    names: list[str],
    *,
    device: torch.device,
    exact_coefficients: list[list[str]] | None,
    transform: str,
) -> np.ndarray:
    """Encoded variables handed to PySR.

    Without exact coefficients and with the identity transform this is the
    model's own encoded output (the original behaviour). Otherwise the raw
    input features are combined with the raw (or exact snapped) coefficients
    and optionally exponentiated; log features are computed in float64.
    """
    if exact_coefficients is None and transform == "identity":
        return compute_encoded_invariant_features(model, X, device=device, batch_size=8192)
    from fractions import Fraction

    if isinstance(model.invariant_pool, LogFeaturePool):
        raw = np.log(np.asarray(X, dtype=np.float64))
    else:
        with torch.no_grad():
            raw = model.raw_invariant_features(
                torch.as_tensor(X, dtype=torch.float32, device=device)
            ).cpu().numpy().astype(np.float64)
    if exact_coefficients is not None:
        B = np.array([[float(Fraction(v)) for v in row] for row in exact_coefficients])
        intercept = np.zeros(B.shape[0])
    else:
        report = encoder_formula_report(model, names, threshold=0.0)
        B = np.asarray(report["raw_coefficients"], dtype=np.float64)
        intercept = np.asarray(report["raw_intercept"], dtype=np.float64)
    encoded = raw @ B.T + intercept[None, :]
    if transform == "exp":
        return np.exp(encoded)
    if transform != "identity":
        raise ValueError("symbolic.encoded_feature_transform must be 'identity' or 'exp'.")
    return encoded


def _rationalize_floats(expression: Any, *, digits: int, zero_tolerance: float = 0.0) -> Any:
    """Round each float to `digits` significant digits as an exact rational;
    floats below `zero_tolerance` in magnitude become 0."""
    import sympy as sp

    replacements = {}
    for value in expression.atoms(sp.Float):
        number = float(value)
        replacements[value] = (
            sp.Integer(0) if abs(number) < zero_tolerance else sp.Rational(f"{number:.{digits}g}")
        )
    return expression.xreplace(replacements)


class _Timeout(Exception):
    pass


def _with_time_limit(seconds: int, function: Any) -> Any:
    """Run function() with a SIGALRM limit when possible; None on timeout."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        return function()

    def _raise(*_: Any) -> None:
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        return function()
    except _Timeout:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def symbolic_ratio_check(
    candidate: Any,
    truth: Any,
    *,
    digits: int = 5,
    residual_tolerance: float = 0.0,
) -> tuple[bool, str]:
    """SRBench-style check that candidate / truth is a nonzero constant.

    Fitted constants (0.119366 for 3/(8*pi)) never cancel exactly, so the
    simplified ratio is evaluated numerically (pi -> float), its floats are
    rounded to `digits` significant digits as exact rationals, and it is
    simplified again; floats below `residual_tolerance` become 0. The squared
    ratio is tried as well so constants can pass through square roots.

    Advisory only: rounding can split two equal constants across a rounding
    boundary, and cancellation leftovers such as exp(-1e-12*(a + b)**2) survive,
    so False is weak evidence. Zeroing leftovers (residual_tolerance > 0) also
    zeroes small coefficients of large rationalized ratios. On the Feynman
    pilot it gave no false positives and 3 false negatives out of 21
    numerically exact candidates, all confirmed correct by reading them.
    """
    import sympy as sp

    last = None
    for expression in (candidate / truth, (candidate / truth) ** 2):
        ratio = expression
        for _ in range(2):
            ratio = sp.simplify(sp.powsimp(ratio, force=True))
            ratio = _rationalize_floats(
                ratio.evalf(), digits=digits, zero_tolerance=residual_tolerance,
            )
        ratio = sp.simplify(ratio)
        last = ratio
        if not ratio.free_symbols and ratio != 0:
            return True, str(ratio)
    return False, str(last)


def recovery_report(
    expanded: str,
    truth: Any,
    symbols: dict[str, Any],
    *,
    X_fresh: np.ndarray,
    numerical_tolerance: float = 1e-6,
    zero_tolerance: float = 1e-6,
    symbolic_seconds: int = 120,
) -> dict[str, object]:
    """Compare a candidate in the original inputs with the known formula.

    - numerical_recovery (headline): relative L2 error below
      `numerical_tolerance` on fresh points over the whole input domain.
    - symbolic_recovery: candidate / truth simplifies to a nonzero constant
      after constants below `zero_tolerance` are dropped from the candidate
      (see symbolic_ratio_check). None when SymPy exceeds `symbolic_seconds`.
    - exact_symbolic_recovery: candidate - truth simplifies to exactly 0;
      fitted floats rarely allow this.
    """
    import sympy as sp

    candidate = sp.sympify(expanded, locals=symbols)
    ordered = list(symbols.values())
    columns = [X_fresh[:, idx] for idx in range(X_fresh.shape[1])]
    with np.errstate(all="ignore"):
        candidate_values = np.broadcast_to(
            np.asarray(sp.lambdify(ordered, candidate, "numpy")(*columns), dtype=np.complex128),
            (X_fresh.shape[0],),
        )
        truth_values = np.broadcast_to(
            np.asarray(sp.lambdify(ordered, truth, "numpy")(*columns), dtype=np.float64),
            (X_fresh.shape[0],),
        )
        relative_l2 = float(
            np.linalg.norm(candidate_values - truth_values) / np.linalg.norm(truth_values)
        )
    numerical = bool(np.isfinite(relative_l2) and relative_l2 < numerical_tolerance)
    strict: bool | None = False
    symbolic: bool | None = False
    ratio: str | None = None
    if np.isfinite(relative_l2) and relative_l2 <= 1e-2:
        # Larger errors cannot be a recovery; skipping them also avoids
        # simplify() stalling on large wrong candidates.
        cleaned = _rationalize_floats(candidate, digits=17, zero_tolerance=zero_tolerance)
        strict = _with_time_limit(symbolic_seconds, lambda: bool(sp.simplify(candidate - truth) == 0))
        checked = _with_time_limit(symbolic_seconds, lambda: symbolic_ratio_check(cleaned, truth))
        symbolic, ratio = (None, None) if checked is None else checked
    return {
        "numerical_recovery": numerical,
        "symbolic_recovery": symbolic,
        "exact_symbolic_recovery": strict,
        "candidate_over_truth": ratio,
        "fresh_sample_relative_l2": relative_l2,
        "fresh_samples": int(X_fresh.shape[0]),
        "numerical_criterion": f"relative L2 on fresh samples < {numerical_tolerance:g}",
        "symbolic_criterion": (
            f"candidate constants below {zero_tolerance:g} dropped; candidate/truth (or its "
            "square) simplifies to a nonzero constant after rounding floats to 5 significant digits"
        ),
    }


def load_known_formula(
    config: Config,
    *,
    fallback_X: np.ndarray,
    n_fresh: int = 10_000,
) -> tuple[Any, dict[str, Any]] | None:
    """Read the generating formula from `<dataset>.json`, if the dataset has one.

    Returns (truth, info) with the symbols, fresh evaluation points drawn over
    the published input ranges (not the training rows), and whether all inputs
    are positive. The pipeline itself never uses this; it is for evaluation.
    """
    import json
    import sympy as sp

    metadata_path = (config.data.data_dir / config.data.dataset_name).with_suffix(".json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "formula" not in metadata:
        return None
    ranges = metadata.get("input_ranges", {})
    names = list(config.data.feature_names)
    positive = all(name in ranges and ranges[name][0] > 0 for name in names)
    symbols = {name: sp.Symbol(name, positive=True if positive else None) for name in names}
    if "substitution" in metadata:
        # Hidden-variable pilots write the formula in theta and say what replaced it.
        theta = sp.Symbol("theta")
        truth = sp.sympify(metadata["formula"], locals={**symbols, "theta": theta})
        substitution = metadata["substitution"].split("=", 1)[-1]
        truth = truth.subs(theta, sp.sympify(substitution, locals=symbols))
    else:
        truth = sp.sympify(metadata["formula"], locals=symbols)
    rng = np.random.default_rng(config.symbolic.random_state)
    X_fresh = (
        np.column_stack([rng.uniform(*ranges[name], size=n_fresh) for name in names])
        if all(name in ranges for name in names)
        else fallback_X
    )
    return truth, {"symbols": symbols, "X_fresh": X_fresh, "positive": positive}


def original_input_evaluation(
    expanded: str,
    known: tuple[Any, dict[str, Any]] | None,
    *,
    test_prediction: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    recovery: dict[str, object] = {"exact_symbolic_recovery": None}
    if known is not None:
        truth, info = known
        try:
            recovery = recovery_report(expanded, truth, info["symbols"], X_fresh=info["X_fresh"])
        except Exception as exc:  # A malformed candidate must not lose the run.
            recovery = {"exact_symbolic_recovery": None, "recovery_error": repr(exc)}
    return {
        "expanded_equation": expanded,
        "truth_expression": None if known is None else str(known[0]),
        **recovery,
        "test_relative_l2": float(
            np.linalg.norm(np.asarray(test_prediction) - y_test) / np.linalg.norm(y_test)
        ),
        "criterion": "SymPy simplification of candidate minus truth equals zero",
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
    feature_names = input_feature_names(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    exact_coefficients = checkpoint.get("exact_raw_coefficients")
    transform = config.symbolic.encoded_feature_transform.lower()

    def symbolic_inputs(X: np.ndarray) -> np.ndarray:
        return _symbolic_inputs(
            model, X, feature_names, device=device,
            exact_coefficients=exact_coefficients, transform=transform,
        )

    X_train = symbolic_inputs(data.X_train)
    X_test = symbolic_inputs(data.X_test)
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
        feature_names,
        threshold=config.sparsification.threshold,
    )
    formulas_path = save_json(
        output_dir / "encoded_invariant_formulas.json",
        {
            "feature_space": "encoded_invariants",
            "variables": variable_names,
            "input_transformation": config.input_transform.kind,
            "input_feature_formulas": feature_names,
            "encoded_feature_transform": transform,
            "exact_raw_coefficients": exact_coefficients,
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

    pysr_model = PySRRegressor(**pysr_regressor_options(
        config, output_directory=output_dir,
    ))
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
    if config.data.kind == "supervised":
        known = load_known_formula(config, fallback_X=data.X_test)
        expanded = _expanded_equation(
            model, pysr_model, feature_names,
            exact_coefficients=exact_coefficients,
            transform=transform,
            positive=known is not None and known[1]["positive"],
        )
        (output_dir / "equation_in_original_inputs.txt").write_text(
            expanded + "\n", encoding="utf-8",
        )
        np.savez_compressed(
            output_dir / "heldout_predictions.npz",
            X_validation=data.X_validation,
            y_validation=data.y_validation,
            X_test=data.X_test,
            y_test=data.y_test,
            prediction_test=test_prediction,
        )
        save_json(
            output_dir / "original_input_evaluation.json",
            original_input_evaluation(
                expanded, known, test_prediction=test_prediction, y_test=data.y_test,
            ),
        )
    metrics_path = save_json(
        output_dir / "metrics.json",
        {
            "feature_space": "encoded_invariants",
            "variables": variable_names,
            "target_source": config.symbolic.target_source,
            "target_transform": config.symbolic.target_transform,
            "train": regression_metrics(train_prediction, y_train),
            "validation": None if data.X_validation is None else regression_metrics(
                np.asarray(pysr_model.predict(symbolic_inputs(data.X_validation)), dtype=np.float64),
                compute_symbolic_target(
                    model, data.X_validation, data.y_validation,
                    target_source=config.symbolic.target_source,
                    target_transform=config.symbolic.target_transform,
                    device=device,
                ),
            ),
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
