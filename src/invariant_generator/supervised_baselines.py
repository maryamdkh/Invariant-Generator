"""Comparison arms for supervised benchmarks (Feynman, LLM-SRBench, synthetic).

Opt-in arms of feynman_benchmark.run_benchmark, scored like the PySR-alone
baseline (original_input_evaluation: fresh-sample relative L2 against the
known formula):
  distillation   (C) MLP with an L1 latent, PySR on latent and outer map
  sparse_linear  (D) STLSQ: y^m = polynomial(raw inputs), m = 1, 2, 3
  library_pysr   (E) PySR on the pipeline's feature library, no encoder
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np
import sympy as sp
import torch

from invariant_generator.adaptive_symbolic import load_known_formula, original_input_evaluation
from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.distillation import DistillationSettings, distill
from invariant_generator.input_features import input_feature_names
from invariant_generator.model import InvariantYieldModel
from invariant_generator.symbolic import (
    _best_equation_text,
    _import_pysr_regressor,
    _save_equations_csv,
    pysr_regressor_options,
)
from invariant_generator.utils import save_json
from invariant_generator.yield_benchmark import _inverse_power, _polynomial_library, _relative_l2, stlsq


def _stop_condition(config: Config, y: np.ndarray) -> Config:
    config = deepcopy(config)
    threshold = max(float(np.var(y)) * 1e-12, 1e-300)
    config.symbolic.early_stop_condition = f"stop_if(loss, complexity) = loss < {threshold:.6e}"
    return config


def _raw_config(config: Config) -> Config:
    """Raw inputs and y itself, as for the PySR-alone baseline."""
    config = deepcopy(config)
    config.data.target_transform = "identity"
    config.input_transform.kind = "pairwise"
    config.input_transform.include_ratios = False
    return config


def _evaluate(expression: str, config: Config, prediction_test: np.ndarray, y_test: np.ndarray, data) -> dict[str, Any]:
    known = load_known_formula(config, fallback_X=data.X_test)
    return original_input_evaluation(expression, known, test_prediction=prediction_test, y_test=y_test)


def run_distillation(config: Config, output_dir: Path) -> dict[str, Any]:
    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _raw_config(config)
    data = prepare_training_data(config)
    names = list(config.data.feature_names)

    def fit_pysr(X: np.ndarray, target: np.ndarray, variables: list[str], directory: Path, seconds: float) -> Any:
        search = _stop_condition(config, target)
        search.symbolic.timeout_hours = seconds / 3600.0
        directory.mkdir(parents=True, exist_ok=True)
        model = PySRRegressor(**pysr_regressor_options(search, output_directory=directory))
        model.fit(X, target, variable_names=variables)
        return model

    neutral = [f"x{i}" for i in range(len(names))]
    predict, info = distill(
        data.X_train, data.y_train, neutral, fit_pysr=fit_pysr, output_dir=output_dir,
        total_pysr_seconds=float(config.symbolic.timeout_hours) * 3600.0,
        settings=DistillationSettings(seed=config.train.seed),
    )
    symbols = {f"x{i}": sp.Symbol(name) for i, name in enumerate(names)}
    outer = sp.sympify(info["outer_equation"])
    inner = {sp.Symbol(z): sp.sympify(expression).subs(symbols, simultaneous=True)
             for z, expression in info["latent_equations"].items()}
    composed = outer.subs(inner, simultaneous=True)
    result = {**info, **_evaluate(str(composed), config, predict(data.X_test), data.y_test, data)}
    save_json(output_dir / "distillation_evaluation.json", result)
    return result


def run_sparse_linear(config: Config, output_dir: Path, *, degree: int = 3,
                      powers: tuple[int, ...] = (1, 2, 3),
                      thresholds: tuple[float, ...] = tuple(10.0 ** np.arange(-8, 0))) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _raw_config(config)
    data = prepare_training_data(config)
    names = list(config.data.feature_names)
    Theta, terms = _polynomial_library(data.X_train, degree)
    holdout = np.random.default_rng(config.train.seed).random(len(Theta)) < 0.1
    started = time.perf_counter()
    best = None
    for m in powers:
        if m % 2 == 0 and np.any(data.y_train < 0):
            continue
        for threshold in thresholds:
            xi = stlsq(Theta[~holdout], data.y_train[~holdout] ** m, threshold)
            error = _relative_l2(_inverse_power(Theta[holdout] @ xi, m), data.y_train[holdout])
            candidate = (error, int(np.count_nonzero(xi)), m, threshold)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    _, _, m, threshold = best
    xi = stlsq(Theta, data.y_train ** m, threshold)
    symbols = [sp.Symbol(name) for name in names]
    polynomial = sum(float(c) * sp.Mul(*[symbols[i] for i in t]) for c, t in zip(xi, terms) if c != 0)
    expression = polynomial if m == 1 else (sp.sqrt(polynomial) if m == 2 else sp.cbrt(polynomial))
    prediction = _inverse_power(_polynomial_library(data.X_test, degree)[0] @ xi, m)
    result = {
        "power": m, "threshold": threshold, "active_terms": int(np.count_nonzero(xi)),
        "library_size": len(terms), "fit_seconds": time.perf_counter() - started,
        **_evaluate(str(expression), config, prediction, data.y_test, data),
    }
    save_json(output_dir / "sparse_linear_evaluation.json", result)
    return result


def run_library_pysr(config: Config, output_dir: Path) -> dict[str, Any]:
    """PySR on the encoder's input features (log, products, ...) without the encoder."""
    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(config)
    config.data.target_transform = "identity"
    data = prepare_training_data(config)
    pool = InvariantYieldModel.from_config(config).double().invariant_pool

    def features(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return pool(torch.as_tensor(X, dtype=torch.float64)).numpy()

    feature_names = input_feature_names(config)
    variables = [f"F{i + 1}" for i in range(len(feature_names))]
    search = _stop_condition(config, data.y_train)
    model = PySRRegressor(**pysr_regressor_options(search, output_directory=output_dir))
    started = time.perf_counter()
    model.fit(features(data.X_train), data.y_train, variable_names=variables)
    elapsed = time.perf_counter() - started
    _save_equations_csv(model, output_dir / "equations.csv")
    symbols = {name: sp.Symbol(name) for name in config.data.feature_names}
    substitution = {sp.Symbol(v): sp.sympify(f, locals=symbols) for v, f in zip(variables, feature_names)}
    expression = sp.sympify(model.sympy()).subs(substitution, simultaneous=True)
    result = {
        "library_size": len(feature_names),
        "best_equation_in_features": _best_equation_text(model),
        "pysr_seconds": elapsed,
        **_evaluate(str(expression), config, np.asarray(model.predict(features(data.X_test))), data.y_test, data),
    }
    save_json(output_dir / "library_pysr_evaluation.json", result)
    return result
