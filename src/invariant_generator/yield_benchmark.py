"""Yield-surface benchmark: compare methods on surface datasets with a known criterion.

Each arm is scored the same way: its prediction on fresh stress points (new
directions on the true surface, scaled by k as in the homogeneity
augmentation) against the generating criterion, recovered when the relative
L2 error is below RECOVERY_THRESHOLD. Rows use the layout of feynman_benchmark.run_benchmark
(one benchmark_summary.json per task), so the cluster job script, merge and
export scripts apply unchanged.

Arms: pipeline (A, Stages 1-3), baseline (B, PySR on the raw stress
components; components that are zero in every sample, e.g. plane stress, are
dropped), distillation (C, invariant_generator.distillation),
library_pysr (E, PySR on the feature library with the structural
tensors fixed to identity, no encoder), sparse_linear (D, sequentially
thresholded least squares on polynomials of the raw components, fitting
y^m for m = 1, 2, 3).
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

from invariant_generator.adaptive import (
    adaptive_sparsification_run_id,
    config_for_adaptive_n,
    run_adaptive_sweep,
)
from invariant_generator.adaptive_symbolic import _symbolic_inputs, train_encoded_symbolic_from_config
from invariant_generator.config import Config
from invariant_generator.data import _resolve_hdf_path, prepare_training_data
from invariant_generator.input_features import input_feature_names
from invariant_generator.model import InvariantYieldModel
from invariant_generator.pysr_watch import LossWatcher, recovery_loss_threshold
from invariant_generator.sparsify import load_model_for_checkpoint, sparsify_encoder_from_checkpoint
from invariant_generator.symbolic import (
    _best_equation_text,
    _import_pysr_regressor,
    _pysr_fit_kwargs,
    _save_equations_csv,
    pysr_regressor_options,
)
from invariant_generator.utils import resolve_device, save_json
from invariant_generator.yield_criteria import CRITERIA

# Relative L2 1e-4 = PySR loss ~1e-8 under inverse_target_squared weights,
# within the loss < 1e-7 usually taken to mean the equation was found.
# The earlier 1e-6 failed float32-learned tensors with the right structure (RESEARCH_LOG 49).
RECOVERY_THRESHOLD = 1e-4
PLANE_STRESS = [0, 1, 5]  # s11, s22, s12 in canonical Voigt order


def load_truth(config: Config) -> tuple[Callable[[np.ndarray], np.ndarray], bool] | None:
    """Equivalent stress f(sigma) (canonical Voigt input) from the dataset JSON,
    and whether the data is plane stress; None when there is no ground truth."""
    meta_path = _resolve_hdf_path(config.data.data_dir, config.data.dataset_name).with_suffix(".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "M_mandel" in meta:  # scripts/prepare_hill_ladder.py
        M = np.asarray(meta["M_mandel"], dtype=np.float64)
        plane = meta["stress_format"] == "plane_stress_2d"

        def quadratic(X: np.ndarray) -> np.ndarray:
            mandel = np.asarray(X, dtype=np.float64)[:, PLANE_STRESS] if plane else np.array(X, dtype=np.float64)
            if plane:
                mandel[:, 2] *= np.sqrt(2.0)
            else:
                mandel[:, 3:] *= np.sqrt(2.0)
            return np.sqrt(np.einsum("ni,ij,nj->n", mandel, M, mandel))

        return quadratic, plane
    if meta.get("criterion") in CRITERIA:  # scripts/prepare_yield_benchmark.py
        return CRITERIA[meta["criterion"]].equivalent_stress, False
    return None


def fresh_points(config: Config, truth: Callable[[np.ndarray], np.ndarray], plane: bool,
                 *, samples: int = 5000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """New surface directions scaled by k in the augmentation range; target
    k^p * surface_target as in the training data."""
    rng = np.random.default_rng(seed)
    X = np.zeros((samples, 6))
    columns = PLANE_STRESS if plane else list(range(6))
    X[:, columns] = rng.normal(size=(samples, len(columns)))
    X /= truth(X)[:, None]
    low, high = config.augmentation.k_range
    k = rng.uniform(low, high, size=samples)
    y = k ** config.augmentation.homogeneity_degree * config.augmentation.surface_target
    return X * k[:, None], y


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _score(prediction_fresh: np.ndarray, y_fresh: np.ndarray,
           prediction_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    fresh = _relative_l2(prediction_fresh, y_fresh)
    return {
        "fresh_sample_relative_l2": fresh,
        "fresh_samples": int(len(y_fresh)),
        "test_relative_l2": _relative_l2(prediction_test, y_test),
        "numerical_recovery": bool(fresh < RECOVERY_THRESHOLD),
        "numerical_criterion": f"relative L2 on fresh surface samples < {RECOVERY_THRESHOLD:g}",
    }


def _with_stop_condition(config: Config, y: np.ndarray) -> Config:
    """Copy of the config for one PySR search. The yield configs' own PySR
    settings (stop condition, time limit) are kept: they were tuned on these
    data, and runs are usually stopped by hand (q + Enter) once the loss
    has converged."""
    del y
    return deepcopy(config)


def run_baseline(config: Config, output_dir: Path, fresh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """PySR alone on the raw stress components."""
    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(config)
    active = [j for j in range(data.X_train.shape[1]) if np.any(data.X_train[:, j] != 0)]
    names = [data.feature_names[j] for j in active]
    config = _with_stop_condition(config, data.y_train)
    watch = LossWatcher(output_dir, recovery_loss_threshold(config.symbolic.weight_mode, data.y_train, RECOVERY_THRESHOLD))
    stop_rule = config.symbolic.early_stop_condition
    config.symbolic.early_stop_condition = watch.julia_stop_condition(stop_rule)
    model = PySRRegressor(**pysr_regressor_options(config, output_directory=output_dir))
    started = time.perf_counter()
    with watch:
        model.fit(data.X_train[:, active], data.y_train, variable_names=names,
                  **_pysr_fit_kwargs(config, data.y_train))
    elapsed = time.perf_counter() - started
    _save_equations_csv(model, output_dir / "equations.csv")
    X_fresh, y_fresh = fresh
    result = {
        "inputs": names,
        "expanded_equation": _best_equation_text(model),
        "pysr_seconds": elapsed,
        "time_to_recovery_s": watch.time_to_threshold,
        **watch.report(),
        "early_stop_condition": stop_rule,
        **_score(model.predict(X_fresh[:, active]), y_fresh,
                 model.predict(data.X_test[:, active]), data.y_test),
    }
    save_json(output_dir / "baseline_evaluation.json", result)
    return result


def _evaluate_equation(equation: str, J: np.ndarray) -> np.ndarray:
    """Evaluate a PySR equation in J1..Jn (PySR's own operator names)."""
    namespace = {
        "square": np.square, "cube": lambda v: v**3, "cbrt": np.cbrt, "sqrt": np.sqrt,
        "exp": np.exp, "log": np.log, "abs": np.abs,
        **{f"J{i + 1}": J[:, i] for i in range(J.shape[1])},
    }
    with np.errstate(all="ignore"):
        value = eval(equation, {"__builtins__": {}}, namespace)  # noqa: S307 (our own PySR output)
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (J.shape[0],)).copy()


def run_pipeline(config: Config, fresh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Stage 1 -> Stage 2 -> Stage 3, scored on fresh surface points."""
    result: dict[str, Any] = {}
    started = time.perf_counter()
    sweep = run_adaptive_sweep(config)
    result["stage1_seconds"] = time.perf_counter() - started
    result["stage1_runs"] = [
        {"n": run.n, "val_metric": run.test_metrics.get(config.adaptive.metric)} for run in sweep.runs
    ]
    result["selected_n"] = sweep.selected_n
    if sweep.selected_n is None:
        result["reduction"] = False
        result["numerical_recovery"] = False
        result["note"] = "Stage 1 accepted no n"
        return result
    result["reduction"] = True

    stage_config = config_for_adaptive_n(config, sweep.selected_n)
    started = time.perf_counter()
    sparse = sparsify_encoder_from_checkpoint(stage_config, checkpoint_path=sweep.selected_checkpoint)
    result["stage2_seconds"] = time.perf_counter() - started
    summary = json.loads(sparse.summary_path.read_text(encoding="utf-8"))
    result["stage2_val_metric"] = summary["test_metrics"].get(config.adaptive.metric)
    result["encoder_formulas"] = summary.get("formulas")

    symbolic_config = config_for_adaptive_n(config, sweep.selected_n)
    symbolic_config.train.run_id = adaptive_sparsification_run_id(symbolic_config)
    data = prepare_training_data(symbolic_config)
    symbolic_config = _with_stop_condition(symbolic_config, data.y_train)
    watch = LossWatcher(Path(symbolic_config.symbolic.output_directory),
                        recovery_loss_threshold(symbolic_config.symbolic.weight_mode, data.y_train, RECOVERY_THRESHOLD))
    symbolic_config.symbolic.early_stop_condition = watch.julia_stop_condition(
        symbolic_config.symbolic.early_stop_condition
    )
    started = time.perf_counter()
    with watch:
        stage3 = train_encoded_symbolic_from_config(symbolic_config, checkpoint_path=sparse.checkpoint_path)
    result["pysr_seconds"] = time.perf_counter() - started
    result.update(watch.report())
    # Stages 1-2 run once before the search, so they count towards the time to recovery.
    result["time_to_recovery_s"] = (
        None if watch.time_to_threshold is None
        else result["stage1_seconds"] + result["stage2_seconds"] + watch.time_to_threshold
    )
    result["best_equation_in_groups"] = stage3.best_equation
    formulas = (summary.get("formulas") or {}).get("formulas") or []
    result["expanded_equation"] = stage3.best_equation + "".join(
        f"; {f['raw_formula']}" for f in formulas if "raw_formula" in f
    )

    device = resolve_device(config.train.device)
    model = load_model_for_checkpoint(stage_config, sparse.checkpoint_path).to(device)
    checkpoint = torch.load(sparse.checkpoint_path, map_location=device, weights_only=False)

    def predict(X: np.ndarray) -> np.ndarray:
        J = _symbolic_inputs(
            model, X, input_feature_names(config), device=device,
            exact_coefficients=checkpoint.get("exact_raw_coefficients"),
            transform=config.symbolic.encoded_feature_transform.lower(),
        )
        return _evaluate_equation(stage3.best_equation, J)

    X_fresh, y_fresh = fresh
    result.update(_score(predict(X_fresh), y_fresh, predict(data.X_test), data.y_test))
    return result


def fixed_feature_library(config: Config, X_reference: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    """The encoder's input features with structural tensors fixed to identity
    (no learned anisotropy). Columns that vanish or duplicate an earlier
    column on X_reference are dropped (with a = A = I, e.g. I4 = I1, I8 = 0)."""
    model = InvariantYieldModel.from_config(config).double()
    pool = model.invariant_pool
    with torch.no_grad():
        if getattr(pool, "raw_a", None) is not None:
            pool.raw_a.copy_(torch.eye(3, dtype=torch.float64))
        if getattr(pool, "raw_A", None) is not None:
            if tuple(pool.raw_A.shape) == (6, 6):  # Mandel factor (hard PSD)
                pool.raw_A.copy_(torch.eye(6, dtype=torch.float64))
            else:
                eye = torch.eye(3, dtype=torch.float64)
                pool.raw_A.copy_(0.5 * (torch.einsum("ik,jl->ijkl", eye, eye) + torch.einsum("il,jk->ijkl", eye, eye)))

    def features(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model.raw_invariant_features(torch.as_tensor(X, dtype=torch.float64)).numpy()

    names = input_feature_names(config)
    F = features(X_reference)
    keep: list[int] = []
    for j in range(F.shape[1]):
        column = F[:, j]
        scale = np.max(np.abs(column))
        if not np.isfinite(scale) or scale < 1e-12:
            continue
        if any(np.allclose(column, F[:, k], rtol=1e-9, atol=1e-12 * scale) for k in keep):
            continue
        keep.append(j)
    return (lambda X: features(X)[:, keep]), [names[j] for j in keep]


def run_library_pysr(config: Config, output_dir: Path, fresh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """PySR on the feature library itself, without the encoder."""
    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(config)
    features, names = fixed_feature_library(config, data.X_train)
    F_train = features(data.X_train)
    variables = [f"F{i + 1}" for i in range(len(names))]  # neutral names for Julia/SymPy
    config = _with_stop_condition(config, data.y_train)
    watch = LossWatcher(output_dir, recovery_loss_threshold(config.symbolic.weight_mode, data.y_train, RECOVERY_THRESHOLD))
    config.symbolic.early_stop_condition = watch.julia_stop_condition(config.symbolic.early_stop_condition)
    model = PySRRegressor(**pysr_regressor_options(config, output_directory=output_dir))
    started = time.perf_counter()
    with watch:
        model.fit(F_train, data.y_train, variable_names=variables, **_pysr_fit_kwargs(config, data.y_train))
    elapsed = time.perf_counter() - started
    _save_equations_csv(model, output_dir / "equations.csv")
    X_fresh, y_fresh = fresh
    result = {
        "library": dict(zip(variables, names)),
        "expanded_equation": _best_equation_text(model),
        "pysr_seconds": elapsed,
        "time_to_recovery_s": watch.time_to_threshold,
        **watch.report(),
        **_score(model.predict(features(X_fresh)), y_fresh,
                 model.predict(features(data.X_test)), data.y_test),
    }
    save_json(output_dir / "library_pysr_evaluation.json", result)
    return result


def _polynomial_library(X: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    from itertools import combinations_with_replacement

    terms = [()] + [t for d in range(1, degree + 1) for t in combinations_with_replacement(range(X.shape[1]), d)]
    columns = [np.prod(X[:, list(t)], axis=1) if t else np.ones(len(X)) for t in terms]
    return np.column_stack(columns), terms


def stlsq(Theta: np.ndarray, target: np.ndarray, threshold: float, *, ridge: float = 1e-10,
          iterations: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares (SINDy) on standardized columns."""
    scale = np.linalg.norm(Theta, axis=0)
    scale[scale == 0] = 1.0
    T = Theta / scale
    active = np.ones(T.shape[1], dtype=bool)
    xi = np.zeros(T.shape[1])
    for _ in range(iterations):
        if not active.any():
            break
        A = T[:, active]
        xi_active = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ target)
        xi = np.zeros(T.shape[1])
        xi[active] = xi_active
        small = np.abs(xi) < threshold * np.max(np.abs(xi))
        if not (active & small).any():
            break
        active &= ~small
    return xi / scale


def _inverse_power(value: np.ndarray, m: int) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** (1.0 / m) if m % 2 else np.clip(value, 0.0, None) ** (1.0 / m)


def run_sparse_linear(config: Config, output_dir: Path, fresh: tuple[np.ndarray, np.ndarray],
                      *, degree: int = 3, max_power: int = 3,
                      thresholds: tuple[float, ...] = tuple(10.0 ** np.arange(-8, 0))) -> dict[str, Any]:
    """y^m = polynomial(raw components) with STLSQ; (m, threshold) chosen on a
    held-out 10% of the training rows, then refit on all training rows.

    m runs over 1..max_power; the library degree is max(degree, m), so a
    surface whose m-th power is a degree-m polynomial (CB2001: m = 6) is
    representable. The defaults reproduce the Hill-ladder runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(config)
    active = [j for j in range(data.X_train.shape[1]) if np.any(data.X_train[:, j] != 0)]
    names = [data.feature_names[j] for j in active]
    libraries = {d: _polynomial_library(data.X_train[:, active], d)
                 for d in {max(degree, m) for m in range(1, max_power + 1)}}
    rng = np.random.default_rng(config.train.seed)
    holdout = rng.random(len(data.X_train)) < 0.1
    started = time.perf_counter()
    best: tuple[float, int, int, float] | None = None  # (error, terms, m, threshold)
    for m in range(1, max_power + 1):
        Theta = libraries[max(degree, m)][0]
        for threshold in thresholds:
            xi = stlsq(Theta[~holdout], data.y_train[~holdout] ** m, threshold)
            prediction = _inverse_power(Theta[holdout] @ xi, m)
            error = _relative_l2(prediction, data.y_train[holdout])
            candidate = (error, int(np.count_nonzero(xi)), m, threshold)
            if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
                best = candidate
    assert best is not None
    _, _, m, threshold = best
    degree = max(degree, m)
    Theta, terms = libraries[degree]
    xi = stlsq(Theta, data.y_train ** m, threshold)
    elapsed = time.perf_counter() - started

    def predict(X: np.ndarray) -> np.ndarray:
        return _inverse_power(_polynomial_library(X[:, active], degree)[0] @ xi, m)

    monomials = ["*".join(names[i] for i in t) if t else "1" for t in terms]
    equation = " + ".join(f"{c:.12g}*{monomial}" for c, monomial in zip(xi, monomials) if c != 0)
    X_fresh, y_fresh = fresh
    result = {
        "expanded_equation": f"({equation})^(1/{m})",
        "power": m,
        "threshold": threshold,
        "active_terms": int(np.count_nonzero(xi)),
        "library_degree": degree,
        "library_size": len(terms),
        "fit_seconds": elapsed,
        **_score(predict(X_fresh), y_fresh, predict(data.X_test), data.y_test),
    }
    save_json(output_dir / "sparse_linear_evaluation.json", result)
    return result


def run_distillation(config: Config, output_dir: Path, fresh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Symbolic distillation (C): MLP with an L1 latent, PySR on the latent
    components and on the outer map; same total PySR time as one search."""
    from invariant_generator.distillation import DistillationSettings, distill

    PySRRegressor = _import_pysr_regressor()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_training_data(config)
    active = [j for j in range(data.X_train.shape[1]) if np.any(data.X_train[:, j] != 0)]
    names = [data.feature_names[j] for j in active]
    y_train = data.y_train

    def fit_pysr(X: np.ndarray, target: np.ndarray, variables: list[str], directory: Path, seconds: float) -> Any:
        search = _with_stop_condition(config, target)
        search.symbolic.timeout_hours = seconds / 3600.0
        if target is not y_train:
            # Latent components cross zero; relative weights would blow up there.
            search.symbolic.weight_mode = "none"
            search.symbolic.elementwise_loss = "loss(prediction, target) = (prediction - target)^2"
        directory.mkdir(parents=True, exist_ok=True)
        model = PySRRegressor(**pysr_regressor_options(search, output_directory=directory))
        model.fit(X, target, variable_names=variables, **_pysr_fit_kwargs(search, target))
        return model

    predict, info = distill(
        data.X_train[:, active], y_train, names,
        fit_pysr=fit_pysr, output_dir=output_dir,
        total_pysr_seconds=float(config.symbolic.timeout_hours) * 3600.0,
        settings=DistillationSettings(seed=config.train.seed),
    )
    X_fresh, y_fresh = fresh
    result = {
        **info,
        "expanded_equation": f"{info['outer_equation']} with {info['latent_equations']}",
        **_score(predict(X_fresh[:, active]), y_fresh, predict(data.X_test[:, active]), data.y_test),
    }
    save_json(output_dir / "distillation_evaluation.json", result)
    return result


def run_yield_benchmark(
    config: Config, *, results_dir: Path, arms: tuple[str, ...], resume: bool = True,
    sparse_max_power: int = 3,
) -> Path:
    """One dataset, several arms; writes <results_dir>/benchmark_summary.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    name = Path(config.data.dataset_name).name
    config = deepcopy(config)
    config.train.results_dir = results_dir
    config.train.split_dir = results_dir / "splits"
    config.adaptive.results_subdir = name
    config.symbolic.output_directory = results_dir / name / "stage3_pysr"
    summary_path = results_dir / "benchmark_summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if resume and summary_path.exists() else []
    row = rows[0] if rows else {"equation_id": name, "dataset": str(config.data.dataset_name)}
    truth = load_truth(config)
    if truth is None:
        raise ValueError(f"No ground truth for {config.data.dataset_name} (no M_mandel or criterion in its JSON).")
    fresh = fresh_points(config, *truth)
    for arm in arms:
        if arm in row and "error" not in row[arm]:
            continue
        print(f"[YIELD] {name}: {arm}", flush=True)
        try:
            if arm == "pipeline":
                row[arm] = run_pipeline(config, fresh)
            elif arm == "baseline":
                row[arm] = run_baseline(config, results_dir / name / "baseline_pysr", fresh)
            elif arm == "library_pysr":
                row[arm] = run_library_pysr(config, results_dir / name / "library_pysr", fresh)
            elif arm == "sparse_linear":
                row[arm] = run_sparse_linear(config, results_dir / name / "sparse_linear", fresh,
                                             max_power=sparse_max_power)
            elif arm == "distillation":
                row[arm] = run_distillation(config, results_dir / name / "distillation", fresh)
            else:
                raise ValueError(f"Unknown arm {arm!r}")
        except Exception as exc:
            row[arm] = {"error": repr(exc), "traceback": traceback.format_exc()}
        save_json(summary_path, [row])
    return summary_path
