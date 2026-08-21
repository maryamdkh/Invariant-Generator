from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import json
import re
from typing import Callable

import h5py
import numpy as np
import torch

from invariant_generator.adaptive import (
    adaptive_metric_threshold,
    adaptive_results_dir,
    adaptive_run_passes,
    adaptive_sparsification_run_id,
    config_for_adaptive_n,
    run_adaptive_sweep,
)
from invariant_generator.adaptive_symbolic import (
    compute_encoded_invariant_features,
    train_encoded_symbolic_from_config,
)
from invariant_generator.config import Config, INVARIANT_NAMES, PROJECT_ROOT, load_config
from invariant_generator.data import canonicalize_stress_features
from invariant_generator.invariants import INVARIANT_DEGREES, InvariantPool
from invariant_generator.model import InvariantYieldModel
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint
from invariant_generator.utils import resolve_device, save_json, to_jsonable


FormulaFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class BenchmarkFormula:
    name: str
    expression: str
    family: str
    difficulty: str
    active_invariants: tuple[str, ...]
    degree: float
    fn: FormulaFn


@dataclass(frozen=True, slots=True)
class GeneratedBenchmarkDataset:
    case_id: str
    formula: BenchmarkFormula
    dataset_path: Path
    pipeline_config_path: Path
    metadata_path: Path
    quality_path: Path
    case_dir: Path
    generation_quality: dict[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkResultRow:
    case_id: str
    formula_name: str
    family: str | None
    difficulty: str | None
    classification: str | None
    train_rmse: float | None
    test_rmse: float | None
    recovery_relative_l2: float | None
    recovery_max_relative_error: float | None
    homogeneity_error: float | None
    active_invariant_match: bool | None
    rejection_fraction: float | None
    surface_value_max_abs_error: float | None
    best_equation: str | None


def _h(values: np.ndarray, index: int) -> np.ndarray:
    return values[:, index - 1]


PIPELINE_CONFIG_NAME = "pipeline_config.toml"
ADAPTIVE_TEMPLATE_CONFIG = PROJECT_ROOT / "configs" / "adaptive_encoder_rotated_hill.toml"


FORMULA_REGISTRY: dict[str, BenchmarkFormula] = {
    "single_H2": BenchmarkFormula(
        name="single_H2",
        expression="H2",
        family="simple",
        difficulty="easy",
        active_invariants=("I2",),
        degree=1.0,
        fn=lambda H: _h(H, 2),
    ),
    "norm_H2_H5": BenchmarkFormula(
        name="norm_H2_H5",
        expression="sqrt((0.8*H2)^2 + (0.4*H5)^2)",
        family="norm_like",
        difficulty="medium",
        active_invariants=("I2", "I5"),
        degree=1.0,
        fn=lambda H: np.sqrt((0.8 * _h(H, 2)) ** 2 + (0.4 * _h(H, 5)) ** 2),
    ),
    "mixed_linear_norm": BenchmarkFormula(
        name="mixed_linear_norm",
        expression="sqrt((0.7*H1 - 0.2*H3)^2 + (0.5*H4 + 0.6*H8)^2)",
        family="norm_like",
        difficulty="medium_hard",
        active_invariants=("I1", "I3", "I4", "I8"),
        degree=1.0,
        fn=lambda H: np.sqrt(
            (0.7 * _h(H, 1) - 0.2 * _h(H, 3)) ** 2
            + (0.5 * _h(H, 4) + 0.6 * _h(H, 8)) ** 2
        ),
    ),
    "ratio_positive": BenchmarkFormula(
        name="ratio_positive",
        expression="sqrt((H1*H4)^2) / sqrt(H2^2 + H5^2)",
        family="mixed_homogeneous",
        difficulty="hard",
        active_invariants=("I1", "I2", "I4", "I5"),
        degree=1.0,
        fn=lambda H: np.sqrt((_h(H, 1) * _h(H, 4)) ** 2)
        / np.sqrt(_h(H, 2) ** 2 + _h(H, 5) ** 2),
    ),
}


def benchmark_formula_names(config: Config) -> list[str]:
    names = list(config.benchmark.formula_names)
    if not names:
        raise ValueError("benchmark.formula_names must contain at least one formula name.")
    unknown = [name for name in names if name not in FORMULA_REGISTRY]
    if unknown:
        available = ", ".join(sorted(FORMULA_REGISTRY))
        raise ValueError(f"Unknown benchmark formulas {unknown}. Available: {available}.")
    return names


def benchmark_root(config: Config) -> Path:
    return config.train.results_dir / config.benchmark.output_subdir


def case_id_for_formula(formula_name: str) -> str:
    return formula_name


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_value(value: object, *, key: str | None = None) -> str:
    if key == "maxdepth" and value is None:
        return "-1"
    if key == "k_values" and value is None:
        return "[]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [
            f"{_toml_key(str(name))} = {_toml_value(item)}"
            for name, item in value.items()
        ]
        return "{ " + ", ".join(parts) + " }"
    if value is None:
        raise ValueError(f"Cannot serialize None for TOML key {key!r}.")
    raise TypeError(f"Unsupported TOML value for {key!r}: {type(value).__name__}")


def _write_config_toml(config: Config, path: Path) -> Path:
    section_order = [
        "data",
        "noise",
        "train_input_noise",
        "augmentation",
        "invariants",
        "encoder",
        "normalization",
        "model",
        "loss",
        "constraints",
        "train",
        "adaptive",
        "sparsification",
        "symbolic",
    ]
    lines = [
        "# Auto-generated synthetic benchmark pipeline config.",
        "# Edit this file directly if you want to change training, adaptive,",
        "# sparsification, or PySR settings for this specific synthetic case.",
        "",
    ]
    for section_name in section_order:
        section = getattr(config, section_name)
        if section_name == "constraints":
            for constraint_name in ("A_psd", "a_psd"):
                constraint = getattr(section, constraint_name)
                lines.append(f"[constraints.{constraint_name}]")
                for field in fields(constraint):
                    value = getattr(constraint, field.name)
                    lines.append(f"{field.name} = {_toml_value(value, key=field.name)}")
                lines.append("")
            continue

        if not is_dataclass(section):
            continue
        lines.append(f"[{section_name}]")
        for field in fields(section):
            value = getattr(section, field.name)
            lines.append(f"{field.name} = {_toml_value(value, key=field.name)}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def benchmark_case_config(
    config: Config,
    dataset: GeneratedBenchmarkDataset | Path,
    formula_name: str,
) -> Config:
    pipeline_config = load_config(ADAPTIVE_TEMPLATE_CONFIG)
    pipeline_config.benchmark = deepcopy(config.benchmark)

    case_id = case_id_for_formula(formula_name)
    dataset_path = dataset.dataset_path if isinstance(dataset, GeneratedBenchmarkDataset) else Path(dataset)
    case_dir = benchmark_root(config) / case_id

    pipeline_config.data.data_dir = dataset_path.parent.resolve()
    pipeline_config.data.dataset_name = dataset_path.name
    pipeline_config.data.dataset_key = config.benchmark.dataset_key
    pipeline_config.data.stress_format = "mandel_3d"
    pipeline_config.data.shuffle = True

    pipeline_config.train.results_dir = config.train.results_dir.resolve()
    pipeline_config.train.split_dir = (case_dir / "splits").resolve()
    pipeline_config.train.use_saved_split = False
    pipeline_config.train.save_split_if_missing = True
    pipeline_config.train.run_id = str(Path(config.benchmark.output_subdir) / case_id / "base")

    pipeline_config.adaptive.results_subdir = str(Path(config.benchmark.output_subdir) / case_id)
    pipeline_config.sparsification.run_id = "stage2_sparse"
    pipeline_config.symbolic.output_subdir = "stage3_pysr"
    pipeline_config.symbolic.output_directory = (
        config.train.results_dir / config.benchmark.output_subdir / case_id / "stage3_pysr"
    ).resolve()
    pipeline_config.symbolic.target_source = "data"
    pipeline_config.symbolic.target_transform = "identity"
    pipeline_config.symbolic.feature_space = "encoded_invariants"

    pipeline_config.invariants.selected = INVARIANT_NAMES.copy()
    pipeline_config.invariants.enable_second_order = True
    pipeline_config.invariants.enable_fourth_order = True
    pipeline_config.invariants.homogenize = True
    pipeline_config.normalization.enabled = True
    pipeline_config.normalization.mode = "scale_only"
    pipeline_config.augmentation.homogeneity_degree = 1.0
    pipeline_config.augmentation.surface_target = 1.0
    return pipeline_config


def _formula_seed_index(config: Config, formula_name: str) -> int:
    configured = list(config.benchmark.formula_names)
    if formula_name in configured:
        return configured.index(formula_name)
    return sorted(FORMULA_REGISTRY).index(formula_name)


def _voigt_to_mandel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    X[:, 3:] *= np.sqrt(2.0)
    return X


def _sample_voigt_directions(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    X = rng.normal(size=(n, 6)).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-15)


def formula_complexity(formula: BenchmarkFormula) -> dict[str, object]:
    expression = formula.expression
    operator_counts = {
        "+": expression.count("+"),
        "-": expression.count("-"),
        "*": expression.count("*"),
        "/": expression.count("/"),
        "sqrt": len(re.findall(r"\bsqrt\b", expression)),
        "square": len(re.findall(r"\bsquare\b|\^2", expression)),
    }
    return {
        "expression_length": len(expression),
        "active_invariant_count": len(formula.active_invariants),
        "operator_counts": operator_counts,
        "operator_total": int(sum(operator_counts.values())),
        "has_ratio": operator_counts["/"] > 0,
        "has_sqrt": operator_counts["sqrt"] > 0,
        "declared_degree": formula.degree,
        "family": formula.family,
        "difficulty": formula.difficulty,
    }


def _make_hidden_pool(seed: int) -> InvariantPool:
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        pool = InvariantPool(
            INVARIANT_NAMES,
            enable_second_order=True,
            enable_fourth_order=True,
            homogenize=True,
            init_scale=0.05,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    pool.eval()
    return pool


@torch.no_grad()
def compute_hidden_homogenized_invariants(
    pool: InvariantPool,
    X_voigt: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    X_tensor = torch.as_tensor(np.asarray(X_voigt, dtype=np.float64), dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    outputs: list[np.ndarray] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size]
        outputs.append(pool(batch).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


def evaluate_hidden_formula(
    formula: BenchmarkFormula,
    pool: InvariantPool,
    X_voigt: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    H = compute_hidden_homogenized_invariants(pool, X_voigt, batch_size=batch_size)
    return np.asarray(formula.fn(H), dtype=np.float64).reshape(-1)


def _hidden_tensor_metadata(pool: InvariantPool) -> dict[str, object]:
    payload: dict[str, object] = {}
    if pool.raw_a is not None:
        payload["raw_a"] = pool.raw_a.detach().cpu().numpy().astype(float).tolist()
    if pool.raw_A is not None:
        payload["raw_A"] = pool.raw_A.detach().cpu().numpy().astype(float).tolist()
    return payload


def generate_benchmark_dataset(
    config: Config,
    formula_name: str,
) -> GeneratedBenchmarkDataset:
    if formula_name not in FORMULA_REGISTRY:
        raise ValueError(f"Unknown benchmark formula: {formula_name}")

    formula = FORMULA_REGISTRY[formula_name]
    case_id = case_id_for_formula(formula.name)
    root = benchmark_root(config)
    case_dir = root / case_id
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / f"{case_id}.hdf"
    pipeline_config_path = case_dir / PIPELINE_CONFIG_NAME
    metadata_path = case_dir / "benchmark_metadata.json"
    quality_path = case_dir / "generation_quality.json"

    formula_index = _formula_seed_index(config, formula_name)
    hidden_seed = int(config.benchmark.direction_seed) + 1009 * formula_index
    pool = _make_hidden_pool(hidden_seed)
    rng = np.random.default_rng(int(config.benchmark.direction_seed) + formula_index)

    target_n = int(config.benchmark.n_samples)
    if target_n <= 0:
        raise ValueError("benchmark.n_samples must be positive.")
    batch_size = int(config.benchmark.batch_size)
    min_value = float(config.benchmark.min_formula_value)
    max_rejection_fraction = float(config.benchmark.max_rejection_fraction)
    if not 0.0 <= max_rejection_fraction < 1.0:
        raise ValueError("benchmark.max_rejection_fraction must be in [0, 1).")
    max_attempts = int(np.ceil(target_n / max(1.0 - max_rejection_fraction, 1e-12)))

    accepted: list[np.ndarray] = []
    attempted = 0
    rejected_nonfinite = 0
    rejected_small_or_nonpositive = 0
    rejected_projection = 0

    while sum(part.shape[0] for part in accepted) < target_n and attempted < max_attempts:
        need = target_n - sum(part.shape[0] for part in accepted)
        draw_n = min(max(batch_size, need), max_attempts - attempted)
        directions = _sample_voigt_directions(rng, draw_n)
        values = evaluate_hidden_formula(
            formula,
            pool,
            directions,
            batch_size=batch_size,
        )
        attempted += draw_n

        finite = np.isfinite(values)
        positive = values > min_value
        valid = finite & positive
        rejected_nonfinite += int((~finite).sum())
        rejected_small_or_nonpositive += int((finite & ~positive).sum())

        if not np.any(valid):
            continue
        surface = directions[valid] / values[valid, None]
        projected_values = evaluate_hidden_formula(
            formula,
            pool,
            surface,
            batch_size=batch_size,
        )
        projected_valid = np.isfinite(projected_values) & np.isclose(
            projected_values,
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )
        rejected_projection += int((~projected_valid).sum())
        if np.any(projected_valid):
            accepted.append(surface[projected_valid])

    if not accepted:
        raise RuntimeError(f"No valid benchmark samples generated for {formula.name}.")
    X_surface_voigt = np.vstack(accepted)[:target_n]
    if X_surface_voigt.shape[0] < target_n:
        raise RuntimeError(
            f"Generated only {X_surface_voigt.shape[0]} valid samples for {formula.name}; "
            f"requested {target_n}."
        )

    X_surface_mandel = _voigt_to_mandel(X_surface_voigt)
    with h5py.File(dataset_path, "w") as h5:
        h5.create_dataset(config.benchmark.dataset_key, data=X_surface_mandel)

    final_values = evaluate_hidden_formula(
        formula,
        pool,
        X_surface_voigt,
        batch_size=batch_size,
    )
    rejection_count = int(attempted - X_surface_voigt.shape[0])
    generation_quality = {
        "n_samples": int(X_surface_voigt.shape[0]),
        "attempted": int(attempted),
        "rejected": rejection_count,
        "rejection_fraction": float(rejection_count / max(attempted, 1)),
        "rejected_nonfinite": rejected_nonfinite,
        "rejected_small_or_nonpositive": rejected_small_or_nonpositive,
        "rejected_projection": rejected_projection,
        "surface_value_mean": float(np.mean(final_values)),
        "surface_value_std": float(np.std(final_values)),
        "surface_value_max_abs_error": float(np.max(np.abs(final_values - 1.0))),
    }
    save_json(quality_path, generation_quality)

    pipeline_config = benchmark_case_config(config, dataset_path, formula.name)
    _write_config_toml(pipeline_config, pipeline_config_path)

    metadata = {
        "case_id": case_id,
        "formula_name": formula.name,
        "formula_expression": formula.expression,
        "formula_family": formula.family,
        "difficulty": formula.difficulty,
        "degree": formula.degree,
        "active_invariants": list(formula.active_invariants),
        "selected_invariants": INVARIANT_NAMES,
        "homogenize": True,
        "homogenization_map": INVARIANT_DEGREES,
        "stress_format": "mandel_3d",
        "dataset_key": config.benchmark.dataset_key,
        "dataset_path": str(dataset_path),
        "pipeline_config_path": str(pipeline_config_path),
        "pipeline_config_template": str(ADAPTIVE_TEMPLATE_CONFIG),
        "direction_seed": config.benchmark.direction_seed,
        "hidden_seed": hidden_seed,
        "generation_quality": generation_quality,
        "hidden_tensors": _hidden_tensor_metadata(pool),
    }
    save_json(metadata_path, metadata)

    return GeneratedBenchmarkDataset(
        case_id=case_id,
        formula=formula,
        dataset_path=dataset_path,
        pipeline_config_path=pipeline_config_path,
        metadata_path=metadata_path,
        quality_path=quality_path,
        case_dir=case_dir,
        generation_quality=generation_quality,
    )


def _load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_generated_dataset(config: Config, formula_name: str) -> GeneratedBenchmarkDataset:
    formula = FORMULA_REGISTRY[formula_name]
    case_id = case_id_for_formula(formula_name)
    case_dir = benchmark_root(config) / case_id
    metadata_path = case_dir / "benchmark_metadata.json"
    quality_path = case_dir / "generation_quality.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Benchmark metadata not found for {formula_name}: {metadata_path}"
        )
    metadata = _load_json(metadata_path)
    dataset_path = Path(str(metadata["dataset_path"]))
    pipeline_config_path = Path(
        str(metadata.get("pipeline_config_path", case_dir / PIPELINE_CONFIG_NAME))
    )
    quality = _load_json(quality_path) if quality_path.exists() else {}
    return GeneratedBenchmarkDataset(
        case_id=case_id,
        formula=formula,
        dataset_path=dataset_path,
        pipeline_config_path=pipeline_config_path,
        metadata_path=metadata_path,
        quality_path=quality_path,
        case_dir=case_dir,
        generation_quality=quality,
    )


def _metadata_for_dataset(dataset: GeneratedBenchmarkDataset) -> dict[str, object]:
    return _load_json(dataset.metadata_path)


def _load_voigt_dataset(dataset: GeneratedBenchmarkDataset) -> np.ndarray:
    metadata = _metadata_for_dataset(dataset)
    dataset_key = str(metadata.get("dataset_key", "stress"))
    stress_format = str(metadata.get("stress_format", "mandel_3d"))
    with h5py.File(dataset.dataset_path, "r") as h5:
        if dataset_key not in h5:
            raise KeyError(
                f"Dataset key {dataset_key!r} not found in {dataset.dataset_path}. "
                f"Available keys: {list(h5.keys())}"
            )
        X_raw = np.asarray(h5[dataset_key], dtype=np.float64)
    X_voigt, _ = canonicalize_stress_features(X_raw, stress_format=stress_format)
    return X_voigt


def _quality_metric_payload(
    values: np.ndarray,
    *,
    eps: float,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    payload = {
        "finite_fraction": float(finite.mean()) if values.size else 0.0,
        "positive_fraction": float((values[finite] > eps).mean()) if np.any(finite) else 0.0,
    }
    if np.any(finite):
        finite_values = values[finite]
        payload.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
            }
        )
    return payload


def evaluate_benchmark_dataset_quality(
    config: Config,
    formula_name: str,
) -> dict[str, object]:
    dataset = _load_generated_dataset(config, formula_name)
    formula = dataset.formula
    metadata = _metadata_for_dataset(dataset)
    hidden_seed = int(metadata.get("hidden_seed", config.benchmark.direction_seed))
    formula_index = _formula_seed_index(config, formula_name)
    pool = _make_hidden_pool(hidden_seed)
    batch_size = int(config.benchmark.batch_size)
    eps = float(config.benchmark.min_formula_value)

    X_surface = _load_voigt_dataset(dataset)
    surface_values = evaluate_hidden_formula(
        formula,
        pool,
        X_surface,
        batch_size=batch_size,
    )
    surface_error = surface_values - 1.0

    rng = np.random.default_rng(int(config.benchmark.validation_seed) + formula_index)
    n_validation = int(config.benchmark.validation_samples)
    X_random = _sample_voigt_directions(rng, n_validation)
    random_values = evaluate_hidden_formula(
        formula,
        pool,
        X_random,
        batch_size=batch_size,
    )
    k = rng.uniform(
        float(config.benchmark.homogeneity_k_min),
        float(config.benchmark.homogeneity_k_max),
        size=n_validation,
    )
    scaled_values = evaluate_hidden_formula(
        formula,
        pool,
        X_random * k[:, None],
        batch_size=batch_size,
    )
    homogeneity_finite = np.isfinite(random_values) & np.isfinite(scaled_values)
    if not np.any(homogeneity_finite):
        raise RuntimeError(f"No finite homogeneity values for {formula.name}.")
    random_h = random_values[homogeneity_finite]
    scaled_h = scaled_values[homogeneity_finite]
    k_h = k[homogeneity_finite]
    homogeneity_residual = scaled_h - k_h * random_h
    homogeneity_error = float(
        np.linalg.norm(homogeneity_residual)
        / max(np.linalg.norm(k_h * random_h), config.invariants.eps)
    )
    homogeneity_max_relative_error = float(
        np.max(
            np.abs(homogeneity_residual)
            / np.maximum(np.abs(k_h * random_h), config.invariants.eps)
        )
    )

    H_surface = compute_hidden_homogenized_invariants(
        pool,
        X_surface,
        batch_size=batch_size,
    )
    invariant_stats = {
        name: {
            "mean": float(np.mean(H_surface[:, idx])),
            "std": float(np.std(H_surface[:, idx])),
            "min": float(np.min(H_surface[:, idx])),
            "max": float(np.max(H_surface[:, idx])),
        }
        for idx, name in enumerate(INVARIANT_NAMES)
    }
    stress_norms = np.linalg.norm(X_surface, axis=1)
    quality = {
        "case_id": dataset.case_id,
        "formula_name": formula.name,
        "formula_expression": formula.expression,
        "formula_family": formula.family,
        "difficulty": formula.difficulty,
        "active_invariants": list(formula.active_invariants),
        "formula_complexity": formula_complexity(formula),
        "formula_value_random_directions": _quality_metric_payload(
            random_values,
            eps=eps,
        ),
        "formula_homogeneity": {
            "relative_l2_error": homogeneity_error,
            "max_relative_error": homogeneity_max_relative_error,
            "finite_fraction": float(homogeneity_finite.mean()),
            "k_min": float(config.benchmark.homogeneity_k_min),
            "k_max": float(config.benchmark.homogeneity_k_max),
            "n_samples": n_validation,
        },
        "dataset_surface": {
            "n_samples": int(X_surface.shape[0]),
            "n_features": int(X_surface.shape[1]),
            "surface_value_mean": float(np.mean(surface_values)),
            "surface_value_std": float(np.std(surface_values)),
            "surface_value_max_abs_error": float(np.max(np.abs(surface_error))),
            "surface_value_relative_l2_error": float(
                np.linalg.norm(surface_error)
                / max(np.linalg.norm(np.ones_like(surface_values)), config.invariants.eps)
            ),
            "finite_fraction": float(np.isfinite(X_surface).all(axis=1).mean()),
            "stress_norm_min": float(np.min(stress_norms)),
            "stress_norm_max": float(np.max(stress_norms)),
            "stress_norm_mean": float(np.mean(stress_norms)),
        },
        "invariant_statistics_on_surface": invariant_stats,
        "metadata_path": str(dataset.metadata_path),
        "dataset_path": str(dataset.dataset_path),
    }
    save_json(dataset.case_dir / "formula_dataset_quality.json", quality)
    return quality


def _safe_eval_equation(expression: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    expression = expression.strip()
    if "=" in expression:
        expression = expression.split("=", maxsplit=1)[1].strip()
    env: dict[str, object] = {
        "sqrt": np.sqrt,
        "square": np.square,
        "abs": np.abs,
        "np": np,
        **variables,
    }
    result = eval(expression, {"__builtins__": {}}, env)
    return np.asarray(result, dtype=np.float64).reshape(-1)


def _encoded_variable_indices(expression: str) -> list[int]:
    indices = sorted({int(match) for match in re.findall(r"\bJ(\d+)\b", expression)})
    return indices


def _evaluate_recovered_formula(
    model: InvariantYieldModel,
    expression: str,
    X_voigt: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    J = compute_encoded_invariant_features(
        model,
        X_voigt,
        device=device,
        batch_size=batch_size,
    )
    variables = {f"J{idx + 1}": J[:, idx] for idx in range(J.shape[1])}
    return _safe_eval_equation(expression, variables)


def _recovered_active_invariants(expression: str, formulas_path: Path) -> list[str]:
    if not formulas_path.exists():
        return []
    payload = _load_json(formulas_path)
    formulas = payload.get("formulas", [])
    if not isinstance(formulas, list):
        return []
    active: set[str] = set()
    for j_index in _encoded_variable_indices(expression):
        list_index = j_index - 1
        if list_index < 0 or list_index >= len(formulas):
            continue
        row = formulas[list_index]
        if isinstance(row, dict):
            active.update(str(name) for name in row.get("active_terms", []))
    return sorted(active)


def classify_recovery(
    *,
    active_match: bool,
    relative_l2: float,
    max_relative_error: float,
    predictive_passes: bool,
    exact_relative_l2: float,
    numerical_relative_l2: float,
    max_relative_error_threshold: float,
) -> str:
    if active_match and relative_l2 <= exact_relative_l2:
        return "exact_symbolic_recovery"
    if (
        relative_l2 <= numerical_relative_l2
        and max_relative_error <= max_relative_error_threshold
    ):
        return "numerical_equivalent_recovery"
    if predictive_passes:
        return "predictive_only_recovery"
    return "failed_recovery"


def evaluate_benchmark_recovery(
    config: Config,
    formula_name: str,
    *,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    dataset = _load_generated_dataset(config, formula_name)
    formula = dataset.formula
    formula_index = _formula_seed_index(config, formula_name)
    hidden_seed = int(config.benchmark.direction_seed) + 1009 * formula_index
    hidden_pool = _make_hidden_pool(hidden_seed)

    case_config = benchmark_case_config(config, dataset, formula_name)
    device = resolve_device(case_config.train.device)
    model = InvariantYieldModel.from_config(case_config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    output_dir = adaptive_results_dir(case_config) / case_config.symbolic.output_subdir
    best_equation_path = output_dir / "best_equation.txt"
    if not best_equation_path.exists():
        raise FileNotFoundError(f"Best PySR equation not found: {best_equation_path}")
    best_equation = best_equation_path.read_text(encoding="utf-8").strip()

    rng = np.random.default_rng(int(config.benchmark.validation_seed) + formula_index)
    X_val = _sample_voigt_directions(rng, int(config.benchmark.validation_samples))
    y_true = evaluate_hidden_formula(
        formula,
        hidden_pool,
        X_val,
        batch_size=case_config.benchmark.batch_size,
    )
    y_rec = _evaluate_recovered_formula(
        model,
        best_equation,
        X_val,
        device=device,
        batch_size=case_config.benchmark.batch_size,
    )
    finite = np.isfinite(y_true) & np.isfinite(y_rec)
    if not np.any(finite):
        raise RuntimeError(f"No finite recovery comparison values for {formula_name}.")
    y_true = y_true[finite]
    y_rec = y_rec[finite]
    diff = y_rec - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    relative_l2 = float(
        np.linalg.norm(diff) / max(np.linalg.norm(y_true), config.invariants.eps)
    )
    max_relative_error = float(
        np.max(np.abs(diff) / np.maximum(np.abs(y_true), config.invariants.eps))
    )

    hom_n = min(int(config.benchmark.homogeneity_samples), X_val.shape[0])
    X_h = X_val[:hom_n]
    y_base = _evaluate_recovered_formula(
        model,
        best_equation,
        X_h,
        device=device,
        batch_size=case_config.benchmark.batch_size,
    )
    k = rng.uniform(
        float(config.benchmark.homogeneity_k_min),
        float(config.benchmark.homogeneity_k_max),
        size=hom_n,
    )
    y_scaled = _evaluate_recovered_formula(
        model,
        best_equation,
        X_h * k[:, None],
        device=device,
        batch_size=case_config.benchmark.batch_size,
    )
    homogeneity_error = float(
        np.linalg.norm(y_scaled - k * y_base)
        / max(np.linalg.norm(k * y_base), config.invariants.eps)
    )

    formulas_path = output_dir / "encoded_invariant_formulas.json"
    recovered_active = _recovered_active_invariants(best_equation, formulas_path)
    true_active = sorted(formula.active_invariants)
    active_match = set(recovered_active) == set(true_active)

    sparse_summary_path = (
        config.train.results_dir
        / adaptive_sparsification_run_id(case_config)
        / case_config.sparsification.summary_name
    )
    predictive_passes = False
    train_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    if sparse_summary_path.exists():
        sparse_summary = _load_json(sparse_summary_path)
        train_metrics = {
            str(k): float(v)
            for k, v in dict(sparse_summary.get("train_metrics", {})).items()
        }
        test_metrics = {
            str(k): float(v)
            for k, v in dict(sparse_summary.get("test_metrics", {})).items()
        }
        metric, threshold = adaptive_metric_threshold(case_config)
        if metric in train_metrics and metric in test_metrics:
            predictive_passes = adaptive_run_passes(
                train_metrics,
                test_metrics,
                metric=metric,
                threshold=threshold,
                max_generalization_gap=case_config.adaptive.max_generalization_gap,
            )

    classification = classify_recovery(
        active_match=active_match,
        relative_l2=relative_l2,
        max_relative_error=max_relative_error,
        predictive_passes=predictive_passes,
        exact_relative_l2=float(config.benchmark.exact_relative_l2),
        numerical_relative_l2=float(config.benchmark.numerical_relative_l2),
        max_relative_error_threshold=float(config.benchmark.max_relative_error),
    )
    payload = {
        "case_id": dataset.case_id,
        "formula_name": formula.name,
        "formula_expression": formula.expression,
        "best_equation": best_equation,
        "classification": classification,
        "true_active_invariants": true_active,
        "recovered_active_invariants": recovered_active,
        "active_invariant_match": active_match,
        "recovery_metrics": {
            "rmse": rmse,
            "mae": mae,
            "relative_l2": relative_l2,
            "max_relative_error": max_relative_error,
            "finite_fraction": float(finite.mean()),
            "homogeneity_error": homogeneity_error,
        },
        "prediction_metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "passes": predictive_passes,
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "best_equation": str(best_equation_path),
            "encoded_formulas": str(formulas_path),
            "sparse_summary": str(sparse_summary_path),
        },
    }
    save_json(dataset.case_dir / "recovery_metrics.json", payload)
    return payload


def run_benchmark_case(
    config: Config,
    formula_name: str,
    *,
    stage: str = "all",
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if stage not in {"generate", "quality", "pipeline", "evaluate", "all"}:
        raise ValueError(
            "stage must be 'generate', 'quality', 'pipeline', 'evaluate', or 'all'."
        )

    dataset: GeneratedBenchmarkDataset
    if stage in {"generate", "all"}:
        dataset = generate_benchmark_dataset(config, formula_name)
    else:
        dataset = _load_generated_dataset(config, formula_name)

    case_config = benchmark_case_config(config, dataset, formula_name)
    summary: dict[str, object] = {
        "case_id": dataset.case_id,
        "formula_name": formula_name,
        "dataset": str(dataset.dataset_path),
        "metadata": str(dataset.metadata_path),
        "generation_quality": dataset.generation_quality,
    }

    if stage in {"quality", "all"}:
        summary["formula_dataset_quality"] = evaluate_benchmark_dataset_quality(
            config,
            formula_name,
        )

    sparse_checkpoint: Path | None = None
    if stage in {"pipeline", "all"}:
        sweep = run_adaptive_sweep(case_config)
        summary["stage1"] = {
            "summary_path": str(sweep.summary_path),
            "selected_n": sweep.selected_n,
            "selected_checkpoint": None
            if sweep.selected_checkpoint is None
            else str(sweep.selected_checkpoint),
        }
        if sweep.selected_n is None or sweep.selected_checkpoint is None:
            summary["error"] = "adaptive sweep did not select n"
            save_json(dataset.case_dir / "case_summary.json", summary)
            return summary

        stage_config = config_for_adaptive_n(case_config, sweep.selected_n)
        sparse = sparsify_encoder_from_checkpoint(
            stage_config,
            checkpoint_path=sweep.selected_checkpoint,
        )
        sparse_checkpoint = sparse.checkpoint_path
        summary["stage2"] = {
            "summary_path": str(sparse.summary_path),
            "checkpoint": str(sparse.checkpoint_path),
            "mask_path": str(sparse.mask_path),
        }

        symbolic_config = config_for_adaptive_n(case_config, sweep.selected_n)
        symbolic_config.train.run_id = adaptive_sparsification_run_id(symbolic_config)
        symbolic = train_encoded_symbolic_from_config(
            symbolic_config,
            checkpoint_path=sparse_checkpoint,
            config_path=config_path,
        )
        summary["stage3"] = {
            "output_dir": str(symbolic.output_dir),
            "best_equation": symbolic.best_equation,
            "metrics_path": str(symbolic.metrics_path),
            "formulas_path": str(symbolic.formulas_path),
        }

    if stage in {"evaluate", "all"}:
        if sparse_checkpoint is None:
            sparse_checkpoint = (
                config.train.results_dir
                / adaptive_sparsification_run_id(case_config)
                / "checkpoint_best.pt"
            )
        recovery = evaluate_benchmark_recovery(
            config,
            formula_name,
            checkpoint_path=sparse_checkpoint,
        )
        summary["recovery"] = recovery

    save_json(dataset.case_dir / "case_summary.json", summary)
    return summary


def run_benchmark_suite(
    config: Config,
    *,
    stage: str = "all",
    case: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    names = [case] if case is not None else benchmark_formula_names(config)
    for name in names:
        if name not in FORMULA_REGISTRY:
            raise ValueError(f"Unknown benchmark formula: {name}")

    root = benchmark_root(config)
    root.mkdir(parents=True, exist_ok=True)
    case_summaries = [
        run_benchmark_case(config, name, stage=stage, config_path=config_path)
        for name in names
    ]
    aggregate = {
        "suite": config.benchmark.suite,
        "stage": stage,
        "cases": case_summaries,
    }
    summary_path = save_json(root / "benchmark_summary.json", aggregate)
    return {
        "summary_path": str(summary_path),
        "root": str(root),
        "suite": config.benchmark.suite,
        "stage": stage,
        "cases": to_jsonable(case_summaries),
    }


def _nested_float(payload: dict[str, object], keys: tuple[str, ...]) -> float | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _nested_bool(payload: dict[str, object], keys: tuple[str, ...]) -> bool | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return bool(current)


def collect_benchmark_results(config: Config) -> dict[str, object]:
    root = benchmark_root(config)
    rows: list[BenchmarkResultRow] = []
    for formula_name in benchmark_formula_names(config):
        formula = FORMULA_REGISTRY[formula_name]
        case_dir = root / case_id_for_formula(formula_name)
        recovery_path = case_dir / "recovery_metrics.json"
        generation_quality_path = case_dir / "generation_quality.json"
        quality_path = case_dir / "formula_dataset_quality.json"
        recovery = _load_json(recovery_path) if recovery_path.exists() else {}
        generation_quality = (
            _load_json(generation_quality_path) if generation_quality_path.exists() else {}
        )
        quality = _load_json(quality_path) if quality_path.exists() else {}

        row = BenchmarkResultRow(
            case_id=case_id_for_formula(formula_name),
            formula_name=formula_name,
            family=formula.family,
            difficulty=formula.difficulty,
            classification=(
                str(recovery["classification"]) if "classification" in recovery else None
            ),
            train_rmse=_nested_float(recovery, ("prediction_metrics", "train", "rmse")),
            test_rmse=_nested_float(recovery, ("prediction_metrics", "test", "rmse")),
            recovery_relative_l2=_nested_float(
                recovery,
                ("recovery_metrics", "relative_l2"),
            ),
            recovery_max_relative_error=_nested_float(
                recovery,
                ("recovery_metrics", "max_relative_error"),
            ),
            homogeneity_error=_nested_float(
                recovery,
                ("recovery_metrics", "homogeneity_error"),
            ),
            active_invariant_match=_nested_bool(recovery, ("active_invariant_match",)),
            rejection_fraction=_nested_float(generation_quality, ("rejection_fraction",)),
            surface_value_max_abs_error=(
                _nested_float(quality, ("dataset_surface", "surface_value_max_abs_error"))
                or _nested_float(generation_quality, ("surface_value_max_abs_error",))
            ),
            best_equation=(
                str(recovery["best_equation"]) if "best_equation" in recovery else None
            ),
        )
        rows.append(row)

    rows_payload = [to_jsonable(row) for row in rows]
    classified = [row for row in rows if row.classification is not None]
    classification_counts: dict[str, int] = {}
    for row in classified:
        assert row.classification is not None
        classification_counts[row.classification] = (
            classification_counts.get(row.classification, 0) + 1
        )
    aggregate = {
        "suite": config.benchmark.suite,
        "root": str(root),
        "n_cases": len(rows),
        "n_completed": len(classified),
        "classification_counts": classification_counts,
        "rows": rows_payload,
    }
    summary_path = save_json(root / "benchmark_result_comparison.json", aggregate)
    aggregate["summary_path"] = str(summary_path)
    return aggregate
