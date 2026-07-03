from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from invariant_generator.benchmark import (
    FORMULA_REGISTRY,
    _make_hidden_pool,
    benchmark_root,
    classify_recovery,
    collect_benchmark_results,
    compute_hidden_homogenized_invariants,
    evaluate_benchmark_dataset_quality,
    evaluate_hidden_formula,
    generate_benchmark_dataset,
    run_benchmark_suite,
)
from invariant_generator.utils import save_json
from invariant_generator.config import Config, load_config
from invariant_generator.data import canonicalize_stress_features


def _small_benchmark_config(tmp_path: Path) -> Config:
    config = Config()
    config.train.results_dir = tmp_path / "results"
    config.train.split_dir = tmp_path / "splits"
    config.benchmark.formula_names = ["single_H2"]
    config.benchmark.n_samples = 24
    config.benchmark.validation_samples = 32
    config.benchmark.homogeneity_samples = 16
    config.benchmark.batch_size = 16
    config.benchmark.output_subdir = "bench"
    return config


def test_fixed_benchmark_formulas_are_first_order_homogeneous():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 6))
    k = 2.75
    pool = _make_hidden_pool(123)

    for formula in FORMULA_REGISTRY.values():
        base = evaluate_hidden_formula(formula, pool, X)
        scaled = evaluate_hidden_formula(formula, pool, k * X)
        np.testing.assert_allclose(scaled, k * base, rtol=1e-5, atol=1e-5)


def test_generate_benchmark_dataset_saves_surface_points(tmp_path):
    config = _small_benchmark_config(tmp_path)
    dataset = generate_benchmark_dataset(config, "single_H2")

    assert dataset.dataset_path.exists()
    assert dataset.pipeline_config_path.exists()
    assert dataset.metadata_path.exists()
    assert dataset.quality_path.exists()
    assert dataset.generation_quality["surface_value_max_abs_error"] < 1e-5

    pipeline_config = load_config(dataset.pipeline_config_path)
    assert pipeline_config.data.data_dir == dataset.dataset_path.parent
    assert pipeline_config.data.dataset_name == dataset.dataset_path.name
    assert pipeline_config.data.stress_format == "mandel_3d"
    assert pipeline_config.adaptive.results_subdir == "bench/single_H2"
    assert pipeline_config.normalization.mode == "scale_only"

    with h5py.File(dataset.dataset_path, "r") as h5:
        X_mandel = np.asarray(h5["stress"], dtype=np.float64)
    X_voigt, _ = canonicalize_stress_features(X_mandel, stress_format="mandel_3d")
    pool = _make_hidden_pool(config.benchmark.direction_seed)
    values = evaluate_hidden_formula(dataset.formula, pool, X_voigt)

    assert X_mandel.shape == (config.benchmark.n_samples, 6)
    np.testing.assert_allclose(values, np.ones_like(values), rtol=1e-5, atol=1e-6)


def test_evaluate_benchmark_dataset_quality_reports_formula_and_surface_metrics(tmp_path):
    config = _small_benchmark_config(tmp_path)
    generate_benchmark_dataset(config, "single_H2")

    quality = evaluate_benchmark_dataset_quality(config, "single_H2")

    assert quality["formula_complexity"]["active_invariant_count"] == 1
    assert quality["formula_homogeneity"]["relative_l2_error"] < 1e-5
    assert quality["dataset_surface"]["surface_value_max_abs_error"] < 1e-5
    assert quality["formula_value_random_directions"]["finite_fraction"] == 1.0
    assert (benchmark_root(config) / "single_H2" / "formula_dataset_quality.json").exists()


def test_hidden_invariant_matrix_has_expected_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 6))
    pool = _make_hidden_pool(123)
    values = compute_hidden_homogenized_invariants(pool, X, batch_size=2)

    assert values.shape == (5, 13)
    assert np.isfinite(values).all()


def test_classify_recovery_prefers_exact_then_numerical_then_predictive():
    common = {
        "exact_relative_l2": 1e-6,
        "numerical_relative_l2": 1e-3,
        "max_relative_error_threshold": 1e-2,
    }

    assert (
        classify_recovery(
            active_match=True,
            relative_l2=1e-8,
            max_relative_error=1e-8,
            predictive_passes=True,
            **common,
        )
        == "exact_symbolic_recovery"
    )
    assert (
        classify_recovery(
            active_match=False,
            relative_l2=1e-4,
            max_relative_error=1e-3,
            predictive_passes=False,
            **common,
        )
        == "numerical_equivalent_recovery"
    )
    assert (
        classify_recovery(
            active_match=False,
            relative_l2=1e-1,
            max_relative_error=1.0,
            predictive_passes=True,
            **common,
        )
        == "predictive_only_recovery"
    )


def test_benchmark_runner_uses_existing_pipeline_stages(tmp_path, monkeypatch):
    config = _small_benchmark_config(tmp_path)
    calls: list[str] = []

    def fake_run_adaptive_sweep(case_config):
        calls.append("stage1")
        path = benchmark_root(config) / "single_H2" / "fake_stage1.json"
        return SimpleNamespace(
            summary_path=path,
            selected_n=1,
            selected_checkpoint=path.with_suffix(".pt"),
        )

    def fake_sparsify_encoder_from_checkpoint(stage_config, *, checkpoint_path):
        calls.append("stage2")
        path = benchmark_root(config) / "single_H2" / "fake_sparse.json"
        return SimpleNamespace(
            summary_path=path,
            checkpoint_path=path.with_suffix(".pt"),
            mask_path=path.with_name("fake_mask.json"),
        )

    def fake_train_encoded_symbolic_from_config(symbolic_config, *, checkpoint_path, config_path):
        calls.append("stage3")
        path = benchmark_root(config) / "single_H2" / "fake_symbolic"
        return SimpleNamespace(
            output_dir=path,
            best_equation="J1",
            metrics_path=path / "metrics.json",
            formulas_path=path / "encoded_invariant_formulas.json",
        )

    def fake_evaluate_benchmark_recovery(case_config, formula_name, *, checkpoint_path):
        calls.append("evaluate")
        return {"classification": "exact_symbolic_recovery"}

    monkeypatch.setattr(
        "invariant_generator.benchmark.run_adaptive_sweep",
        fake_run_adaptive_sweep,
    )
    monkeypatch.setattr(
        "invariant_generator.benchmark.sparsify_encoder_from_checkpoint",
        fake_sparsify_encoder_from_checkpoint,
    )
    monkeypatch.setattr(
        "invariant_generator.benchmark.train_encoded_symbolic_from_config",
        fake_train_encoded_symbolic_from_config,
    )
    monkeypatch.setattr(
        "invariant_generator.benchmark.evaluate_benchmark_recovery",
        fake_evaluate_benchmark_recovery,
    )

    result = run_benchmark_suite(config, stage="all", config_path="config.toml")

    assert calls == ["stage1", "stage2", "stage3", "evaluate"]
    assert Path(result["summary_path"]).exists()
    assert result["cases"][0]["recovery"]["classification"] == "exact_symbolic_recovery"


def test_collect_benchmark_results_reads_completed_case_summaries(tmp_path):
    config = _small_benchmark_config(tmp_path)
    dataset = generate_benchmark_dataset(config, "single_H2")
    evaluate_benchmark_dataset_quality(config, "single_H2")
    save_json(
        dataset.case_dir / "recovery_metrics.json",
        {
            "classification": "numerical_equivalent_recovery",
            "best_equation": "J1",
            "active_invariant_match": True,
            "prediction_metrics": {
                "train": {"rmse": 1e-4},
                "test": {"rmse": 2e-4},
            },
            "recovery_metrics": {
                "relative_l2": 3e-4,
                "max_relative_error": 4e-4,
                "homogeneity_error": 5e-4,
            },
        },
    )

    comparison = collect_benchmark_results(config)

    assert comparison["n_cases"] == 1
    assert comparison["n_completed"] == 1
    assert comparison["classification_counts"] == {"numerical_equivalent_recovery": 1}
    assert comparison["rows"][0]["test_rmse"] == 2e-4
    assert (benchmark_root(config) / "benchmark_result_comparison.json").exists()
