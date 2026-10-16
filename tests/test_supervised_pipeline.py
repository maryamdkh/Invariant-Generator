from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from invariant_generator.config import Config
from invariant_generator.adaptive import (
    adaptive_metric_threshold,
    adaptive_n_bounds,
    with_selection_metric,
)
from invariant_generator.data import prepare_training_data
from invariant_generator.input_features import PairwiseFeaturePool, input_feature_names
from invariant_generator.model import InvariantYieldModel
from invariant_generator.sparsify import (
    active_term_candidates,
    candidate_support_masks,
    sparsify_encoder_from_checkpoint,
    threshold_encoder_mask,
)


def test_pairwise_features_have_stable_order_and_reject_invalid_ratios():
    pool = PairwiseFeaturePool(["a", "b"])
    x = torch.tensor([[2.0, 4.0]])
    torch.testing.assert_close(
        pool(x), torch.tensor([[2.0, 4.0, 4.0, 8.0, 16.0, 0.5, 2.0]])
    )
    with pytest.raises(ValueError, match="denominator"):
        pool(torch.tensor([[0.0, 4.0]]))
    assert PairwiseFeaturePool(["a", "b", "c"]).output_dim == 15


def test_supervised_split_is_saved_and_test_is_not_used_for_selection(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X = np.column_stack((np.linspace(0.5, 1.5, 30), np.linspace(1.0, 2.0, 30)))
    y = X[:, 0] * X[:, 1]
    with h5py.File(data_dir / "case.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    config = Config()
    config.data.kind = "supervised"
    config.data.data_dir = data_dir
    config.data.dataset_name = "case"
    config.data.dataset_key = "X"
    config.data.target_key = "y"
    config.data.feature_names = ["a", "b"]
    config.data.validation_size = 0.2
    config.data.test_size = 0.2
    config.train.split_dir = tmp_path / "splits"
    config.augmentation.enabled = False
    config.input_transform.kind = "pairwise"
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    first = prepare_training_data(config)
    second = prepare_training_data(config)
    assert first.split_path.exists()
    assert first.X_train.shape == (18, 2)
    assert first.X_selection.shape == (6, 2)
    assert first.X_test.shape == (6, 2)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    assert not np.array_equal(first.X_selection, first.X_test)
    assert len(input_feature_names(config)) == 7
    assert adaptive_n_bounds(config) == (1, 2)
    config.sparsification.adaptive_max_active_terms = True
    assert active_term_candidates(config) == [1, 2, 3, 4]
    model = InvariantYieldModel.from_config(config)
    assert model(torch.as_tensor(first.X_train[:4], dtype=torch.float32)).shape == (4,)

    with h5py.File(data_dir / "case.h5", "r+") as f:
        f["y"][0] = 999.0
    with pytest.raises(ValueError, match="does not match"):
        prepare_training_data(config)


def test_nrmse_is_scale_free_and_rejects_constant_targets():
    config = Config()
    config.adaptive.metric = "nrmse"
    config.adaptive.nrmse_threshold = 0.05
    assert adaptive_metric_threshold(config) == ("nrmse", 0.05)
    metrics = with_selection_metric({"rmse": 0.2}, np.array([1.0, 3.0]), metric="nrmse")
    assert metrics["nrmse"] == pytest.approx(0.2)
    assert with_selection_metric({"rmse": 0.2}, np.ones(3), metric="rmse") == {"rmse": 0.2}
    with pytest.raises(ValueError, match="non-constant"):
        with_selection_metric({"rmse": 0.2}, np.ones(3), metric="nrmse")


def test_support_masks_magnitude_matches_threshold_and_exhaustive_enumerates():
    S = np.array([[0.9, 0.1, 0.5, 0.0]])
    magnitude = candidate_support_masks(S, cap=2, threshold=0.05)
    assert len(magnitude) == 1
    np.testing.assert_array_equal(
        magnitude[0], threshold_encoder_mask(S, threshold=0.05, max_active_terms_per_row=2)
    )

    exhaustive = candidate_support_masks(S, cap=2, threshold=0.05, support_search="exhaustive")
    assert len(exhaustive) == 6
    assert all(mask.sum() == 2 for mask in exhaustive)
    assert len({mask.tobytes() for mask in exhaustive}) == 6
    full = candidate_support_masks(S, cap=0, threshold=0.05, support_search="exhaustive")
    assert len(full) == 1 and full[0].all()

    # Two interchangeable rows, one term each from three columns: 3 + 2 + 1 masks.
    two_rows = candidate_support_masks(
        np.zeros((2, 3)), cap=1, threshold=0.0, support_search="exhaustive",
    )
    assert len(two_rows) == 6
    with pytest.raises(ValueError, match="max_support_candidates"):
        candidate_support_masks(
            np.zeros((2, 3)), cap=1, threshold=0.0,
            support_search="exhaustive", max_candidates=5,
        )


def test_exhaustive_stage2_refits_every_support(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 1.5, size=(60, 2))
    y = np.exp(-(X[:, 0] + X[:, 1]))
    with h5py.File(data_dir / "case.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    config = Config()
    config.data.kind = "supervised"
    config.data.data_dir = data_dir
    config.data.dataset_name = "case"
    config.data.dataset_key = "X"
    config.data.feature_names = ["a", "b"]
    config.train.split_dir = tmp_path / "splits"
    config.train.results_dir = tmp_path / "results"
    config.train.device = "cpu"
    config.augmentation.enabled = False
    config.input_transform.kind = "pairwise"
    config.encoder.enabled = True
    config.encoder.output_dim = 1
    config.encoder.init = "random"
    config.adaptive.metric = "nrmse"
    config.adaptive.nrmse_threshold = 1e-12  # nothing passes: every cap is evaluated
    config.sparsification.method = "lasso"
    config.sparsification.epochs = 0
    config.sparsification.support_search = "exhaustive"
    config.sparsification.adaptive_max_active_terms = True
    config.sparsification.max_active_terms_candidates = [1, 2]
    config.sparsification.masked_refit_epochs = 3
    config.sparsification.masked_refit_schedule = "cosine"
    model = InvariantYieldModel.from_config(config)
    checkpoint = tmp_path / "stage1.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    result = sparsify_encoder_from_checkpoint(config, checkpoint_path=checkpoint)
    summary = json.loads(result.summary_path.read_text())
    assert summary["support_search"] == "exhaustive"
    by_cap = {c["max_active_terms_per_row"]: c for c in summary["candidate_results"]}
    assert len(by_cap[1]["support_candidates"]) == 7
    assert len(by_cap[2]["support_candidates"]) == 21
    ranked = [c["test_metrics"]["nrmse"] for c in by_cap[2]["support_candidates"]]
    assert ranked == sorted(ranked)
    assert by_cap[2]["test_metrics"]["nrmse"] == pytest.approx(ranked[0])
    assert summary["active_counts"] in ([1], [2])


def test_recovery_report_accepts_fitted_constants_and_rejects_wrong_structure():
    import sympy as sp

    from invariant_generator.adaptive_symbolic import recovery_report

    symbols = {"a": sp.Symbol("a"), "b": sp.Symbol("b")}
    truth = sp.exp(-(symbols["a"] + symbols["b"]) ** 2 / 2) / sp.sqrt(2 * sp.pi)
    X_fresh = np.random.default_rng(0).uniform(0.25, 1.25, size=(200, 2))
    # Fitted constants: 0.39894228 for 1/sqrt(2*pi), 0.14314426 * 1.8689515**2 for 1/2.
    good = recovery_report(
        "0.398942280401*exp(-0.143144261234*(1.8689515*a + 1.8689515*b)**2)",
        truth, symbols, X_fresh=X_fresh,
    )
    assert not good["exact_symbolic_recovery"]
    # The advisory symbolic check leaves exp(-1e-12*(a + b)**2) here; numerics decide.
    assert good["numerical_recovery"]
    tiny_offset = recovery_report(
        "0.398942280401*exp(-(a + b + 3.7e-8)**2/2)", truth, symbols, X_fresh=X_fresh,
    )
    assert tiny_offset["numerical_recovery"] and tiny_offset["symbolic_recovery"]
    # Right structure but encoder coefficients 1e-4 apart: approximate, not recovered.
    approximate = recovery_report(
        "0.398942*exp(-0.0749*(2.5831*a + 2.5826*b)**2)", truth, symbols, X_fresh=X_fresh,
    )
    assert approximate["fresh_sample_relative_l2"] < 1e-3
    assert not approximate["numerical_recovery"] and not approximate["symbolic_recovery"]
    for wrong in (
        "0.398942*exp(-0.0749*(2.5831*a + 2.4*b)**2)",
        "0.0848/b",
        "0.4*exp(-0.5*a**2 - 1.0*a*b - 0.5*b**2) + 1e-5*a",
    ):
        report = recovery_report(wrong, truth, symbols, X_fresh=X_fresh)
        assert not report["numerical_recovery"] and not report["symbolic_recovery"]


def test_symbolic_check_handles_pi_constants_and_named_theta(tmp_path):
    import sympy as sp

    from invariant_generator.adaptive_symbolic import load_known_formula, recovery_report

    G, rho, alpha, c, d = sp.symbols("G rho alpha c d", positive=True)
    symbols = {"G": G, "rho": rho, "alpha": alpha, "c": c, "d": d}
    truth = sp.sqrt(8 * sp.pi * G * rho / 3 - alpha * c**2 / d**2)
    X_fresh = np.column_stack([
        np.random.default_rng(1).uniform(1, 2, 300),
        np.random.default_rng(2).uniform(1, 2, 300),
        np.random.default_rng(3).uniform(1, 2, 300),
        np.random.default_rng(4).uniform(1, 2, 300),
        np.random.default_rng(5).uniform(1, 2, 300),
    ])
    # 8.37758 is 8*pi/3, close to 67/8; it must not be read as a fraction.
    report = recovery_report(
        "sqrt(8.3775803954499274*G*rho - alpha*c**2/d**2)", truth, symbols, X_fresh=X_fresh,
    )
    assert report["numerical_recovery"] and report["symbolic_recovery"]

    # A dataset variable named theta must not be replaced by a placeholder.
    (tmp_path / "case.json").write_text(json.dumps({
        "formula": "E_n/(1 + E_n*(1 - cos(theta))/(m*c**2))",
        "input_ranges": {"E_n": [1, 3], "m": [1, 3], "c": [1, 3], "theta": [1, 3]},
    }))
    config = Config()
    config.data.data_dir = tmp_path
    config.data.dataset_name = "case"
    config.data.feature_names = ["E_n", "m", "c", "theta"]
    known_truth, info = load_known_formula(config, fallback_X=None)
    report = recovery_report(
        "E_n*c**2/(-E_n*cos(theta)/m + E_n/m + c**2)",
        known_truth, info["symbols"], X_fresh=info["X_fresh"],
    )
    assert report["symbolic_recovery"]


def test_log_target_keeps_the_split_and_round_trips_through_stage3(tmp_path):
    from invariant_generator.symbolic import transform_symbolic_target

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 2.0, size=(40, 2))
    y = X[:, 0] ** 2 / X[:, 1]
    with h5py.File(data_dir / "case.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    config = Config()
    config.data.kind = "supervised"
    config.data.data_dir = data_dir
    config.data.dataset_name = "case"
    config.data.dataset_key = "X"
    config.data.feature_names = ["a", "b"]
    config.train.split_dir = tmp_path / "splits"
    config.augmentation.enabled = False
    config.input_transform.kind = "log"
    raw = prepare_training_data(config)
    config.data.target_transform = "log"
    logged = prepare_training_data(config)
    np.testing.assert_array_equal(raw.X_test, logged.X_test)
    np.testing.assert_allclose(logged.y_train, np.log(raw.y_train))
    np.testing.assert_allclose(
        transform_symbolic_target(logged.y_test, target_transform="exp"), raw.y_test,
    )
    with h5py.File(data_dir / "case.h5", "r+") as f:
        f["y"][0] = -1.0
    config.train.use_saved_split = False
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_training_data(config)


def test_known_formula_loads_for_dotted_names_and_eval_inputs(tmp_path):
    from invariant_generator.adaptive_symbolic import load_known_formula

    name = "lsrt_II.36.38_1_0"  # LLM-SRBench names contain dots
    X_eval = np.random.default_rng(0).uniform(1, 2, size=(50, 2))
    with h5py.File(tmp_path / f"{name}.h5", "w") as f:
        f.create_dataset("X", data=X_eval)
        f.create_dataset("y", data=X_eval[:, 0] * X_eval[:, 1])
        f.create_dataset("X_eval", data=X_eval)
    (tmp_path / f"{name}.json").write_text(json.dumps({
        "formula": "a*b", "eval_key": "X_eval", "inputs_positive": True,
    }))
    config = Config()
    config.data.data_dir = tmp_path
    config.data.dataset_name = name
    config.data.feature_names = ["a", "b"]
    known = load_known_formula(config, fallback_X=None)
    assert known is not None
    truth, info = known
    np.testing.assert_array_equal(info["X_fresh"], X_eval)
    assert info["positive"] and str(truth) == "a*b"
