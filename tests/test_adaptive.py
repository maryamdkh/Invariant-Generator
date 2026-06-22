from pathlib import Path

import numpy as np
import torch

from invariant_generator.adaptive import (
    adaptive_metric_threshold,
    adaptive_run_passes,
    run_adaptive_sweep,
)
from invariant_generator.adaptive_symbolic import compute_encoded_invariant_features
from invariant_generator.config import Config, load_config
from invariant_generator.formulas import encoder_formula_report, encoder_to_raw_coefficients
from invariant_generator.model import InvariantYieldModel
from invariant_generator.sparsify import apply_encoder_mask, threshold_encoder_mask
from invariant_generator.train import TrainResult


def test_adaptive_stop_logic_supports_mse_and_rmse():
    config = Config()
    config.adaptive.metric = "mse"
    config.adaptive.mse_threshold = 1e-4
    assert adaptive_metric_threshold(config) == ("mse", 1e-4)
    assert adaptive_run_passes(
        {"mse": 9e-5},
        {"mse": 1e-4},
        metric="mse",
        threshold=1e-4,
    )
    assert not adaptive_run_passes(
        {"mse": 9e-5},
        {"mse": 2e-4},
        metric="mse",
        threshold=1e-4,
    )

    config.adaptive.metric = "rmse"
    config.adaptive.rmse_threshold = 1e-3
    assert adaptive_metric_threshold(config) == ("rmse", 1e-3)
    assert not adaptive_run_passes(
        {"rmse": 5e-4},
        {"rmse": 9e-4},
        metric="rmse",
        threshold=1e-3,
        max_generalization_gap=1e-4,
    )

def test_stage1_selects_smallest_successful_n(tmp_path, monkeypatch):
    config = Config()
    config.invariants.selected = ["I1", "I2", "I3"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.train.results_dir = tmp_path / "results"
    config.adaptive.run_id_prefix = "adaptive_test"
    config.adaptive.n_min = 1
    config.adaptive.n_max = 3
    config.adaptive.metric = "rmse"
    config.adaptive.rmse_threshold = 0.2

    def fake_train_from_config(run_config):
        exp = run_config.experiment_dir
        exp.mkdir(parents=True, exist_ok=True)
        checkpoint = exp / "checkpoint_best.pt"
        checkpoint.write_text("fake", encoding="utf-8")
        return TrainResult(
            experiment_dir=exp,
            best_checkpoint=checkpoint,
            recovery_checkpoint=exp / "checkpoint_latest.pt",
            history_path=exp / "history.json",
            best_epoch=1,
            best_test_mse=0.0,
        )

    def fake_evaluate(run_config, _checkpoint):
        n = run_config.encoder.output_dim
        value = {1: 0.5, 2: 0.1, 3: 0.05}[n]
        return {"rmse": value, "mse": value**2}, {"rmse": value, "mse": value**2}

    monkeypatch.setattr("invariant_generator.adaptive.train_from_config", fake_train_from_config)
    monkeypatch.setattr(
        "invariant_generator.adaptive.evaluate_checkpoint_on_train_and_test",
        fake_evaluate,
    )

    result = run_adaptive_sweep(config)

    assert result.selected_n == 2
    assert result.selected_checkpoint == tmp_path / "results" / "adaptive_test_n02" / "checkpoint_best.pt"
    assert [run.n for run in result.runs] == [1, 2]
    assert result.summary_path.exists()


def test_forward_stage1_patience_requires_consecutive_passing_dimensions(
    tmp_path,
    monkeypatch,
):
    config = Config()
    config.invariants.selected = ["I1", "I2", "I3", "I4", "I5"]
    config.invariants.enable_second_order = True
    config.invariants.enable_fourth_order = False
    config.train.results_dir = tmp_path / "results"
    config.adaptive.run_id_prefix = "forward_patience_test"
    config.adaptive.metric = "rmse"
    config.adaptive.n_min = 1
    config.adaptive.n_max = 5
    config.adaptive.rmse_threshold = 0.2
    config.adaptive.patience = 1

    def fake_train_from_config(run_config):
        exp = run_config.experiment_dir
        exp.mkdir(parents=True, exist_ok=True)
        checkpoint = exp / "checkpoint_best.pt"
        checkpoint.write_text("fake", encoding="utf-8")
        return TrainResult(
            experiment_dir=exp,
            best_checkpoint=checkpoint,
            recovery_checkpoint=exp / "checkpoint_latest.pt",
            history_path=exp / "history.json",
            best_epoch=1,
            best_test_mse=0.0,
        )

    def fake_evaluate(run_config, _checkpoint):
        n = run_config.encoder.output_dim
        value = {1: 0.4, 2: 0.1, 3: 0.3, 4: 0.15, 5: 0.12}[n]
        return {"rmse": value, "mse": value**2}, {"rmse": value, "mse": value**2}

    monkeypatch.setattr("invariant_generator.adaptive.train_from_config", fake_train_from_config)
    monkeypatch.setattr(
        "invariant_generator.adaptive.evaluate_checkpoint_on_train_and_test",
        fake_evaluate,
    )

    result = run_adaptive_sweep(config)

    assert [run.n for run in result.runs] == [1, 2, 3, 4, 5]
    assert [run.selected for run in result.runs] == [False, False, False, True, False]
    assert result.selected_n == 4
    assert result.selected_checkpoint == (
        tmp_path / "results" / "forward_patience_test_n04" / "checkpoint_best.pt"
    )


def test_lasso_mask_bookkeeping_and_fixed_mask_zero_preservation():
    S = np.array([[0.0, 1e-4, 0.0], [0.5, 0.0, -0.2]], dtype=float)
    mask = threshold_encoder_mask(S, threshold=1e-3)
    np.testing.assert_array_equal(mask, [[0, 1, 0], [1, 0, 1]])

    config = Config()
    config.invariants.selected = ["I1", "I2", "I3"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    model = InvariantYieldModel.from_config(config)
    with torch.no_grad():
        assert model.encoder is not None
        model.encoder.raw_weight.fill_(1.0)
    apply_encoder_mask(model, mask)
    np.testing.assert_allclose(model.encoder_matrix().detach().numpy(), mask.astype(float))


def test_gated_encoder_effective_weights_and_thresholded_masks():
    config = Config()
    config.invariants.selected = ["I1", "I2"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 1
    model = InvariantYieldModel.from_config(config)
    assert model.encoder is not None
    with torch.no_grad():
        model.encoder.raw_weight.copy_(torch.tensor([[2.0, 4.0]]))
    model.encoder.enable_gates(init_probability=0.5)

    np.testing.assert_allclose(model.encoder_gates().detach().numpy(), [[0.5, 0.5]])
    np.testing.assert_allclose(model.encoder_matrix().detach().numpy(), [[1.0, 2.0]])
    mask = threshold_encoder_mask(model.encoder_matrix(), threshold=1.5)
    np.testing.assert_array_equal(mask, [[0, 1]])


def test_encoded_pysr_features_use_model_invariant_features_not_raw_columns():
    config = Config()
    config.invariants.selected = ["I1", "I2"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 1
    model = InvariantYieldModel.from_config(config)
    assert model.encoder is not None
    with torch.no_grad():
        model.encoder.raw_weight.copy_(torch.tensor([[0.0, 10.0]]))

    X = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    encoded = compute_encoded_invariant_features(
        model,
        X,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        expected = model.invariant_features(torch.as_tensor(X, dtype=torch.float32)).numpy()
        raw = model.raw_invariant_features(torch.as_tensor(X, dtype=torch.float32)).numpy()

    np.testing.assert_allclose(encoded, expected)
    assert encoded.shape == (1, 1)
    assert not np.allclose(encoded[:, 0], raw[:, 0])


def test_formula_rendering_with_and_without_normalization():
    S = np.array([[2.0, 4.0]], dtype=float)
    intercept, beta = encoder_to_raw_coefficients(
        S,
        mean=np.array([1.0, 10.0]),
        std=np.array([2.0, 5.0]),
    )
    np.testing.assert_allclose(intercept, [-9.0])
    np.testing.assert_allclose(beta, [[1.0, 0.8]])

    config = Config()
    config.invariants.selected = ["I1", "I2"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 1
    config.normalization.enabled = True
    model = InvariantYieldModel.from_config(config)
    assert model.encoder is not None
    assert model.normalizer is not None
    with torch.no_grad():
        model.encoder.raw_weight.copy_(torch.tensor(S, dtype=torch.float32))
        model.normalizer.set_statistics(
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 5.0]),
        )

    report = encoder_formula_report(model, ["I1", "I2"], threshold=1e-12)
    formula = report["formulas"][0]
    assert formula["normalized_formula"] == "J1 = 2*z_I1 + 4*z_I2"
    assert formula["raw_formula"] == "J1 = -9 + I1 + 0.8*I2"

    no_norm_intercept, no_norm_beta = encoder_to_raw_coefficients(S)
    np.testing.assert_allclose(no_norm_intercept, [0.0])
    np.testing.assert_allclose(no_norm_beta, S)


def test_adaptive_config_loads_new_sections():
    config = load_config(Path("configs") / "adaptive_encoder_rotated_hill.toml")
    assert config.adaptive.metric == "rmse"
    assert config.adaptive.patience == 0
    assert config.sparsification.method in {"lasso", "gated"}
    assert config.symbolic.feature_space == "encoded_invariants"
