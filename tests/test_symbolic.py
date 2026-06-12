import json

import numpy as np
import pytest
import torch
from unittest.mock import patch

from invariant_generator.config import Config
from invariant_generator.data import PreparedData
from invariant_generator.model import InvariantYieldModel
from invariant_generator.symbolic import (
    _import_pysr_regressor,
    _pysr_fit_kwargs,
    compute_selected_invariant_features,
    pysr_sample_weights,
    select_top_invariants_from_encoder,
    train_symbolic_from_config,
)


def test_select_top_invariants_by_encoder_column_norm_with_stable_ties():
    S = torch.tensor(
        [
            [3.0, 2.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    selection = select_top_invariants_from_encoder(
        S,
        ["I1", "I2", "I3", "I4"],
        top_k=3,
    )

    assert selection.names == ["I1", "I2", "I3"]
    assert selection.indices == [0, 1, 2]
    np.testing.assert_allclose(selection.scores, [3.0, 2.0, 2.0])


def test_scaled_encoder_norm_can_change_invariant_ranking():
    S = torch.tensor([[2.0, 1.0]])
    raw = select_top_invariants_from_encoder(
        S,
        ["I1", "I2"],
        top_k=1,
        feature_selection="encoder_norm",
        feature_stds=[0.1, 10.0],
    )
    scaled = select_top_invariants_from_encoder(
        S,
        ["I1", "I2"],
        top_k=1,
        feature_selection="scaled_encoder_norm",
        feature_stds=[0.1, 10.0],
    )

    assert raw.names == ["I1"]
    assert scaled.names == ["I2"]
    np.testing.assert_allclose(scaled.raw_scores, [1.0])
    np.testing.assert_allclose(scaled.scaled_scores, [10.0])


def test_compute_selected_invariant_features_uses_pre_encoder_columns():
    config = Config()
    config.invariants.selected = ["I1", "I2", "I3"]
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    model = InvariantYieldModel.from_config(config)

    with torch.no_grad():
        assert model.encoder is not None
        model.encoder.weight.copy_(
            torch.tensor(
                [
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            )
        )

    X = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    selected = compute_selected_invariant_features(
        model,
        X,
        [0, 2],
        device=torch.device("cpu"),
        batch_size=1,
    )

    with torch.no_grad():
        all_invariants = model.invariant_pool(torch.as_tensor(X, dtype=torch.float32))
    np.testing.assert_allclose(selected, all_invariants[:, [0, 2]].numpy())
    assert selected.shape == (2, 2)


def test_missing_pysr_dependency_message_is_actionable(monkeypatch):
    def _raise_import_error(_name):
        raise ImportError("missing")

    monkeypatch.setattr("invariant_generator.symbolic.import_module", _raise_import_error)

    with pytest.raises(ImportError, match="uv sync --extra symbolic"):
        _import_pysr_regressor()


def test_relative_pysr_weights_follow_inverse_target_squared():
    weights = pysr_sample_weights(
        np.array([0.5, 1.0, 2.0], dtype=np.float64),
        weight_mode="inverse_target_squared",
    )

    np.testing.assert_allclose(weights, [4.0, 1.0, 0.25])


def test_pysr_fit_kwargs_passes_weights_only_for_weighted_loss():
    config = Config()
    target = np.array([0.5, 1.0, 2.0], dtype=np.float64)

    weighted = _pysr_fit_kwargs(config, target)
    assert set(weighted) == {"weights"}
    np.testing.assert_allclose(weighted["weights"], [4.0, 1.0, 0.25])

    config.symbolic.elementwise_loss = "loss(prediction, target) = (prediction - target)^2"
    assert _pysr_fit_kwargs(config, target) == {}


def test_train_symbolic_logs_selected_invariants_before_fit(tmp_path, capsys):
    class FakePySRRegressor:
        def __init__(self, **_kwargs):
            self.equations_ = []

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, X):
            return np.ones(X.shape[0], dtype=np.float64)

        def get_best(self):
            return {"equation": "1.0"}

    config = Config()
    config.invariants.selected = ["I1", "I2", "I3"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    config.symbolic.top_k = 2
    config.symbolic.feature_selection = "encoder_norm"
    config.train.results_dir = tmp_path / "results"
    config.train.run_id = "run"
    config.train.device = "cpu"

    model = InvariantYieldModel.from_config(config)
    with torch.no_grad():
        assert model.encoder is not None
        model.encoder.weight.copy_(
            torch.tensor(
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            )
        )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    prepared = PreparedData(
        X_train=np.ones((2, 6), dtype=np.float64),
        y_train=np.ones(2, dtype=np.float64),
        X_test=np.ones((1, 6), dtype=np.float64),
        y_test=np.ones(1, dtype=np.float64),
        feature_names=["s11", "s22", "s33", "s23", "s13", "s12"],
        split_path=tmp_path / "split.npz",
    )

    with (
        patch("invariant_generator.symbolic._import_pysr_regressor", return_value=FakePySRRegressor),
        patch("invariant_generator.symbolic.prepare_training_data", return_value=prepared),
    ):
        result = train_symbolic_from_config(config, checkpoint_path=checkpoint_path)

    captured = capsys.readouterr()
    assert "PySR invariant source: encoder_norm" in captured.out
    assert "I2(score=3" in captured.out
    assert "I3(score=2" in captured.out
    assert result.config_snapshot_path == result.output_dir / "config.json"
    snapshot = json.loads(result.config_snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["symbolic"]["top_k"] == 2
    assert snapshot["symbolic"]["feature_selection"] == "encoder_norm"
    assert snapshot["train"]["run_id"] == "run"
