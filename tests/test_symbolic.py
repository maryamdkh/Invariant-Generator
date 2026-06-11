import numpy as np
import pytest
import torch

from invariant_generator.config import Config
from invariant_generator.model import InvariantYieldModel
from invariant_generator.symbolic import (
    _import_pysr_regressor,
    compute_selected_invariant_features,
    select_top_invariants_from_encoder,
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
