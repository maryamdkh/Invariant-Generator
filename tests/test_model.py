import torch

from invariant_generator.config import Config
from invariant_generator.model import InvariantYieldModel


def test_identity_encoder_with_reduced_output_activates_leading_columns_only():
    config = Config()
    config.invariants.selected = ["I1", "I2", "I3", "I4", "I5"]
    config.invariants.enable_second_order = True
    config.invariants.enable_fourth_order = False
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    config.encoder.init = "identity"

    model = InvariantYieldModel.from_config(config)
    S = model.encoder_matrix()

    assert S is not None
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(S.detach(), expected)


def test_invariant_standardizer_applies_saved_statistics():
    config = Config()
    config.invariants.selected = ["I1", "I2"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.normalization.enabled = True

    model = InvariantYieldModel.from_config(config)
    assert model.normalizer is not None
    model.normalizer.set_statistics(
        torch.tensor([2.0, 10.0]),
        torch.tensor([2.0, 5.0]),
    )

    stress = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    raw = model.raw_invariant_features(stress)
    normalized = model.normalized_invariant_features(stress)

    torch.testing.assert_close(
        normalized,
        (raw - torch.tensor([2.0, 10.0])) / torch.tensor([2.0, 5.0]),
    )
    assert model.invariant_normalization_state() == {
        "mode": "standard",
        "mean": [2.0, 10.0],
        "std": [2.0, 5.0],
        "eps": 1e-08,
    }


def test_scale_only_normalizer_preserves_linear_scaling():
    config = Config()
    config.invariants.selected = ["I1", "I2"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = False
    config.invariants.homogenize = True
    config.normalization.enabled = True
    config.normalization.mode = "scale_only"

    model = InvariantYieldModel.from_config(config)
    assert model.normalizer is not None
    model.normalizer.set_statistics(
        torch.tensor([2.0, 10.0]),
        torch.tensor([2.0, 5.0]),
    )

    stress = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    k = 2.5

    base = model.normalized_invariant_features(stress)
    scaled = model.normalized_invariant_features(k * stress)

    torch.testing.assert_close(scaled, k * base)
    assert model.invariant_normalization_state()["mode"] == "scale_only"


def test_normalization_mode_none_disables_normalizer():
    config = Config()
    config.normalization.enabled = True
    config.normalization.mode = "none"

    model = InvariantYieldModel.from_config(config)

    assert model.normalizer is None
    assert model.invariant_normalization_state() is None
