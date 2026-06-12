from dataclasses import asdict, fields, is_dataclass

from invariant_generator.config import (
    Config,
    INVARIANT_NAMES,
    coerce_config_dataclasses,
    load_config,
)
from invariant_generator.model import InvariantYieldModel


def test_default_invariant_config_uses_full_candidate_pool():
    config = Config()

    assert config.invariants.selected == INVARIANT_NAMES
    assert config.invariants.enable_second_order is True
    assert config.invariants.enable_fourth_order is True


def test_default_toml_can_omit_selected_invariants():
    config = load_config("configs/default.toml")

    assert config.invariants.selected == INVARIANT_NAMES
    assert config.symbolic.maxdepth is None
    assert config.symbolic.output_subdir == "symbolic2"
    assert config.symbolic.feature_selection == "scaled_encoder_norm"
    assert config.symbolic.selected_invariants == []
    assert config.symbolic.target_transform == "identity"
    assert config.constraints.A_psd.enabled is False
    assert config.constraints.A_psd.mode == "check"
    assert config.symbolic.constraints["*"] == (8, 8)
    assert config.symbolic.nested_constraints == {
        "square": {"square": 1},
        "sqrt": {"sqrt": 1},
    }
    assert config.symbolic.weight_mode == "inverse_target_squared"


def test_checkpoint_style_shallow_restore_rebuilds_nested_constraints():
    original = Config()
    original.constraints.A_psd.enabled = True
    original.constraints.A_psd.mode = "hard"
    saved_config = asdict(original)

    restored = Config()
    for section_name, values in saved_config.items():
        if not hasattr(restored, section_name):
            continue
        section = getattr(restored, section_name)
        if not is_dataclass(section) or not isinstance(values, dict):
            continue
        known_fields = {field.name for field in fields(section)}
        for key, value in values.items():
            if key in known_fields:
                setattr(section, key, value)

    assert isinstance(restored.constraints.A_psd, dict)
    coerce_config_dataclasses(restored)

    assert restored.constraints.A_psd.enabled is True
    assert restored.constraints.A_psd.mode == "hard"

    model = InvariantYieldModel.from_config(restored)
    assert model.invariant_pool.raw_A is not None
    assert model.invariant_pool.raw_A.shape == (6, 6)
