from invariant_generator.config import Config, INVARIANT_NAMES, load_config


def test_default_invariant_config_uses_full_candidate_pool():
    config = Config()

    assert config.invariants.selected == INVARIANT_NAMES
    assert config.invariants.enable_second_order is True
    assert config.invariants.enable_fourth_order is True


def test_default_toml_can_omit_selected_invariants():
    config = load_config("configs/default.toml")

    assert config.invariants.selected == INVARIANT_NAMES
    assert config.symbolic.maxdepth is None
    assert config.symbolic.output_subdir == "symbolic"
