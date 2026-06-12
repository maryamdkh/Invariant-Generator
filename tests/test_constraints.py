import numpy as np
import pytest
import torch

from invariant_generator.config import Config, load_config
from invariant_generator.constraints import (
    fourth_order_to_mandel_matrix,
    mandel_matrix_to_fourth_order,
    psd_eigenvalues,
    stress_tensor_to_mandel,
)
from invariant_generator.diagnostics import constraint_diagnostics
from invariant_generator.losses import YieldSurfaceLoss
from invariant_generator.model import InvariantYieldModel


def test_mandel_mapping_matches_tensor_contraction_with_shear_terms():
    raw = torch.randn(6, 6)
    mandel = 0.5 * (raw + raw.T)
    A = mandel_matrix_to_fourth_order(mandel)

    sigma = torch.randn(12, 3, 3)
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    sigma_mandel = stress_tensor_to_mandel(sigma)

    tensor_q = torch.einsum("...ij,ijkl,...kl->...", sigma, A, sigma)
    mandel_q = torch.einsum("...a,ab,...b->...", sigma_mandel, mandel, sigma_mandel)

    torch.testing.assert_close(tensor_q, mandel_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fourth_order_to_mandel_matrix(A), mandel, rtol=1e-5, atol=1e-5)


def test_hard_A_psd_parameterization_guarantees_nonnegative_quadratic_form():
    config = Config()
    config.invariants.selected = ["I11"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = True
    config.constraints.A_psd.enabled = True
    config.constraints.A_psd.mode = "hard"

    model = InvariantYieldModel.from_config(config)
    A = model.invariant_pool.effective_fourth_order_tensor()
    eigvals = psd_eigenvalues(fourth_order_to_mandel_matrix(A))
    assert torch.all(eigvals >= -1e-7)

    stress = torch.randn(32, 6)
    values = model.invariant_pool(stress).squeeze(-1)
    assert torch.all(values >= -1e-6)


def test_A_psd_check_mode_reports_violation_without_hard_parameterization():
    config = Config()
    config.invariants.selected = ["I11"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = True
    config.constraints.A_psd.enabled = True
    config.constraints.A_psd.mode = "check"

    model = InvariantYieldModel.from_config(config)
    assert model.invariant_pool.raw_A is not None
    assert model.invariant_pool.raw_A.shape == (3, 3, 3, 3)

    negative_A = mandel_matrix_to_fourth_order(-torch.eye(6))
    with torch.no_grad():
        model.invariant_pool.raw_A.copy_(negative_A)

    diagnostics = constraint_diagnostics(model, config.constraints)
    assert diagnostics["A_psd"]["passed"] is False
    assert diagnostics["A_psd"]["min_eigenvalue"] < 0.0


def test_A_psd_penalty_contributes_only_for_negative_eigenvalues():
    config = Config()
    config.invariants.selected = ["I11"]
    config.invariants.enable_second_order = False
    config.invariants.enable_fourth_order = True
    config.constraints.A_psd.enabled = True
    config.constraints.A_psd.mode = "penalty"
    config.constraints.A_psd.penalty_weight = 2.0

    model = InvariantYieldModel.from_config(config)
    criterion = YieldSurfaceLoss(config.loss, config.constraints)
    prediction = torch.zeros(1)
    target = torch.zeros(1)

    with torch.no_grad():
        model.invariant_pool.raw_A.copy_(mandel_matrix_to_fourth_order(torch.eye(6)))
    positive_loss = criterion(model, prediction, target)
    torch.testing.assert_close(positive_loss.constraint, torch.zeros(()))

    with torch.no_grad():
        model.invariant_pool.raw_A.copy_(mandel_matrix_to_fourth_order(-torch.eye(6)))
    negative_loss = criterion(model, prediction, target)
    assert float(negative_loss.constraint.detach()) > 0.0


def test_unknown_constraint_config_key_is_rejected(tmp_path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
        [constraints.unknown]
        enabled = true
        """,
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match=r"constraints.*unknown"):
        load_config(config_path)
