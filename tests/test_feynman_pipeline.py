from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import sympy as sp
import torch

from invariant_generator.adaptive_symbolic import _expanded_equation
from invariant_generator.config import Config
from invariant_generator.feynman import log_reducibility, parse_pmlb_metadata
from invariant_generator.input_features import LogFeaturePool, input_feature_names
from invariant_generator.model import InvariantYieldModel
from invariant_generator.sparsify import (
    canonical_encoder_rows,
    snap_encoder_rows,
    sparsify_encoder_from_checkpoint,
)


def _log_config(tmp_path, names, *, output_dim=1):
    config = Config()
    config.data.kind = "supervised"
    config.data.data_dir = tmp_path / "data"
    config.data.dataset_name = "case"
    config.data.dataset_key = "X"
    config.data.feature_names = list(names)
    config.train.split_dir = tmp_path / "splits"
    config.train.results_dir = tmp_path / "results"
    config.train.device = "cpu"
    config.augmentation.enabled = False
    config.input_transform.kind = "log"
    config.encoder.enabled = True
    config.encoder.output_dim = output_dim
    config.encoder.init = "random"
    config.normalization.enabled = True
    config.normalization.mode = "standard"
    config.model.output_activation = "none"
    config.model.standardize_target = True
    config.adaptive.metric = "nrmse"
    return config


def test_log_features_and_target_scaler_are_opt_in():
    pool = LogFeaturePool(["a", "b"])
    torch.testing.assert_close(pool(torch.tensor([[1.0, np.e]])), torch.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive"):
        pool(torch.tensor([[0.0, 1.0]]))

    default_keys = set(InvariantYieldModel.from_config(Config()).state_dict())
    assert not any(key.startswith("target_scaler") for key in default_keys)

    config = Config()
    config.input_transform.kind = "log"
    config.data.feature_names = ["a", "b"]
    config.encoder.enabled = True
    config.encoder.output_dim = 1
    config.model.standardize_target = True
    config.model.output_activation = "none"
    assert input_feature_names(config) == ["log(a)", "log(b)"]
    model = InvariantYieldModel.from_config(config)
    model.target_scaler.set_statistics(10.0, 2.0)
    x = torch.tensor([[1.5, 2.5]])
    expected = model.regressor(model.invariant_features(x)) * 2.0 + 10.0
    torch.testing.assert_close(model(x), expected)


def test_canonical_rows_and_snapping_recover_integer_exponents():
    truth = np.array([-2.0, -2.0, 1.0, 2.0, 2.0, 4.0, -1.0])
    learned = 0.759 * truth + np.random.default_rng(0).normal(0, 1e-3, truth.shape)
    snapped, residual = snap_encoder_rows(canonical_encoder_rows(learned[None, :]), denominator=1)
    assert residual < 0.01
    ratio = snapped[0] / truth
    np.testing.assert_allclose(ratio, ratio[0])

    # Two groups mixed by an arbitrary invertible matrix: the row space survives.
    groups = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, -2.0]])
    mixed = np.array([[0.7, 1.3], [-2.1, 0.4]]) @ groups
    snapped, _ = snap_encoder_rows(canonical_encoder_rows(mixed), denominator=1)
    assert np.linalg.matrix_rank(np.vstack([snapped, groups])) == 2
    halves, _ = snap_encoder_rows(np.array([[1.0, 0.5, -1.5]]), denominator=2)
    np.testing.assert_allclose(np.abs(halves), [[1.0, 0.5, 1.5]])


def test_snap_stage_freezes_exact_exponents(tmp_path):
    (tmp_path / "data").mkdir()
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 3.0, size=(80, 3))
    y = np.exp(-X[:, 0] * X[:, 1] ** 2 / X[:, 2])
    with h5py.File(tmp_path / "data" / "case.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    config = _log_config(tmp_path, ["a", "b", "c"])
    config.adaptive.nrmse_threshold = 10.0  # accept the first snapped candidate
    config.sparsification.method = "snap"
    config.sparsification.masked_refit_epochs = 3
    model = InvariantYieldModel.from_config(config)
    log_std = np.log(X).std(axis=0)
    with torch.no_grad():
        model.normalizer.std.copy_(torch.as_tensor(log_std, dtype=torch.float32))
        noisy = np.array([1.0, 2.0, -1.0]) * 0.8 + 1e-3
        model.encoder.raw_weight.copy_(torch.as_tensor(noisy * log_std, dtype=torch.float32)[None])
    checkpoint = tmp_path / "stage1.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    result = sparsify_encoder_from_checkpoint(config, checkpoint_path=checkpoint)
    summary = json.loads(result.summary_path.read_text())
    assert summary["snapped"] and summary["selected_denominator"] == 1
    exponents = [float(sp.Rational(v)) for v in summary["exact_raw_coefficients"][0]]
    assert exponents in ([1.0, 2.0, -1.0], [-1.0, -2.0, 1.0], [0.5, 1.0, -0.5], [-0.5, -1.0, 0.5])
    saved = torch.load(result.checkpoint_path, weights_only=False)
    assert saved["exact_raw_coefficients"] == summary["exact_raw_coefficients"]


def test_expanded_equation_turns_log_groups_into_monomials(tmp_path):
    config = _log_config(tmp_path, ["a", "b", "c"])
    model = InvariantYieldModel.from_config(config)

    class FakePySR:
        def sympy(self):
            return sp.exp(-sp.Symbol("J1"))

    expanded = _expanded_equation(
        model, FakePySR(), input_feature_names(config),
        exact_coefficients=[["1", "2", "-1"]], transform="exp", positive=True,
    )
    a, b, c = sp.symbols("a b c", positive=True)
    candidate = sp.sympify(expanded, locals={"a": a, "b": b, "c": c})
    assert sp.simplify(candidate - sp.exp(-a * b**2 / c)) == 0


def test_feynman_metadata_parser_and_reducibility(tmp_path):
    path = tmp_path / "feynman_I_40_1.yaml"
    path.write_text(
        "dataset: feynman_I_40_1\n"
        "description: |\n"
        "    A synthetic physics model.\n"
        "    \n"
        "    n = n_0*exp(-m*g*x/(kb*T))\n"
        "    \n"
        "    n_0 in [1.0,5.0]\n"
        "    m in [1.0,5.0]\n"
        "    x in [1.0,5.0]\n"
        "    T in [1.0,5.0]\n"
        "    g in [1.0,5.0]\n"
        "    kb in [1.0,5.0]\n"
        "source: test\n",
        encoding="utf-8",
    )
    problem = parse_pmlb_metadata(path)
    assert problem.target == "n"
    assert problem.variables == ["n_0", "m", "x", "T", "g", "kb"]
    X, y = problem.sample(50, seed=0)
    np.testing.assert_allclose(y, X[:, 0] * np.exp(-X[:, 1] * X[:, 4] * X[:, 2] / (X[:, 5] * X[:, 3])))
    assert log_reducibility(problem) == 2


def test_sparsest_basis_keeps_row_space_and_simplifies_groups():
    from invariant_generator.sparsify import sparsest_integer_basis

    # Echelon rows learned for x1*y1 + x2*y2 + x3*y3 (inputs x1, x2, x3, y1, y2, y3).
    echelon = np.array([
        [1 / 3, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [-1 / 3, 1, 0, 0, 0, 0],
    ])
    basis = sparsest_integer_basis(echelon)
    assert np.linalg.matrix_rank(np.vstack([basis, echelon])) == 4
    rows = {tuple(int(v) for v in row) for row in basis}
    assert {(1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 0), (0, 0, 1, 0, 0, 1)} <= rows


def test_reference_gradient_spectrum_finds_the_active_direction():
    from invariant_generator.adaptive import reference_gradient_spectrum

    config = Config()
    config.input_transform.kind = "log"
    config.data.feature_names = ["a", "b", "c"]
    config.encoder.enabled = False
    model = InvariantYieldModel.from_config(config)
    direction = torch.tensor([1.0, -2.0, 0.5])

    class SingleIndex(torch.nn.Module):
        def forward(self, features):
            return torch.tanh(features @ direction)

    model.regressor = SingleIndex()
    X = np.random.default_rng(0).uniform(0.5, 3.0, size=(200, 3))
    spectrum, Vt = reference_gradient_spectrum(model, X, device=torch.device("cpu"))
    assert spectrum[1] < 1e-6
    cosine = abs(Vt[0] @ direction.numpy()) / np.linalg.norm(direction.numpy())
    assert cosine == pytest.approx(1.0, abs=1e-6)
