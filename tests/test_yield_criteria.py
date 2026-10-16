from __future__ import annotations

import numpy as np
import pytest

from invariant_generator.yield_criteria import CB2001, CRITERIA, Yld2004_18p

KEYS = ["12", "13", "21", "23", "31", "32", "44", "55", "66"]


def _deviator_tensor(X):
    s = X.copy()
    s[:, :3] -= X[:, :3].mean(axis=1, keepdims=True)
    xx, yy, zz, yz, zx, xy = s.T
    return np.stack([
        np.stack([xx, xy, zx], -1), np.stack([xy, yy, yz], -1), np.stack([zx, yz, zz], -1),
    ], -2)


def _directional_yield(criterion, degrees):
    ratios = []
    for angle in np.radians(degrees):
        c, s = np.cos(angle), np.sin(angle)
        ratios.append(1.0 / criterion.equivalent_stress(np.array([[c * c, s * s, 0, 0, 0, s * c]]))[0])
    return np.array(ratios)


def test_isotropic_limits_reduce_to_classical_invariants_and_hershey():
    X = np.random.default_rng(0).normal(size=(300, 6))
    T = _deviator_tensor(X)
    J2 = 0.5 * np.einsum("nij,nji->n", T, T)
    J3 = np.linalg.det(T)
    j2, j3 = CB2001(a=(1.0,) * 6, b=(1.0,) * 11, c=1.0).invariants(X)
    np.testing.assert_allclose(j2, J2, atol=1e-12)
    np.testing.assert_allclose(j3, J3, atol=1e-12)

    ones = {key: 1.0 for key in KEYS}
    S = np.linalg.eigvalsh(T)
    hershey = ((np.abs(S[:, :, None] - S[:, None, :]) ** 8).sum(axis=(1, 2)) / 4) ** (1 / 8)
    np.testing.assert_allclose(Yld2004_18p(ones, ones).equivalent_stress(X), hershey, rtol=1e-12)


@pytest.mark.parametrize("name", sorted(CRITERIA))
def test_presets_are_valid_homogeneous_and_pressure_insensitive(name):
    criterion = CRITERIA[name]
    X = np.random.default_rng(1).normal(size=(2000, 6))
    value = criterion.equivalent_stress(X)
    assert np.isfinite(value).all() and (value > 0).all()
    np.testing.assert_allclose(criterion.equivalent_stress(2.5 * X), 2.5 * value, rtol=1e-10)
    np.testing.assert_allclose(
        criterion.equivalent_stress(X + np.array([1.0, 1.0, 1.0, 0, 0, 0])), value, rtol=1e-10,
    )
    assert criterion.equivalent_stress(np.array([[1.0, 0, 0, 0, 0, 0]]))[0] == pytest.approx(1.0)


def test_presets_reproduce_published_directional_yield_stresses():
    degrees = [0, 15, 30, 45, 60, 75, 90]
    # Barlat et al. (2005) Fig. 7, 2090-T3 (read from the figure, +-0.02).
    paper_2090 = np.array([1.00, 0.96, 0.91, 0.81, 0.81, 0.88, 0.91])
    np.testing.assert_allclose(_directional_yield(CRITERIA["yld2004_2090T3"], degrees), paper_2090, atol=0.02)
    np.testing.assert_allclose(_directional_yield(CRITERIA["cb2001_2090T3"], degrees), paper_2090, atol=0.05)
    # Cazacu & Barlat (2001) Fig. 4, 6016-T4 (read from the figure, +-0.01).
    paper_6016 = np.array([1.00, 1.003, 0.995, 0.983, 0.965, 0.950, 0.943])
    np.testing.assert_allclose(_directional_yield(CRITERIA["cb2001_6016T4"], degrees), paper_6016, atol=0.01)


def test_orthotropic_polynomial_pool_spans_cb2001_invariants():
    import torch

    from invariant_generator.config import Config
    from invariant_generator.input_features import OrthotropicPolynomialPool, input_feature_names
    from invariant_generator.model import InvariantYieldModel

    pool = OrthotropicPolynomialPool()
    assert pool.output_dim == 29
    X = np.random.default_rng(2).normal(size=(1000, 6))
    F = pool(torch.as_tensor(X)).numpy()
    for target in CRITERIA["cb2001_2090T3"].invariants(X):
        weights, *_ = np.linalg.lstsq(F, target, rcond=None)
        np.testing.assert_allclose(F @ weights, target, atol=1e-10)

    config = Config()
    config.input_transform.kind = "orthotropic_polynomial"
    config.constraints.A_psd.enabled = False
    config.constraints.a_psd.enabled = False
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    assert len(input_feature_names(config)) == 29
    model = InvariantYieldModel.from_config(config)
    assert model(torch.as_tensor(X[:5], dtype=torch.float32)).shape == (5,)


def test_graded_rows_split_a_mixed_quadratic_cubic_row_space():
    import torch

    from invariant_generator.input_features import OrthotropicPolynomialPool
    from invariant_generator.sparsify import graded_encoder_rows

    pool = OrthotropicPolynomialPool()
    degrees = np.array([len(term) for term in pool.terms])
    X = np.random.default_rng(3).normal(size=(1000, 6))
    F = pool(torch.as_tensor(X)).numpy()
    truth = np.vstack([
        np.linalg.lstsq(F, target, rcond=None)[0]
        for target in CRITERIA["cb2001_2090T3"].invariants(X)
    ])
    mixed = np.array([[0.7, -1.3], [2.0, 0.4]]) @ truth
    rows, residual = graded_encoder_rows(mixed, degrees)
    assert residual < 1e-8
    for row in rows:
        assert len(set(degrees[np.abs(row) > 1e-12])) == 1
        cosines = [abs(row @ t) / (np.linalg.norm(row) * np.linalg.norm(t)) for t in truth]
        assert max(cosines) == pytest.approx(1.0, abs=1e-8)
