import torch

from invariant_generator.invariants import (
    InvariantPool,
    stress_vector_to_tensor,
    symmetric_tensor_to_stress_vector,
)


def _random_rotation() -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def test_stress_vector_tensor_roundtrip():
    stress = torch.randn(16, 6)
    sigma = stress_vector_to_tensor(stress)
    recovered = symmetric_tensor_to_stress_vector(sigma)
    torch.testing.assert_close(recovered, stress)


def test_basic_invariants_are_rotation_invariant():
    stress = torch.randn(32, 6)
    sigma = stress_vector_to_tensor(stress)
    R = _random_rotation()

    sigma_rot = torch.einsum("ij,bjk,lk->bil", R, sigma, R)
    stress_rot = symmetric_tensor_to_stress_vector(sigma_rot)

    pool = InvariantPool(["I1", "I2", "I3"])
    values = pool(stress)
    values_rot = pool(stress_rot)

    torch.testing.assert_close(values_rot, values, rtol=1e-5, atol=1e-5)


def test_structural_invariants_have_trainable_gradients():
    stress = torch.randn(8, 6)
    pool = InvariantPool(
        ["I4", "I11"],
        enable_second_order=True,
        enable_fourth_order=True,
    )

    loss = pool(stress).sum()
    loss.backward()

    assert pool.raw_a is not None
    assert pool.raw_A is not None
    assert pool.raw_a.grad is not None
    assert pool.raw_A.grad is not None
    assert torch.isfinite(pool.raw_a.grad).all()
    assert torch.isfinite(pool.raw_A.grad).all()


def test_homogenized_quadratic_invariant_scales_linearly():
    stress = torch.randn(12, 6)
    k = 2.5
    pool = InvariantPool(["I2"], homogenize=True)

    base = pool(stress)
    scaled = pool(k * stress)

    torch.testing.assert_close(scaled, k * base, rtol=1e-5, atol=1e-5)
