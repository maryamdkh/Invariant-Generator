from __future__ import annotations

import math

import torch


MANDEL_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)


def mandel_basis(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return an orthonormal Mandel basis for symmetric 3x3 tensors."""
    basis = torch.zeros((6, 3, 3), dtype=dtype, device=device)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0

    shear_scale = 1.0 / math.sqrt(2.0)
    for idx, (i, j) in enumerate(MANDEL_PAIRS[3:], start=3):
        basis[idx, i, j] = shear_scale
        basis[idx, j, i] = shear_scale
    return basis


def stress_tensor_to_mandel(sigma: torch.Tensor) -> torch.Tensor:
    """Convert symmetric stress tensors to Mandel vectors."""
    if sigma.shape[-2:] != (3, 3):
        raise ValueError(f"sigma must end with shape (3, 3), got {sigma.shape}.")

    sqrt2 = math.sqrt(2.0)
    return torch.stack(
        [
            sigma[..., 0, 0],
            sigma[..., 1, 1],
            sigma[..., 2, 2],
            sqrt2 * sigma[..., 1, 2],
            sqrt2 * sigma[..., 0, 2],
            sqrt2 * sigma[..., 0, 1],
        ],
        dim=-1,
    )


def mandel_matrix_to_fourth_order(matrix: torch.Tensor) -> torch.Tensor:
    """Map a 6x6 Mandel matrix to a fourth-order tensor with symmetries."""
    if matrix.shape[-2:] != (6, 6):
        raise ValueError(f"matrix must end with shape (6, 6), got {matrix.shape}.")

    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    basis = mandel_basis(dtype=matrix.dtype, device=matrix.device)
    return torch.einsum("...ab,aij,bkl->...ijkl", matrix, basis, basis)


def fourth_order_to_mandel_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Map a fourth-order tensor to its 6x6 Mandel representation."""
    if tensor.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"tensor must end with shape (3, 3, 3, 3), got {tensor.shape}.")

    basis = mandel_basis(dtype=tensor.dtype, device=tensor.device)
    matrix = torch.einsum("aij,...ijkl,bkl->...ab", basis, tensor, basis)
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def psd_matrix_from_factor(
    factor: torch.Tensor,
    *,
    min_eigenvalue: float = 0.0,
) -> torch.Tensor:
    """Create a symmetric PSD Mandel matrix from an unconstrained factor."""
    if factor.shape[-2:] != (6, 6):
        raise ValueError(f"factor must end with shape (6, 6), got {factor.shape}.")

    matrix = factor @ factor.transpose(-1, -2)
    if min_eigenvalue:
        eye = torch.eye(6, dtype=factor.dtype, device=factor.device)
        matrix = matrix + float(min_eigenvalue) * eye
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def psd_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """Return ordered eigenvalues of a symmetric 6x6 matrix."""
    if matrix.shape[-2:] != (6, 6):
        raise ValueError(f"matrix must end with shape (6, 6), got {matrix.shape}.")
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    return torch.linalg.eigvalsh(matrix)


def psd_penalty(
    matrix: torch.Tensor,
    *,
    min_eigenvalue: float = 0.0,
) -> torch.Tensor:
    """Squared penalty for eigenvalues below the requested lower bound."""
    eigvals = psd_eigenvalues(matrix)
    violation = torch.relu(float(min_eigenvalue) - eigvals)
    return violation.pow(2).sum()
