from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


BASIC_INVARIANTS = {"I1", "I2", "I3"}
SECOND_ORDER_INVARIANTS = {"I4", "I5", "I6", "I7", "I8", "I9", "I10"}
FOURTH_ORDER_INVARIANTS = {"I11", "I12", "I13"}
ALL_INVARIANTS = BASIC_INVARIANTS | SECOND_ORDER_INVARIANTS | FOURTH_ORDER_INVARIANTS

# Polynomial degree in stress sigma. This is used only when homogenize=True.
INVARIANT_DEGREES = {
    "I1": 1,
    "I2": 2,
    "I3": 3,
    "I4": 1,
    "I5": 1,
    "I6": 2,
    "I7": 2,
    "I8": 1,
    "I9": 2,
    "I10": 3,
    "I11": 2,
    "I12": 3,
    "I13": 4,
}


def stress_vector_to_tensor(stress: torch.Tensor) -> torch.Tensor:
    """
    Convert canonical 6-component stress vectors to symmetric 3x3 tensors.

    Input order:
        [s11, s22, s33, s23, s13, s12]
    """
    if stress.shape[-1] != 6:
        raise ValueError(
            "Stress input must have last dimension 6 ordered as "
            "[s11, s22, s33, s23, s13, s12]."
        )

    s11, s22, s33, s23, s13, s12 = stress.unbind(dim=-1)
    sigma = stress.new_zeros(stress.shape[:-1] + (3, 3))

    sigma[..., 0, 0] = s11
    sigma[..., 1, 1] = s22
    sigma[..., 2, 2] = s33
    sigma[..., 1, 2] = s23
    sigma[..., 2, 1] = s23
    sigma[..., 0, 2] = s13
    sigma[..., 2, 0] = s13
    sigma[..., 0, 1] = s12
    sigma[..., 1, 0] = s12

    return sigma


def symmetric_tensor_to_stress_vector(sigma: torch.Tensor) -> torch.Tensor:
    """Inverse of stress_vector_to_tensor for symmetric 3x3 tensors."""
    if sigma.shape[-2:] != (3, 3):
        raise ValueError(f"sigma must end with shape (3, 3), got {sigma.shape}.")
    return torch.stack(
        [
            sigma[..., 0, 0],
            sigma[..., 1, 1],
            sigma[..., 2, 2],
            sigma[..., 1, 2],
            sigma[..., 0, 2],
            sigma[..., 0, 1],
        ],
        dim=-1,
    )


def _trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1)


def _vector_norm(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor.reshape(-1), ord=2).clamp_min(eps)


def _signed_root(value: torch.Tensor, degree: int, eps: float) -> torch.Tensor:
    if degree == 1:
        return value
    # This keeps odd-degree negative invariants real-valued.
    return torch.sign(value) * torch.pow(torch.abs(value).clamp_min(eps), 1.0 / degree)


def _symmetrize_fourth_order(A: torch.Tensor) -> torch.Tensor:
    """
    Enforce minor and major symmetries suitable for sigma : A : sigma.

    This keeps the effective fourth-order structure tensor physically cleaner
    while still leaving the raw tensor fully trainable.
    """
    perms = [
        A,
        A.permute(1, 0, 2, 3),
        A.permute(0, 1, 3, 2),
        A.permute(1, 0, 3, 2),
        A.permute(2, 3, 0, 1),
        A.permute(3, 2, 0, 1),
        A.permute(2, 3, 1, 0),
        A.permute(3, 2, 1, 0),
    ]
    return torch.stack(perms, dim=0).mean(dim=0)


def _validate_invariant_names(names: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(names)
    if not selected:
        raise ValueError("At least one invariant must be selected.")

    unknown = sorted(set(selected).difference(ALL_INVARIANTS))
    if unknown:
        raise ValueError(f"Unknown invariant names: {unknown}")

    return selected


class InvariantPool(nn.Module):
    """
    Compute a configurable pool of stress invariants.

    Trainable red components in the notes are represented by:
        - raw_a: second-order structure tensor a
        - raw_A: fourth-order structure tensor A
    """

    def __init__(
        self,
        selected: Iterable[str],
        *,
        enable_second_order: bool = False,
        enable_fourth_order: bool = False,
        homogenize: bool = False,
        init_scale: float = 0.05,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.selected = _validate_invariant_names(selected)
        self.homogenize = bool(homogenize)
        self.eps = float(eps)

        needs_second = bool(set(self.selected) & SECOND_ORDER_INVARIANTS)
        needs_fourth = bool(set(self.selected) & FOURTH_ORDER_INVARIANTS)
        if needs_second and not enable_second_order:
            raise ValueError(
                "Selected second-order structural invariants require "
                "enable_second_order=True."
            )
        if needs_fourth and not enable_fourth_order:
            raise ValueError(
                "Selected fourth-order structural invariants require "
                "enable_fourth_order=True."
            )

        if enable_second_order:
            self.raw_a = nn.Parameter(self._unit_random((3, 3), init_scale))
        else:
            self.register_parameter("raw_a", None)

        if enable_fourth_order:
            self.raw_A = nn.Parameter(self._unit_random((3, 3, 3, 3), init_scale))
        else:
            self.register_parameter("raw_A", None)

    @staticmethod
    def _unit_random(shape: tuple[int, ...], init_scale: float) -> torch.Tensor:
        tensor = torch.randn(shape, dtype=torch.float32) * float(init_scale)
        norm = torch.linalg.vector_norm(tensor.reshape(-1), ord=2)
        if float(norm) < 1e-14:
            tensor = torch.randn(shape, dtype=torch.float64)
            norm = torch.linalg.vector_norm(tensor.reshape(-1), ord=2)
        return tensor / norm.clamp_min(1e-14)

    @property
    def output_dim(self) -> int:
        return len(self.selected)

    def effective_second_order_parts(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.raw_a is None:
            raise RuntimeError("Second-order structure tensor is not enabled.")

        a = self.raw_a
        s = 0.5 * (a + a.transpose(-1, -2))
        w = 0.5 * (a - a.transpose(-1, -2))
        return s, w

    def effective_fourth_order_tensor(self) -> torch.Tensor:
        if self.raw_A is None:
            raise RuntimeError("Fourth-order structure tensor is not enabled.")
        return _symmetrize_fourth_order(self.raw_A)

    def structural_norms(self) -> dict[str, torch.Tensor]:
        """Return 2-norms used by L_structure."""
        norms: dict[str, torch.Tensor] = {}
        if self.raw_a is not None:
            norms["a"] = _vector_norm(self.raw_a, self.eps)
        if self.raw_A is not None:
            norms["A"] = _vector_norm(self.effective_fourth_order_tensor(), self.eps)
        return norms

    def _compute_all(self, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
        sigma2 = sigma @ sigma
        sigma3 = sigma2 @ sigma

        values: dict[str, torch.Tensor] = {
            "I1": _trace(sigma),
            "I2": _trace(sigma2),
            "I3": _trace(sigma3),
        }

        if self.raw_a is not None:
            s, w = self.effective_second_order_parts()
            s = s.to(dtype=sigma.dtype, device=sigma.device)
            w = w.to(dtype=sigma.dtype, device=sigma.device)
            s2 = s @ s
            w2 = w @ w

            values.update(
                {
                    "I4": _trace(s @ sigma),
                    "I5": _trace(s2 @ sigma),
                    "I6": _trace(s @ sigma2),
                    "I7": _trace(s2 @ sigma2),
                    "I8": _trace(sigma @ w2),
                    "I9": _trace(sigma2 @ w2),
                    "I10": _trace(sigma2 @ w2 @ sigma @ w),
                }
            )

        if self.raw_A is not None:
            A = self.effective_fourth_order_tensor().to(dtype=sigma.dtype, device=sigma.device)
            values.update(
                {
                    "I11": torch.einsum("...ij,ijkl,...kl->...", sigma, A, sigma),
                    "I12": torch.einsum("...ij,ijkl,...kl->...", sigma2, A, sigma),
                    "I13": torch.einsum("...ij,ijkl,...kl->...", sigma2, A, sigma2),
                }
            )

        return values

    def forward(self, stress: torch.Tensor) -> torch.Tensor:
        sigma = stress_vector_to_tensor(stress)
        values = self._compute_all(sigma)

        selected_values = []
        for name in self.selected:
            value = values[name]
            if self.homogenize:
                value = _signed_root(value, INVARIANT_DEGREES[name], self.eps)
            selected_values.append(value)

        return torch.stack(selected_values, dim=-1)
