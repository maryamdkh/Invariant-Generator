"""Interpretable input maps shared by the adaptive encoder stages."""

from __future__ import annotations

import torch
from torch import nn

from invariant_generator.config import Config


def orthotropic_polynomial_terms() -> list[tuple[str, tuple[int, ...]]]:
    """Degree-2 and degree-3 monomials in the orthotropic integrity basis.

    For orthotropy with axes x, y, z every polynomial stress invariant is a
    polynomial in sxx, syy, szz, syz^2, sxz^2, sxy^2 and syz*sxz*sxy (Cazacu &
    Barlat 2001, Eqs. 2-3; Zheng 1994). Terms are index tuples into the Voigt
    vector [sxx, syy, szz, syz, sxz, sxy].
    """
    names = ["sxx", "syy", "szz", "syz", "sxz", "sxy"]
    normal = [0, 1, 2]
    shear = [3, 4, 5]
    terms: list[tuple[int, ...]] = []
    terms += [(i, j) for a, i in enumerate(normal) for j in normal[a:]]
    terms += [(k, k) for k in shear]
    terms += [(i, j, l) for a, i in enumerate(normal) for b, j in enumerate(normal[a:], a)
              for l in normal[b:]]
    terms += [(k, k, i) for k in shear for i in normal]
    terms += [(3, 4, 5)]

    def label(term: tuple[int, ...]) -> str:
        return "*".join(names[index] for index in term)

    return [(label(term), term) for term in terms]


def input_feature_names(config: Config) -> list[str]:
    if config.input_transform.kind == "stress_invariants":
        return list(config.invariants.selected)
    if config.input_transform.kind == "orthotropic_polynomial":
        return [name for name, _ in orthotropic_polynomial_terms()]
    if config.input_transform.kind == "log":
        names = list(config.data.feature_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Log input names must be nonempty and unique.")
        return [f"log({name})" for name in names]
    if config.input_transform.kind != "pairwise":
        raise ValueError(f"Unknown input transformation: {config.input_transform.kind!r}")
    names = list(config.data.feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Pairwise input names must be nonempty and unique.")
    features = names.copy()
    if config.input_transform.include_products:
        features.extend(
            f"{names[i]}*{names[j]}"
            for i in range(len(names)) for j in range(i, len(names))
        )
    if config.input_transform.include_ratios:
        features.extend(
            f"{names[i]}/{names[j]}"
            for i in range(len(names)) for j in range(len(names)) if i != j
        )
    if len(features) > config.input_transform.max_features:
        raise ValueError(
            f"Pairwise map would create {len(features)} features, exceeding "
            f"max_features={config.input_transform.max_features}."
        )
    return features


class PairwiseFeaturePool(nn.Module):
    """One-level raw, product and ratio features; no recursive expansion."""

    def __init__(
        self,
        input_names: list[str],
        *,
        include_products: bool = True,
        include_ratios: bool = True,
        denominator_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("Input names must be nonempty and unique.")
        if denominator_floor <= 0:
            raise ValueError("denominator_floor must be positive.")
        self.input_names = list(input_names)
        self.input_dim = len(input_names)
        self.include_products = include_products
        self.include_ratios = include_ratios
        self.denominator_floor = float(denominator_floor)
        self.output_dim = (
            self.input_dim
            + (self.input_dim * (self.input_dim + 1) // 2 if include_products else 0)
            + (self.input_dim * (self.input_dim - 1) if include_ratios else 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (N, {self.input_dim}), got {tuple(x.shape)}")
        parts = [x]
        if self.include_products:
            parts.extend(
                (x[:, i] * x[:, j]).unsqueeze(1)
                for i in range(self.input_dim) for j in range(i, self.input_dim)
            )
        if self.include_ratios and self.input_dim > 1:
            if torch.any(x.abs() < self.denominator_floor):
                raise ValueError(
                    "A ratio denominator is too close to zero; disable ratios "
                    "or use a domain bounded away from zero."
                )
            parts.extend(
                (x[:, i] / x[:, j]).unsqueeze(1)
                for i in range(self.input_dim) for j in range(self.input_dim) if i != j
            )
        return torch.cat(parts, dim=1)

    def structural_norms(self) -> dict[str, torch.Tensor]:
        return {}


class OrthotropicPolynomialPool(nn.Module):
    """Degree-2 and degree-3 orthotropic invariant monomials of the stress.

    A linear encoder row over these features is a general orthotropic
    quadratic or cubic stress invariant (e.g. Hill's quadratic form, or the
    generalized J2 and J3 of Cazacu & Barlat 2001).
    """

    def __init__(self) -> None:
        super().__init__()
        self.terms = [term for _, term in orthotropic_polynomial_terms()]
        self.input_names = ["s11", "s22", "s33", "s23", "s13", "s12"]
        self.input_dim = 6
        self.output_dim = len(self.terms)

    def forward(self, stress: torch.Tensor) -> torch.Tensor:
        if stress.ndim != 2 or stress.shape[1] != 6:
            raise ValueError(f"Expected Voigt stress of shape (N, 6), got {tuple(stress.shape)}")
        columns = []
        for term in self.terms:
            value = stress[:, term[0]]
            for index in term[1:]:
                value = value * stress[:, index]
            columns.append(value)
        return torch.stack(columns, dim=1)

    def structural_norms(self) -> dict[str, torch.Tensor]:
        return {}


class LogFeaturePool(nn.Module):
    """log(x_i) for strictly positive inputs.

    A linear encoder row on these features is the log of a monomial,
    J = sum_j B_j log(x_j) = log(prod_j x_j^B_j), so the encoder learns
    power-law groups of the inputs (e.g. dimensionless groups).
    """

    def __init__(self, input_names: list[str]) -> None:
        super().__init__()
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("Input names must be nonempty and unique.")
        self.input_names = list(input_names)
        self.input_dim = len(input_names)
        self.output_dim = self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (N, {self.input_dim}), got {tuple(x.shape)}")
        if torch.any(x <= 0):
            raise ValueError("Log features require strictly positive inputs.")
        return torch.log(x)

    def structural_norms(self) -> dict[str, torch.Tensor]:
        return {}
