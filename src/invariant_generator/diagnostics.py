from __future__ import annotations

import numpy as np
import torch

from invariant_generator.config import ConstraintsConfig
from invariant_generator.constraints import fourth_order_to_mandel_matrix, psd_eigenvalues
from invariant_generator.model import InvariantYieldModel


@torch.no_grad()
def constraint_diagnostics(
    model: InvariantYieldModel,
    constraints: ConstraintsConfig,
) -> dict[str, object]:
    """Return configured physics-constraint diagnostics."""
    A_psd = constraints.A_psd
    if not A_psd.enabled:
        return {}

    A = model.invariant_pool.effective_fourth_order_tensor()
    mandel = fourth_order_to_mandel_matrix(A)
    eigvals = psd_eigenvalues(mandel).detach().float().cpu().numpy()
    min_eigenvalue = float(np.min(eigvals))
    passed = min_eigenvalue >= float(A_psd.min_eigenvalue) - float(A_psd.tolerance)
    return {
        "A_psd": {
            "enabled": True,
            "mode": A_psd.mode.lower(),
            "target": A_psd.target,
            "basis": A_psd.basis.lower(),
            "min_eigenvalue": min_eigenvalue,
            "configured_min_eigenvalue": float(A_psd.min_eigenvalue),
            "tolerance": float(A_psd.tolerance),
            "passed": bool(passed),
            "eigenvalues": [float(value) for value in eigvals],
        }
    }


def flatten_constraint_diagnostics(diagnostics: dict[str, object]) -> dict[str, float]:
    """Flatten selected diagnostics for per-epoch history rows."""
    A_psd = diagnostics.get("A_psd")
    if not isinstance(A_psd, dict):
        return {}
    return {
        "constraint_A_psd_min_eigenvalue": float(A_psd["min_eigenvalue"]),
        "constraint_A_psd_passed": 1.0 if bool(A_psd["passed"]) else 0.0,
    }


@torch.no_grad()
def invariant_feature_statistics(
    model: InvariantYieldModel,
    X: np.ndarray,
    invariant_names: list[str],
    *,
    device: torch.device,
    batch_size: int = 8192,
    normalized: bool = False,
) -> dict[str, object]:
    """Compute raw or normalized invariant feature statistics."""
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    outputs: list[np.ndarray] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size].to(device)
        if normalized:
            features = model.normalized_invariant_features(batch)
        else:
            features = model.raw_invariant_features(batch)
        outputs.append(features.detach().cpu().numpy())

    values = np.concatenate(outputs, axis=0)
    return {
        "names": list(invariant_names),
        "encoder_input": bool(normalized),
        "normalized": bool(normalized and model.normalizer is not None),
        "mean": [float(value) for value in values.mean(axis=0)],
        "std": [float(value) for value in values.std(axis=0, ddof=0)],
        "min": [float(value) for value in values.min(axis=0)],
        "max": [float(value) for value in values.max(axis=0)],
    }


def encoder_score_diagnostics(
    model: InvariantYieldModel,
    invariant_names: list[str],
    *,
    feature_std: np.ndarray | list[float] | None = None,
) -> dict[str, object]:
    """Report raw and feature-scale-adjusted encoder column scores."""
    S = model.encoder_matrix()
    if S is None:
        return {}

    S_detached = S.detach().float().cpu()
    raw_l1 = S_detached.abs().sum(dim=0).numpy()
    raw_l2 = torch.linalg.vector_norm(S_detached, ord=2, dim=0).numpy()
    if feature_std is None:
        std = np.ones_like(raw_l2)
    else:
        std = np.asarray(feature_std, dtype=np.float64)
    scaled_l2 = raw_l2 * std

    return {
        "names": list(invariant_names),
        "raw_l1": [float(value) for value in raw_l1],
        "raw_l2": [float(value) for value in raw_l2],
        "feature_std": [float(value) for value in std],
        "scaled_l2": [float(value) for value in scaled_l2],
    }
