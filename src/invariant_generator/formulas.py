from __future__ import annotations

import numpy as np

from invariant_generator.model import InvariantYieldModel


def encoder_to_raw_coefficients(
    S: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=np.float64)
    if mean is None or std is None:
        return np.zeros(S.shape[0], dtype=np.float64), S.copy()

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (S.shape[1],) or std.shape != (S.shape[1],):
        raise ValueError("Normalizer mean/std must match encoder input dimension.")
    safe_std = np.where(np.abs(std) < 1e-15, 1.0, std)
    beta = S / safe_std[None, :]
    intercept = -np.sum(S * mean[None, :] / safe_std[None, :], axis=1)
    return intercept, beta


def render_linear_formula(
    output_name: str,
    coefficients: np.ndarray,
    input_names: list[str],
    *,
    intercept: float = 0.0,
    threshold: float = 1e-10,
    precision: int = 6,
) -> str:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (len(input_names),):
        raise ValueError("coefficients must have one value per input name.")

    terms: list[str] = []
    if abs(float(intercept)) > threshold:
        terms.append(f"{intercept:.{precision}g}")
    for coefficient, input_name in zip(coefficients, input_names):
        coefficient = float(coefficient)
        if abs(coefficient) <= threshold:
            continue
        sign = "-" if coefficient < 0.0 else "+"
        magnitude = abs(coefficient)
        body = input_name if np.isclose(magnitude, 1.0) else f"{magnitude:.{precision}g}*{input_name}"
        if not terms:
            terms.append(body if sign == "+" else f"-{body}")
        else:
            terms.append(f"{sign} {body}")

    return f"{output_name} = " + (" ".join(terms) if terms else "0")


def encoder_formula_report(
    model: InvariantYieldModel,
    invariant_names: list[str],
    *,
    threshold: float = 1e-10,
    precision: int = 6,
) -> dict[str, object]:
    S = model.encoder_matrix()
    if S is None:
        raise ValueError("Formula report requires an enabled encoder.")

    S_np = S.detach().float().cpu().numpy().astype(np.float64)
    normalizer = model.normalizer
    mean = None
    std = None
    if normalizer is not None:
        mean = normalizer.mean.detach().float().cpu().numpy().astype(np.float64)
        std = normalizer.std.detach().float().cpu().numpy().astype(np.float64)

    intercept, raw_coefficients = encoder_to_raw_coefficients(S_np, mean=mean, std=std)
    formulas = []
    for row_idx in range(S_np.shape[0]):
        output_name = f"J{row_idx + 1}"
        active = [
            invariant_names[col_idx]
            for col_idx, value in enumerate(S_np[row_idx])
            if abs(float(value)) > threshold
        ]
        formulas.append(
            {
                "name": output_name,
                "active_terms": active,
                "active_count": len(active),
                "normalized_formula": render_linear_formula(
                    output_name,
                    S_np[row_idx],
                    [f"z_{name}" for name in invariant_names],
                    threshold=threshold,
                    precision=precision,
                ),
                "raw_formula": render_linear_formula(
                    output_name,
                    raw_coefficients[row_idx],
                    invariant_names,
                    intercept=float(intercept[row_idx]),
                    threshold=threshold,
                    precision=precision,
                ),
            }
        )

    return {
        "invariant_names": list(invariant_names),
        "threshold": float(threshold),
        "normalization": None
        if normalizer is None
        else {
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        },
        "S": S_np.tolist(),
        "raw_intercept": intercept.tolist(),
        "raw_coefficients": raw_coefficients.tolist(),
        "formulas": formulas,
    }
