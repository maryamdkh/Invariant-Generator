from __future__ import annotations

import numpy as np
import torch

from invariant_generator.model import InvariantYieldModel


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    return {
        "sse": float(np.sum(error**2)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


@torch.no_grad()
def predict_numpy(
    model: InvariantYieldModel,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    outputs: list[np.ndarray] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size].to(device)
        prediction = model(batch).detach().cpu().numpy()
        outputs.append(prediction)

    return np.concatenate(outputs, axis=0)


def evaluate_model(
    model: InvariantYieldModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> dict[str, float]:
    prediction = predict_numpy(model, X, device=device, batch_size=batch_size)
    return regression_metrics(prediction, y)
