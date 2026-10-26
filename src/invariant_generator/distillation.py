"""Symbolic distillation baseline (Cranmer et al. 2020, arXiv:2006.11287).

An MLP x -> latent z -> y is trained with an L1 penalty on the latent
activations, so few latent components stay informative. The components with
the largest standard deviation are kept; PySR then fits each kept component
from the raw inputs (z_j = h_j(x)) and the target from the kept components
(y = g(z)), and the composition g(h(x)) is the distilled formula. The PySR time
limit is split evenly over the k + 1 searches, so the total budget matches a
single PySR run. Constants are not refit after composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class DistillationSettings:
    latent_dim: int = 8
    hidden: int = 128
    l1_weight: float = 1e-2
    epochs: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    keep_fraction: float = 0.01  # keep components with std > keep_fraction * max std
    max_kept: int = 4
    seed: int = 42


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs),
    )


class DistillationNet(nn.Module):
    def __init__(self, inputs: int, settings: DistillationSettings) -> None:
        super().__init__()
        self.encoder = _mlp(inputs, settings.hidden, settings.latent_dim)
        self.decoder = _mlp(settings.latent_dim, settings.hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z).squeeze(-1), z


def train_distillation_net(X: np.ndarray, y: np.ndarray, settings: DistillationSettings) -> tuple[DistillationNet, dict[str, Any]]:
    torch.manual_seed(settings.seed)
    x_scale = np.std(X, axis=0)
    x_scale[x_scale == 0] = 1.0
    y_scale = float(np.std(y)) or 1.0
    X_t = torch.as_tensor(X / x_scale, dtype=torch.float32)
    y_t = torch.as_tensor(y / y_scale, dtype=torch.float32)
    net = DistillationNet(X.shape[1], settings)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    generator = torch.Generator().manual_seed(settings.seed)
    for _ in range(settings.epochs):
        order = torch.randperm(len(X_t), generator=generator)
        for start in range(0, len(X_t), settings.batch_size):
            batch = order[start:start + settings.batch_size]
            prediction, z = net(X_t[batch])
            loss = torch.mean((prediction - y_t[batch]) ** 2) + settings.l1_weight * torch.mean(torch.abs(z))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    with torch.no_grad():
        prediction, _ = net(X_t)
    info = {"train_nrmse": float(torch.sqrt(torch.mean((prediction - y_t) ** 2)))}
    # Fold the input/target scaling into the callables used below.
    net.x_scale, net.y_scale = x_scale, y_scale  # type: ignore[attr-defined]
    return net, info


def latent(net: DistillationNet, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net.encoder(torch.as_tensor(X / net.x_scale, dtype=torch.float32)).double().numpy()  # type: ignore[attr-defined]


def distill(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    fit_pysr: Callable[[np.ndarray, np.ndarray, list[str], Path, float], Any],
    output_dir: Path,
    total_pysr_seconds: float,
    settings: DistillationSettings | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Return the composed predictor g(h(x)) and a report.

    fit_pysr(X, y, variable_names, output_dir, seconds) must return a fitted
    PySRRegressor (the caller supplies the shared PySR settings).
    """
    settings = settings or DistillationSettings()
    started = time.perf_counter()
    net, info = train_distillation_net(X, y, settings)
    Z = latent(net, X)
    std = Z.std(axis=0)
    order = np.argsort(std)[::-1]
    kept = [int(j) for j in order if std[j] > settings.keep_fraction * std[order[0]]][: settings.max_kept]
    info.update({"network_seconds": time.perf_counter() - started, "latent_std": std.tolist(), "kept": kept})

    seconds = total_pysr_seconds / (len(kept) + 1)
    started = time.perf_counter()
    inner = []
    for j in kept:
        model = fit_pysr(X, Z[:, j], names, output_dir / f"latent_{j}", seconds)
        inner.append(model)
    latent_names = [f"z{j}" for j in kept]
    outer = fit_pysr(Z[:, kept], y, latent_names, output_dir / "outer", seconds)
    info["pysr_seconds"] = time.perf_counter() - started
    info["latent_equations"] = {f"z{j}": str(m.sympy()) for j, m in zip(kept, inner)}
    info["outer_equation"] = str(outer.sympy())

    def predict(X_new: np.ndarray) -> np.ndarray:
        Z_new = np.column_stack([np.asarray(m.predict(X_new), dtype=np.float64) for m in inner])
        return np.asarray(outer.predict(Z_new), dtype=np.float64)

    return predict, info
