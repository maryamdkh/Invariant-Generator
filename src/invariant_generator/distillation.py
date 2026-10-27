"""Symbolic distillation baseline (Cranmer et al. 2020, arXiv:2006.11287).

An MLP x -> latent z -> y is trained with an L1 penalty on the latent
activations, so few latent components stay informative. A plain L1 on z can
be dodged by shrinking z and growing the decoder's first layer (all
components collapse together, nothing becomes sparse), so the penalty is
|z_j| * ||W_j||, with W_j the decoder weights reading z_j: unchanged by that
rescaling. Components are ranked by the same product std(z_j) * ||W_j||. Each
kept component is standardised; PySR then fits it from the raw inputs
(z_j = h_j(x)) and the target from the kept components (y = g(z)), and the
composition g(h(x)) is the distilled formula. The PySR time
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
    l1_weight: float = 5e-2
    epochs: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    keep_fraction: float = 0.2  # keep components with importance > keep_fraction * max importance
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

    def latent_weight_norms(self) -> torch.Tensor:
        """||W_j||: norm of the decoder weights that read latent component j."""
        return torch.linalg.vector_norm(self.decoder[0].weight, dim=0)


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
            penalty = torch.mean(torch.abs(z) * net.latent_weight_norms())
            loss = torch.mean((prediction - y_t[batch]) ** 2) + settings.l1_weight * penalty
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
    with torch.no_grad():
        importance = Z.std(axis=0) * net.latent_weight_norms().double().numpy()
    order = np.argsort(importance)[::-1]
    kept = [int(j) for j in order if importance[j] > settings.keep_fraction * importance[order[0]]][: settings.max_kept]
    # Standardise the kept components so PySR fits targets of unit scale.
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
    info.update({
        "network_seconds": time.perf_counter() - started, "latent_importance": importance.tolist(), "kept": kept,
    })

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
