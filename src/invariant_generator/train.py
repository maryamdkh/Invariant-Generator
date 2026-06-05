from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from invariant_generator.config import Config
from invariant_generator.data import PreparedData, prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.losses import YieldSurfaceLoss
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import (
    now_utc_iso,
    resolve_device,
    save_config_snapshot,
    save_json,
    seed_everything,
    to_jsonable,
)


@dataclass(slots=True)
class TrainResult:
    experiment_dir: Path
    best_checkpoint: Path
    recovery_checkpoint: Path
    history_path: Path
    best_epoch: int
    best_test_mse: float


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _save_checkpoint(
    path: Path,
    *,
    model: InvariantYieldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: Config,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": to_jsonable(config),
        },
        path,
    )
    return path


def train_from_config(config: Config) -> TrainResult:
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)

    experiment_dir = config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config.train.split_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, experiment_dir)

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    criterion = YieldSurfaceLoss(config.loss)
    # No optimizer weight decay is used here: all regularization should appear
    # explicitly in YieldSurfaceLoss so the optimized objective matches the notes.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    loader = _make_loader(
        data.X_train,
        data.y_train,
        batch_size=config.train.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float | int | str]] = []
    best_test_mse = float("inf")
    best_epoch = 0
    best_checkpoint = experiment_dir / "checkpoint_best.pt"
    recovery_checkpoint = experiment_dir / "checkpoint_latest.pt"
    started_at = now_utc_iso()
    t0 = time.perf_counter()

    print(f"[INFO] Device:       {device}")
    print(f"[INFO] Dataset:      {config.data.dataset_name}")
    print(f"[INFO] Split path:   {data.split_path}")
    print(f"[INFO] Train shape:  {data.X_train.shape}")
    print(f"[INFO] Test shape:   {data.X_test.shape}")
    print(f"[INFO] Invariants:   {', '.join(config.invariants.selected)}")
    print(f"[INFO] Results dir:  {experiment_dir}")

    for epoch in trange(1, config.train.epochs + 1, desc="training"):
        model.train()
        epoch_terms = {
            "loss_total": 0.0,
            "loss_data": 0.0,
            "loss_param": 0.0,
            "loss_structure": 0.0,
            "loss_encoder": 0.0,
        }

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = criterion(model, prediction, y_batch)
            loss.total.backward()

            if config.train.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.train.grad_clip_norm,
                )

            optimizer.step()

            detached = loss.detached()
            epoch_terms["loss_total"] += detached["total"]
            epoch_terms["loss_data"] += detached["data"]
            epoch_terms["loss_param"] += detached["param"]
            epoch_terms["loss_structure"] += detached["structure"]
            epoch_terms["loss_encoder"] += detached["encoder"]

        should_log = (
            epoch == 1
            or epoch == config.train.epochs
            or (config.train.log_every > 0 and epoch % config.train.log_every == 0)
        )
        should_save = (
            config.train.save_every > 0 and epoch % config.train.save_every == 0
        )

        if should_log or should_save:
            test_metrics = evaluate_model(
                model,
                data.X_test,
                data.y_test,
                device=device,
                batch_size=8192,
            )
            row: dict[str, float | int | str] = {
                "epoch": epoch,
                **epoch_terms,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            history.append(row)

            if test_metrics["mse"] < best_test_mse:
                best_test_mse = test_metrics["mse"]
                best_epoch = epoch
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=test_metrics,
                    config=config,
                )

            if should_log:
                print(
                    "[INFO] "
                    f"epoch={epoch:05d} "
                    f"loss={epoch_terms['loss_total']:.6g} "
                    f"data={epoch_terms['loss_data']:.6g} "
                    f"test_mse={test_metrics['mse']:.6g}"
                )

        if should_save:
            _save_checkpoint(
                recovery_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=history[-1] if history else {},
                config=config,
            )

    final_metrics = evaluate_model(
        model,
        data.X_test,
        data.y_test,
        device=device,
        batch_size=8192,
    )

    finished_at = now_utc_iso()
    history_path = save_json(
        experiment_dir / "history.json",
        {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": time.perf_counter() - t0,
            "history": history,
            "final_metrics": final_metrics,
            "best_epoch": best_epoch,
            "best_test_mse": best_test_mse,
        },
    )

    # A successful run should leave the durable best checkpoint, not an extra
    # final/latest checkpoint. If training is interrupted, this file remains as
    # the rolling recovery point.
    recovery_checkpoint.unlink(missing_ok=True)

    return TrainResult(
        experiment_dir=experiment_dir,
        best_checkpoint=best_checkpoint,
        recovery_checkpoint=recovery_checkpoint,
        history_path=history_path,
        best_epoch=best_epoch,
        best_test_mse=best_test_mse,
    )
