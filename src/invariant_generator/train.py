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
            "parameter_counts": model.parameter_counts(),
        },
        path,
    )
    return path


def _module_grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0

    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().float().norm(2)
        total += float(grad_norm.cpu()) ** 2
    return total**0.5


def _gradient_diagnostics(model: InvariantYieldModel) -> dict[str, float]:
    return {
        "grad_norm_total": _module_grad_norm(model),
        "grad_norm_invariant_pool": _module_grad_norm(model.invariant_pool),
        "grad_norm_encoder": _module_grad_norm(model.encoder),
        "grad_norm_regressor": _module_grad_norm(model.regressor),
    }


def _encoder_diagnostics(
    model: InvariantYieldModel,
    invariant_names: list[str],
) -> dict[str, float]:
    S = model.encoder_matrix()
    if S is None:
        return {}

    S_detached = S.detach().float().cpu()
    diagnostics = {
        "encoder_weight_l1": float(torch.linalg.vector_norm(S_detached.reshape(-1), ord=1)),
        "encoder_weight_l2": float(torch.linalg.vector_norm(S_detached.reshape(-1), ord=2)),
        "encoder_weight_max_abs": float(S_detached.abs().max()),
    }

    column_l1 = S_detached.abs().sum(dim=0)
    column_l2 = torch.linalg.vector_norm(S_detached, ord=2, dim=0)
    for idx, name in enumerate(invariant_names):
        diagnostics[f"encoder_col_l1_{name}"] = float(column_l1[idx])
        diagnostics[f"encoder_col_l2_{name}"] = float(column_l2[idx])

    return diagnostics


@torch.no_grad()
def _evaluate_loss_terms(
    model: InvariantYieldModel,
    criterion: YieldSurfaceLoss,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    data_loss = torch.zeros((), device=device)
    for start in range(0, X_tensor.shape[0], batch_size):
        X_batch = X_tensor[start : start + batch_size].to(device)
        y_batch = y_tensor[start : start + batch_size].to(device)
        prediction = model(X_batch)
        if prediction.shape != y_batch.shape:
            y_batch = y_batch.reshape_as(prediction)
        data_loss = data_loss + (prediction - y_batch).pow(2).sum()

    zero = data_loss.new_zeros((1,))
    regularization = criterion(model, zero, zero).detached()
    data = float(data_loss.cpu())
    param = regularization["param"]
    structure = regularization["structure"]
    encoder = regularization["encoder"]
    return {
        "loss_total": data + param + structure + encoder,
        "loss_data": data,
        "loss_param": param,
        "loss_structure": structure,
        "loss_encoder": encoder,
    }


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
    parameter_counts = model.parameter_counts()

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
    print(
        "[INFO] Trainable params: "
        f"{parameter_counts['trainable']} "
        f"(invariants={parameter_counts['invariant_pool_trainable']}, "
        f"encoder={parameter_counts['encoder_trainable']}, "
        f"regressor={parameter_counts['regressor_trainable']})"
    )

    for epoch in trange(1, config.train.epochs + 1, desc="training"):
        model.train()
        epoch_terms = {
            "loss_total": 0.0,
            "loss_data": 0.0,
            "loss_param": 0.0,
            "loss_structure": 0.0,
            "loss_encoder": 0.0,
        }
        epoch_grad_terms = {
            "grad_norm_total": 0.0,
            "grad_norm_invariant_pool": 0.0,
            "grad_norm_encoder": 0.0,
            "grad_norm_regressor": 0.0,
        }
        n_grad_observations = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = criterion(model, prediction, y_batch)
            loss.total.backward()

            grad_terms = _gradient_diagnostics(model)
            for key, value in grad_terms.items():
                epoch_grad_terms[key] += value
            n_grad_observations += 1

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

        if n_grad_observations:
            for key in epoch_grad_terms:
                epoch_grad_terms[key] /= n_grad_observations

        if should_log or should_save:
            train_loss_terms = _evaluate_loss_terms(
                model,
                criterion,
                data.X_train,
                data.y_train,
                device=device,
                batch_size=8192,
            )
            test_loss_terms = _evaluate_loss_terms(
                model,
                criterion,
                data.X_test,
                data.y_test,
                device=device,
                batch_size=8192,
            )
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
                **{f"train_{k}": v for k, v in train_loss_terms.items()},
                **{f"test_{k}": v for k, v in test_loss_terms.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **epoch_grad_terms,
                **_encoder_diagnostics(model, config.invariants.selected),
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
                    f"train_total={train_loss_terms['loss_total']:.6g} "
                    f"data={train_loss_terms['loss_data']:.6g} "
                    f"param={train_loss_terms['loss_param']:.6g} "
                    f"structure={train_loss_terms['loss_structure']:.6g} "
                    f"encoder={train_loss_terms['loss_encoder']:.6g} "
                    f"test_total={test_loss_terms['loss_total']:.6g} "
                    f"test_data={test_loss_terms['loss_data']:.6g} "
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
    final_train_loss_terms = _evaluate_loss_terms(
        model,
        criterion,
        data.X_train,
        data.y_train,
        device=device,
        batch_size=8192,
    )
    final_test_loss_terms = _evaluate_loss_terms(
        model,
        criterion,
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
            "final_train_loss": final_train_loss_terms,
            "final_test_loss": final_test_loss_terms,
            "best_epoch": best_epoch,
            "best_test_mse": best_test_mse,
            "parameter_counts": parameter_counts,
            "encoder_diagnostics": _encoder_diagnostics(
                model,
                config.invariants.selected,
            ),
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
