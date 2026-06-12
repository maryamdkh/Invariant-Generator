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
from invariant_generator.diagnostics import (
    constraint_diagnostics,
    encoder_score_diagnostics,
    flatten_constraint_diagnostics,
    invariant_feature_statistics,
)
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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    epoch: int,
    metrics: dict[str, float],
    config: Config,
    stop_reason: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": to_jsonable(config),
        "parameter_counts": model.parameter_counts(),
        "invariant_normalization": model.invariant_normalization_state(),
        "stop_reason": stop_reason,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _make_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not config.train.lr_plateau_enabled:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.lr_plateau_factor,
        patience=config.train.lr_plateau_patience,
        threshold=config.train.lr_plateau_min_delta,
        threshold_mode="abs",
        min_lr=config.train.lr_plateau_min_lr,
    )


def _read_stop_keyword(config: Config) -> str | None:
    stop_file = config.train.stop_file
    if not stop_file.exists():
        return None

    try:
        text = stop_file.read_text(encoding="utf-8").lower()
    except OSError as exc:
        print(f"[WARN] Could not read stop file {stop_file}: {exc}")
        return None

    words = set(text.replace(",", " ").split())
    for keyword in config.train.stop_keywords:
        keyword = str(keyword).lower()
        if keyword in words:
            return keyword
    return None


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
def _fit_invariant_normalizer(
    model: InvariantYieldModel,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> None:
    if model.normalizer is None:
        return

    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]

    outputs: list[torch.Tensor] = []
    for start in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[start : start + batch_size].to(device)
        outputs.append(model.raw_invariant_features(batch).detach().cpu())

    values = torch.cat(outputs, dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(model.normalizer.eps)
    model.normalizer.set_statistics(mean.to(device), std.to(device))


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
    constraint = regularization["constraint"]
    return {
        "loss_total": data + param + structure + encoder + constraint,
        "loss_data": data,
        "loss_param": param,
        "loss_structure": structure,
        "loss_encoder": encoder,
        "loss_constraint": constraint,
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
    _fit_invariant_normalizer(
        model,
        data.X_train,
        device=device,
        batch_size=config.normalization.batch_size,
    )
    criterion = YieldSurfaceLoss(config.loss, config.constraints)

    # No optimizer weight decay is used here: all regularization should appear
    # explicitly in YieldSurfaceLoss so the optimized objective matches the notes.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = _make_plateau_scheduler(optimizer, config)
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
    stop_reason: str | None = None
    evaluations_without_improvement = 0

    print(f"[INFO] Device:       {device}")
    print(f"[INFO] Dataset:      {config.data.dataset_name}")
    print(f"[INFO] Split path:   {data.split_path}")
    print(f"[INFO] Train shape:  {data.X_train.shape}")
    print(f"[INFO] Test shape:   {data.X_test.shape}")
    print(f"[INFO] Invariants:   {', '.join(config.invariants.selected)}")
    print(f"[INFO] Results dir:  {experiment_dir}")
    print(f"[INFO] Initial LR:   {_current_learning_rate(optimizer):.6g}")
    print(f"[INFO] Stop file:    {config.train.stop_file}")
    print(
        "[INFO] Trainable params: "
        f"{parameter_counts['trainable']} "
        f"(invariants={parameter_counts['invariant_pool_trainable']}, "
        f"encoder={parameter_counts['encoder_trainable']}, "
        f"regressor={parameter_counts['regressor_trainable']})"
    )

    for epoch in trange(1, config.train.epochs + 1, desc="training"):
        if epoch > 1:
            stop_keyword = _read_stop_keyword(config)
            if stop_keyword is not None:
                stop_reason = f"stop keyword '{stop_keyword}' found in {config.train.stop_file}"
                print(f"[INFO] Stopping before epoch {epoch}: {stop_reason}")
                break

        model.train()
        epoch_terms = {
            "loss_total": 0.0,
            "loss_data": 0.0,
            "loss_param": 0.0,
            "loss_structure": 0.0,
            "loss_encoder": 0.0,
            "loss_constraint": 0.0,
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
            if not torch.isfinite(loss.total):
                stop_reason = f"non-finite loss at epoch {epoch}"
                print(f"[WARN] {stop_reason}")
                break
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
            epoch_terms["loss_constraint"] += detached["constraint"]

        if stop_reason is not None:
            break

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
            current_constraint_diagnostics = constraint_diagnostics(
                model,
                config.constraints,
            )
            row: dict[str, float | int | str] = {
                "epoch": epoch,
                **epoch_terms,
                **{f"train_{k}": v for k, v in train_loss_terms.items()},
                **{f"test_{k}": v for k, v in test_loss_terms.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **epoch_grad_terms,
                **_encoder_diagnostics(model, config.invariants.selected),
                **flatten_constraint_diagnostics(current_constraint_diagnostics),
            }
            row["learning_rate"] = _current_learning_rate(optimizer)
            history.append(row)

            improved = test_metrics["mse"] < (
                best_test_mse - config.train.early_stopping_min_delta
            )
            if improved:
                best_test_mse = test_metrics["mse"]
                best_epoch = epoch
                evaluations_without_improvement = 0
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=test_metrics,
                    config=config,
                )
            else:
                evaluations_without_improvement += 1

            if scheduler is not None:
                previous_lr = _current_learning_rate(optimizer)
                scheduler.step(test_metrics["mse"])
                current_lr = _current_learning_rate(optimizer)
                if current_lr < previous_lr:
                    row["learning_rate"] = current_lr
                    print(
                        "[INFO] "
                        f"epoch={epoch:05d} "
                        f"reducing learning_rate {previous_lr:.6g} -> {current_lr:.6g}"
                    )

            if (
                config.train.early_stopping_enabled
                and evaluations_without_improvement >= config.train.early_stopping_patience
            ):
                stop_reason = (
                    "early stopping: "
                    f"test_mse did not improve by {config.train.early_stopping_min_delta:g} "
                    f"for {evaluations_without_improvement} evaluations"
                )
                print(f"[INFO] {stop_reason}")

            if should_log:
                print(
                    "[INFO] "
                    f"epoch={epoch:05d} "
                    f"train_total={train_loss_terms['loss_total']:.6g} "
                    f"data={train_loss_terms['loss_data']:.6g} "
                    f"param={train_loss_terms['loss_param']:.6g} "
                    f"structure={train_loss_terms['loss_structure']:.6g} "
                    f"encoder={train_loss_terms['loss_encoder']:.6g} "
                    f"constraint={train_loss_terms['loss_constraint']:.6g} "
                    f"test_total={test_loss_terms['loss_total']:.6g} "
                    f"test_data={test_loss_terms['loss_data']:.6g} "
                    f"test_mse={test_metrics['mse']:.6g} "
                    f"lr={_current_learning_rate(optimizer):.6g} "
                    f"stale={evaluations_without_improvement}"
                )

        if should_save:
            _save_checkpoint(
                recovery_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=history[-1] if history else {},
                config=config,
                stop_reason=stop_reason,
            )

        if stop_reason is not None:
            break

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
    final_constraint_diagnostics = constraint_diagnostics(model, config.constraints)
    final_invariant_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        batch_size=8192,
    )
    final_encoder_input_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        batch_size=8192,
        normalized=True,
    )
    final_encoder_scores = encoder_score_diagnostics(
        model,
        config.invariants.selected,
        feature_std=final_encoder_input_stats["std"],
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
            "last_epoch": history[-1]["epoch"] if history else 0,
            "stop_reason": stop_reason,
            "final_learning_rate": _current_learning_rate(optimizer),
            "parameter_counts": parameter_counts,
            "encoder_diagnostics": _encoder_diagnostics(
                model,
                config.invariants.selected,
            ),
            "constraint_diagnostics": final_constraint_diagnostics,
            "invariant_feature_statistics": final_invariant_stats,
            "encoder_input_feature_statistics": final_encoder_input_stats,
            "encoder_score_diagnostics": final_encoder_scores,
            "invariant_normalization": model.invariant_normalization_state(),
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
