from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.losses import YieldSurfaceLoss
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import (
    resolve_device,
    save_config_snapshot,
    save_json,
    seed_everything,
    to_jsonable,
)


@dataclass(slots=True)
class SparsifyResult:
    experiment_dir: Path
    checkpoint_path: Path
    summary_path: Path
    mask_path: Path
    sparse_history_path: Path | None
    refit_history_path: Path | None


def _make_loader(X: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    if batch_size <= 0:
        batch_size = X_tensor.shape[0]
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def encoder_l1_penalty(model: InvariantYieldModel) -> torch.Tensor:
    S = model.encoder_matrix()
    if S is None:
        return next(model.parameters()).new_zeros(())
    return S.abs().sum()


def encoder_gate_penalty(model: InvariantYieldModel) -> torch.Tensor:
    gates = model.encoder_gates()
    if gates is None:
        return next(model.parameters()).new_zeros(())
    return gates.sum()


@torch.no_grad()
def apply_encoder_mask(model: InvariantYieldModel, mask: torch.Tensor | np.ndarray) -> None:
    if model.encoder is None:
        raise ValueError("Encoder mask requires an enabled encoder.")
    mask_tensor = torch.as_tensor(
        mask,
        dtype=model.encoder.raw_weight.dtype,
        device=model.encoder.raw_weight.device,
    )
    if mask_tensor.shape != model.encoder.raw_weight.shape:
        raise ValueError(
            f"Mask shape {tuple(mask_tensor.shape)} does not match encoder "
            f"shape {tuple(model.encoder.raw_weight.shape)}."
        )
    model.encoder.raw_weight.mul_(mask_tensor)


def threshold_encoder_mask(S: torch.Tensor | np.ndarray, *, threshold: float) -> np.ndarray:
    values = np.asarray(S.detach().cpu().numpy() if isinstance(S, torch.Tensor) else S)
    mask = np.abs(values) > float(threshold)
    for row_idx in range(mask.shape[0]):
        if not np.any(mask[row_idx]):
            strongest = int(np.argmax(np.abs(values[row_idx])))
            mask[row_idx, strongest] = True
    return mask


def _sparsity_penalty(
    model: InvariantYieldModel,
    config: Config,
    *,
    method: str,
) -> torch.Tensor:
    zero = next(model.parameters()).new_zeros(())
    if method == "lasso":
        return config.sparsification.lambda_encoder_l1 * encoder_l1_penalty(model)
    if method == "gated":
        return (
            config.sparsification.lambda_encoder_l1 * encoder_l1_penalty(model)
            + config.sparsification.lambda_gate * encoder_gate_penalty(model)
        )
    if method in {"none", "masked_refit"}:
        return zero
    raise ValueError("sparsification.method must be 'lasso' or 'gated'.")


def _train_sparse_epochs(
    model: InvariantYieldModel,
    config: Config,
    *,
    data,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    method: str,
    mask: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    if epochs <= 0:
        return []

    criterion = YieldSurfaceLoss(config.loss, config.constraints)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loader = _make_loader(data.X_train, data.y_train, batch_size=batch_size, shuffle=True)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        data_loss = 0.0
        sparsity_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            loss = criterion(model, prediction, y_batch)
            sparsity = _sparsity_penalty(model, config, method=method)
            objective = loss.total + sparsity
            if not torch.isfinite(objective):
                raise RuntimeError(f"Non-finite sparsification loss at epoch {epoch}.")
            objective.backward()
            if config.train.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.train.grad_clip_norm,
                )
            optimizer.step()
            if mask is not None:
                apply_encoder_mask(model, mask)
            total += float(objective.detach().cpu())
            data_loss += float(loss.data.detach().cpu())
            sparsity_loss += float(sparsity.detach().cpu())

        history.append(
            {
                "epoch": epoch,
                "loss_total": total,
                "loss_data": data_loss,
                "loss_sparsity": sparsity_loss,
            }
        )
    return history


def load_model_for_checkpoint(config: Config, checkpoint_path: str | Path) -> InvariantYieldModel:
    device = resolve_device(config.train.device)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model


def sparsify_encoder_from_checkpoint(
    config: Config,
    *,
    checkpoint_path: str | Path,
    run_id: str | None = None,
) -> SparsifyResult:
    sparse_config = deepcopy(config)
    sparse_config.train.run_id = run_id or sparse_config.sparsification.run_id
    seed_everything(sparse_config.train.seed)
    device = resolve_device(sparse_config.train.device)
    data = prepare_training_data(sparse_config)
    model = load_model_for_checkpoint(sparse_config, checkpoint_path).to(device)
    if model.encoder is None:
        raise ValueError("Adaptive sparsification requires an enabled encoder.")

    experiment_dir = sparse_config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(sparse_config, experiment_dir)

    source_S = model.encoder_matrix().detach().float().cpu().numpy()
    method = sparse_config.sparsification.method.lower()
    if method not in {"lasso", "gated"}:
        raise ValueError("sparsification.method must be 'lasso' or 'gated'.")
    if method == "gated":
        model.encoder.enable_gates(
            init_probability=sparse_config.sparsification.gate_init_probability
        )

    sparse_history = _train_sparse_epochs(
        model,
        sparse_config,
        data=data,
        device=device,
        epochs=sparse_config.sparsification.epochs,
        learning_rate=sparse_config.sparsification.learning_rate,
        batch_size=sparse_config.sparsification.batch_size,
        method=method,
    )
    if model.encoder.gates is not None:
        model.encoder.collapse_gates()

    dense_S = model.encoder_matrix().detach().float().cpu().numpy()
    mask = threshold_encoder_mask(dense_S, threshold=sparse_config.sparsification.threshold)
    apply_encoder_mask(model, mask)
    sparse_S_before_refit = model.encoder_matrix().detach().float().cpu().numpy()

    refit_history = _train_sparse_epochs(
        model,
        sparse_config,
        data=data,
        device=device,
        epochs=sparse_config.sparsification.masked_refit_epochs,
        learning_rate=sparse_config.sparsification.masked_refit_learning_rate,
        batch_size=sparse_config.sparsification.batch_size,
        method="masked_refit",
        mask=mask,
    )
    apply_encoder_mask(model, mask)
    final_S = model.encoder_matrix().detach().float().cpu().numpy()

    train_metrics = evaluate_model(
        model,
        data.X_train,
        data.y_train,
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
    formula_report = encoder_formula_report(
        model,
        sparse_config.invariants.selected,
        threshold=sparse_config.sparsification.threshold,
    )

    checkpoint_out = experiment_dir / "checkpoint_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": to_jsonable(sparse_config),
            "metrics": test_metrics,
            "parameter_counts": model.parameter_counts(),
            "invariant_normalization": model.invariant_normalization_state(),
            "source_checkpoint": str(checkpoint_path),
        },
        checkpoint_out,
    )
    mask_path = save_json(
        experiment_dir / "adaptive_stage2_mask.json",
        {
            "threshold": sparse_config.sparsification.threshold,
            "mask": mask.astype(int).tolist(),
            "active_counts": mask.sum(axis=1).astype(int).tolist(),
        },
    )
    sparse_history_path = None
    refit_history_path = None
    if sparse_config.sparsification.save_training_logs:
        sparse_history_path = save_json(
            experiment_dir / "adaptive_stage2_sparse_history.json",
            {
                "stage": "sparsity_training",
                "method": method,
                "history": sparse_history,
            },
        )
        refit_history_path = save_json(
            experiment_dir / "adaptive_stage2_refit_history.json",
            {
                "stage": "masked_refit",
                "method": "masked_refit",
                "history": refit_history,
            },
        )
    summary_path = save_json(
        experiment_dir / sparse_config.sparsification.summary_name,
        {
            "source_checkpoint": str(checkpoint_path),
            "method": method,
            "threshold": sparse_config.sparsification.threshold,
            "source_S": source_S.tolist(),
            "trained_dense_S": dense_S.tolist(),
            "sparse_S_before_refit": sparse_S_before_refit.tolist(),
            "final_sparse_S": final_S.tolist(),
            "mask": mask.astype(int).tolist(),
            "active_counts": mask.sum(axis=1).astype(int).tolist(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "sparse_history": sparse_history,
            "refit_history": refit_history,
            "sparse_history_path": None if sparse_history_path is None else str(sparse_history_path),
            "refit_history_path": None if refit_history_path is None else str(refit_history_path),
            "formulas": formula_report,
            "checkpoint": str(checkpoint_out),
            "mask_path": str(mask_path),
        },
    )
    return SparsifyResult(
        experiment_dir=experiment_dir,
        checkpoint_path=checkpoint_out,
        summary_path=summary_path,
        mask_path=mask_path,
        sparse_history_path=sparse_history_path,
        refit_history_path=refit_history_path,
    )
