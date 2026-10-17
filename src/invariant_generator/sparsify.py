from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from invariant_generator.adaptive import (
    adaptive_metric_threshold,
    adaptive_run_passes,
    adaptive_sparsification_run_id,
    with_selection_metric,
)
from invariant_generator.config import Config
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.formulas import encoder_formula_report
from invariant_generator.losses import YieldSurfaceLoss
from invariant_generator.model import InvariantYieldModel
from invariant_generator.input_features import input_feature_names
from invariant_generator.train import (
    _make_train_noise_generator,
    _optimizer_param_groups,
    _train_input_noise_feature_scale,
    apply_train_input_noise,
)
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


def threshold_encoder_mask(
    S: torch.Tensor | np.ndarray,
    *,
    threshold: float,
    max_active_terms_per_row: int = 0,
) -> np.ndarray:
    values = np.asarray(S.detach().cpu().numpy() if isinstance(S, torch.Tensor) else S)
    mask = np.abs(values) > float(threshold)
    max_active = int(max_active_terms_per_row)
    if max_active < 0:
        raise ValueError("max_active_terms_per_row must be >= 0.")

    for row_idx in range(mask.shape[0]):
        if not np.any(mask[row_idx]):
            strongest = int(np.argmax(np.abs(values[row_idx])))
            mask[row_idx, strongest] = True
        if max_active > 0 and int(mask[row_idx].sum()) > max_active:
            active_indices = np.flatnonzero(mask[row_idx])
            ranked = active_indices[
                np.argsort(np.abs(values[row_idx, active_indices]))[::-1]
            ]
            mask[row_idx] = False
            mask[row_idx, ranked[:max_active]] = True
    return mask


def candidate_support_masks(
    S: torch.Tensor | np.ndarray,
    *,
    cap: int,
    threshold: float,
    support_search: str = "magnitude",
    max_candidates: int = 256,
) -> list[np.ndarray]:
    """Return the encoder masks Stage 2 refits for one active-term cap."""
    values = np.asarray(S.detach().cpu().numpy() if isinstance(S, torch.Tensor) else S)
    support_search = support_search.lower()
    if support_search == "magnitude":
        return [
            threshold_encoder_mask(values, threshold=threshold, max_active_terms_per_row=cap)
        ]
    if support_search != "exhaustive":
        raise ValueError("sparsification.support_search must be 'magnitude' or 'exhaustive'.")

    n_rows, n_cols = values.shape
    if cap <= 0 or cap >= n_cols:
        return [np.ones(values.shape, dtype=bool)]
    row_supports = list(itertools.combinations(range(n_cols), int(cap)))
    # Encoded rows are interchangeable inputs to the MLP, so row order is irrelevant.
    total = math.comb(len(row_supports) + n_rows - 1, n_rows)
    if total > int(max_candidates):
        raise ValueError(
            f"Exhaustive support search needs {total} refits for cap={cap}, exceeding "
            f"sparsification.max_support_candidates={max_candidates}. Use "
            "support_search='magnitude', smaller caps, or a smaller feature pool."
        )
    masks: list[np.ndarray] = []
    for combination in itertools.combinations_with_replacement(row_supports, n_rows):
        mask = np.zeros(values.shape, dtype=bool)
        for row_idx, support in enumerate(combination):
            mask[row_idx, list(support)] = True
        masks.append(mask)
    return masks


def canonical_encoder_rows(B: np.ndarray) -> np.ndarray:
    """Reduced row-echelon form of the encoder rows, with full pivoting.

    For n > 1 the MLP can undo any invertible mixing of J, so only the row
    space of B is identifiable. RREF picks one basis of it: each row has a 1
    in its own pivot column and 0 in the other rows' pivot columns.
    """
    R = np.asarray(B, dtype=np.float64).copy()
    n_rows, n_cols = R.shape
    pivots: list[int] = []
    for row in range(n_rows):
        free = [col for col in range(n_cols) if col not in pivots]
        block = np.abs(R[row:, free])
        pivot_row, pivot_free = np.unravel_index(int(np.argmax(block)), block.shape)
        pivot_row += row
        pivot_col = free[pivot_free]
        R[[row, pivot_row]] = R[[pivot_row, row]]
        R[row] /= R[row, pivot_col]
        for other in range(n_rows):
            if other != row:
                R[other] -= R[other, pivot_col] * R[row]
        pivots.append(pivot_col)
    return R


def sparsest_integer_basis(rows: np.ndarray, *, max_coefficient: int = 3) -> np.ndarray:
    """Sparsest basis of the row space spanned by rational exponent rows.

    The MLP can undo any invertible mix of the encoded variables, so any basis
    of the row space carries the same information; sparser groups are simpler
    for symbolic regression. Candidates are integer combinations of the rows
    with coefficients in [-max_coefficient, max_coefficient], each reduced to
    a primitive integer vector (first nonzero entry positive). They are ranked
    by (number of nonzero exponents, sum of |exponents|), and a basis is built
    greedily, which is optimal for an additive score because linearly
    independent sets form a matroid.
    """
    from fractions import Fraction
    from math import gcd, lcm

    rows = np.asarray(rows, dtype=np.float64)
    n_rows = rows.shape[0]
    integer_rows = []
    for row in rows:
        fractions = [Fraction(float(v)).limit_denominator(1000) for v in row]
        scale = lcm(*[f.denominator for f in fractions]) if fractions else 1
        integer_rows.append([int(f * scale) for f in fractions])
    integer_rows = np.asarray(integer_rows, dtype=np.int64)

    candidates: dict[tuple[int, ...], tuple[int, int]] = {}
    for coefficients in itertools.product(range(-max_coefficient, max_coefficient + 1), repeat=n_rows):
        if not any(coefficients):
            continue
        vector = np.asarray(coefficients, dtype=np.int64) @ integer_rows
        nonzero = vector[vector != 0]
        if nonzero.size == 0:
            continue
        divisor = 0
        for value in nonzero:
            divisor = gcd(divisor, int(abs(value)))
        vector = vector // divisor
        if vector[np.flatnonzero(vector)[0]] < 0:
            vector = -vector
        candidates[tuple(int(v) for v in vector)] = (int(np.count_nonzero(vector)), int(np.abs(vector).sum()))

    basis: list[np.ndarray] = []
    for vector, _ in sorted(candidates.items(), key=lambda item: (item[1], item[0])):
        trial = np.vstack(basis + [np.asarray(vector, dtype=np.float64)])
        if np.linalg.matrix_rank(trial) > len(basis):
            basis.append(np.asarray(vector, dtype=np.float64))
        if len(basis) == n_rows:
            break
    return np.vstack(basis)


def snap_encoder_rows(R: np.ndarray, *, denominator: int) -> tuple[np.ndarray, float]:
    """Rescale each row and round it to multiples of 1/denominator.

    A row's scale is free (the MLP absorbs it), so each row is divided by the
    nonzero entry that makes it closest to the rational grid. Returns the
    snapped rows and the largest rounding residual.
    """
    R = np.asarray(R, dtype=np.float64)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("Snap denominators must be positive integers.")
    snapped = np.zeros_like(R)
    worst = 0.0
    for row_idx, row in enumerate(R):
        best: tuple[float, np.ndarray] | None = None
        for value in row[np.abs(row) > 1e-8 * np.abs(row).max()]:
            scaled = row / value
            rounded = np.round(scaled * denominator) / denominator
            residual = float(np.max(np.abs(scaled - rounded)))
            if best is None or residual < best[0]:
                best = (residual, rounded)
        assert best is not None
        worst = max(worst, best[0])
        snapped[row_idx] = best[1]
    return snapped, worst


def graded_encoder_rows(
    B: np.ndarray,
    degrees: np.ndarray,
    *,
    tolerance: float = 1e-2,
) -> tuple[np.ndarray, float] | None:
    """Basis of the row space whose rows each use features of one degree only.

    For polynomial feature maps graded by degree (e.g. quadratic and cubic
    stress invariants), a target like f(J2, J3) has a row space that splits
    into homogeneous parts, but the learned rows can be any mixture. For each
    degree d, combinations c with ||c^T B[:, deg != d]|| small (left singular
    vectors of the off-degree block) give rows supported on degree d.
    Returns (rows, largest relative off-degree residual), or None when the
    row space does not split into exactly n homogeneous rows.
    """
    B = np.asarray(B, dtype=np.float64)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-300)
    degrees = np.asarray(degrees)
    n_rows = B.shape[0]
    rows: list[np.ndarray] = []
    worst = 0.0
    for degree in np.unique(degrees):
        off = B[:, degrees != degree]
        U, singular, _ = np.linalg.svd(off, full_matrices=True)
        singular = np.concatenate([singular, np.zeros(n_rows - singular.size)])
        for index in np.flatnonzero(singular < tolerance):
            combination = U[:, index]
            row = combination @ B
            residual = float(np.linalg.norm(row[degrees != degree]) / np.linalg.norm(row))
            row[degrees != degree] = 0.0
            rows.append(row / np.abs(row).max())
            worst = max(worst, residual)
    if len(rows) != n_rows:
        return None
    return np.vstack(rows), worst


def _snap_encoder_from_checkpoint(
    config: Config,
    *,
    checkpoint_path: str | Path,
) -> SparsifyResult:
    """Stage 2 with a frozen, interpretable encoder basis, then an MLP refit.

    method = "snap": power-law encoders; snap exponents to simple fractions.
    method = "graded": polynomial feature maps; one homogeneous row per
    degree component of the learned row space.
    """
    from fractions import Fraction

    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    data = prepare_training_data(config)
    feature_names = input_feature_names(config)
    model = load_model_for_checkpoint(config, checkpoint_path).to(device)
    if model.encoder is None:
        raise ValueError("Encoder snapping requires an enabled encoder.")
    experiment_dir = config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, experiment_dir)

    metric, metric_threshold = adaptive_metric_threshold(config)
    report = encoder_formula_report(model, feature_names)
    source_raw = np.asarray(report["raw_coefficients"], dtype=np.float64)
    canonical = canonical_encoder_rows(source_raw)
    std = (
        model.normalizer.std.detach().cpu().numpy().astype(np.float64)
        if model.normalizer is not None
        else np.ones(source_raw.shape[1])
    )
    source_state = deepcopy(model.state_dict())

    def evaluate() -> tuple[dict[str, float], dict[str, float]]:
        train = evaluate_model(model, data.X_train, data.y_train, device=device, batch_size=8192)
        selection = evaluate_model(
            model, data.X_selection, data.y_selection, device=device, batch_size=8192,
        )
        return (
            with_selection_metric(train, data.y_train, metric=metric),
            with_selection_metric(selection, data.y_selection, metric=metric),
        )

    source_train, source_test = evaluate()
    candidate_results: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    selected_state: dict[str, torch.Tensor] | None = None
    seen: set[bytes] = set()
    method = config.sparsification.method.lower()
    basis = config.sparsification.snap_basis.lower()
    if basis not in {"rref", "sparsest"}:
        raise ValueError("sparsification.snap_basis must be 'rref' or 'sparsest'.")

    def candidate_bases():
        if method == "graded":
            pool = model.invariant_pool
            if not hasattr(pool, "terms"):
                raise ValueError("method='graded' needs a polynomial feature map with degrees.")
            graded = graded_encoder_rows(
                source_raw, np.array([len(term) for term in pool.terms]),
            )
            if graded is not None:
                yield "graded", graded[0], graded[1]
            return
        for denominator in config.sparsification.snap_denominators:
            snapped, residual = snap_encoder_rows(canonical, denominator=int(denominator))
            if basis == "sparsest":
                # Same row space, so the same information; only the groups change.
                snapped = sparsest_integer_basis(snapped)
            yield int(denominator), snapped, residual

    if method == "graded":
        with torch.no_grad():
            raw_features = model.raw_invariant_features(
                torch.as_tensor(data.X_train, dtype=torch.float32, device=device)
            ).cpu().numpy().astype(np.float64)

    def frozen_refit(rows: np.ndarray, label: object):
        """Freeze the encoder at `rows` (raw coefficients) and refit the MLP."""
        model.load_state_dict(source_state)
        with torch.no_grad():
            model.encoder.raw_weight.copy_(
                torch.as_tensor(rows * std[None, :], dtype=model.encoder.raw_weight.dtype)
            )
        model.encoder.raw_weight.requires_grad_(False)
        history = _train_sparse_epochs(
            model,
            config,
            data=data,
            device=device,
            epochs=config.sparsification.masked_refit_epochs,
            learning_rate=config.sparsification.masked_refit_learning_rate,
            batch_size=config.sparsification.batch_size,
            method="masked_refit",
            desc=f"stage2 frozen-basis refit ({label})",
            schedule=config.sparsification.masked_refit_schedule,
            min_learning_rate=config.sparsification.masked_refit_min_learning_rate,
        )
        model.encoder.raw_weight.requires_grad_(True)
        return (*evaluate(), history)

    acceptance = config.sparsification.snap_acceptance.lower()
    if acceptance not in {"absolute", "relative_to_dense"}:
        raise ValueError("sparsification.snap_acceptance must be 'absolute' or 'relative_to_dense'.")
    dense_refit: dict[str, object] | None = None
    if acceptance == "relative_to_dense":
        # Refit the unsnapped rows with the same frozen procedure, so the test
        # isolates the effect of snapping from how well the refit converges.
        dense_rows = canonical
        if method == "graded":
            dense_rows = canonical / (raw_features @ canonical.T).std(axis=0).clip(1e-300)[:, None]
        dense_train, dense_test, _ = frozen_refit(dense_rows, "unsnapped reference")
        attainable = max(float(dense_train[metric]), float(dense_test[metric]))
        metric_threshold = max(
            float(config.adaptive.relative_factor) * attainable, float(config.adaptive.relative_floor),
        )
        dense_refit = {"train_metrics": dense_train, "test_metrics": dense_test,
                       "acceptance_threshold": metric_threshold}

    for denominator, snapped, residual in candidate_bases():
        if method == "graded":
            # Row scale is free; unit spread keeps J of different degrees on a
            # comparable scale for the refit MLP.
            snapped = snapped / (raw_features @ snapped.T).std(axis=0).clip(1e-300)[:, None]
        if snapped.tobytes() in seen:
            continue
        seen.add(snapped.tobytes())
        train_metrics, test_metrics, history = frozen_refit(snapped, denominator)
        passes = adaptive_run_passes(
            train_metrics,
            test_metrics,
            metric=metric,
            threshold=metric_threshold,
            max_generalization_gap=config.adaptive.max_generalization_gap,
        )
        candidate = {
            "denominator": denominator,
            "rounding_residual": residual,
            # Exact rationals only for snapped exponents; graded rows stay real.
            "exponents": None if method == "graded" else [
                [str(Fraction(float(v)).limit_denominator(int(denominator))) for v in row]
                for row in snapped
            ],
            "rows": snapped.tolist(),
            "passes": passes,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "refit_history": history,
        }
        candidate_results.append(candidate)
        if passes:
            selected, selected_state = candidate, deepcopy(model.state_dict())
            break

    if selected is None:
        # No snapped basis fits: keep the dense Stage 1 encoder as the reduction.
        model.load_state_dict(source_state)
    else:
        model.load_state_dict(selected_state)

    formula_report = encoder_formula_report(model, feature_names)
    final_S = model.encoder_matrix().detach().float().cpu().numpy()
    checkpoint_out = experiment_dir / "checkpoint_best.pt"
    exact = None if selected is None else selected["exponents"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": to_jsonable(config),
            "metrics": source_test if selected is None else selected["test_metrics"],
            "parameter_counts": model.parameter_counts(),
            "invariant_normalization": model.invariant_normalization_state(),
            "source_checkpoint": str(checkpoint_path),
            "exact_raw_coefficients": exact,
        },
        checkpoint_out,
    )
    mask = (np.abs(final_S) > 0).astype(int)
    mask_path = save_json(
        experiment_dir / "adaptive_stage2_mask.json",
        {"mask": mask.tolist(), "active_counts": mask.sum(axis=1).tolist()},
    )
    summary_path = save_json(
        experiment_dir / config.sparsification.summary_name,
        {
            "source_checkpoint": str(checkpoint_path),
            "method": method,
            "selection_metric": metric,
            "selection_partition": "validation" if config.data.kind == "supervised" else "test",
            "selection_metric_threshold": metric_threshold,
            "source_raw_coefficients": source_raw.tolist(),
            "canonical_raw_coefficients": canonical.tolist(),
            "source_train_metrics": source_train,
            "source_test_metrics": source_test,
            "snapped": selected is not None,
            "snap_acceptance": acceptance,
            "dense_refit": dense_refit,
            "selected_denominator": None if selected is None else selected["denominator"],
            "exact_raw_coefficients": exact,
            "candidate_results": candidate_results,
            "final_sparse_S": final_S.tolist(),
            "mask": mask.tolist(),
            "active_counts": mask.sum(axis=1).tolist(),
            "train_metrics": source_train if selected is None else selected["train_metrics"],
            "test_metrics": source_test if selected is None else selected["test_metrics"],
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
        sparse_history_path=None,
        refit_history_path=None,
    )


def active_term_candidates(config: Config) -> list[int]:
    """Return Stage 2 active-term caps to evaluate."""
    if config.sparsification.adaptive_max_active_terms:
        candidates = list(config.sparsification.max_active_terms_candidates)
        if not candidates:
            input_dim = len(input_feature_names(config))
            max_terms = min(4, input_dim) if config.input_transform.kind == "pairwise" else input_dim
            candidates = list(range(1, max_terms + 1))
    else:
        candidates = [int(config.sparsification.max_active_terms_per_row)]

    normalized: list[int] = []
    for value in candidates:
        cap = int(value)
        if cap < 0:
            raise ValueError("max active term candidates must be >= 0.")
        if cap not in normalized:
            normalized.append(cap)
    if not normalized:
        raise ValueError("At least one max active term candidate is required.")
    return normalized


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
    desc: str | None = None,
    schedule: str = "constant",
    min_learning_rate: float = 0.0,
) -> list[dict[str, float | int]]:
    if epochs <= 0:
        return []

    criterion = YieldSurfaceLoss(config.loss, config.constraints)
    optimizer = torch.optim.Adam(
        _optimizer_param_groups(model, config, learning_rate=float(learning_rate)),
        lr=float(learning_rate),
    )
    schedule = schedule.lower()
    if schedule not in {"constant", "cosine"}:
        raise ValueError("sparsification.masked_refit_schedule must be 'constant' or 'cosine'.")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(epochs), eta_min=float(min_learning_rate),
        )
        if schedule == "cosine"
        else None
    )
    loader = _make_loader(data.X_train, data.y_train, batch_size=batch_size, shuffle=True)
    train_noise = config.train_input_noise
    train_noise_feature_scale_np = _train_input_noise_feature_scale(
        data.X_train,
        relative_to_feature_std=train_noise.relative_to_feature_std,
    )
    train_noise_feature_scale = torch.as_tensor(
        train_noise_feature_scale_np,
        dtype=torch.float32,
        device=device,
    )
    train_noise_generator = _make_train_noise_generator(
        seed=train_noise.random_state,
        device=device,
    )
    history: list[dict[str, float | int]] = []
    progress_desc = desc or f"stage2 {method}"
    progress = trange(1, int(epochs) + 1, desc=progress_desc)

    for epoch in progress:
        model.train()
        total = 0.0
        data_loss = 0.0
        sparsity_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_batch = apply_train_input_noise(
                X_batch,
                enabled=train_noise.enabled,
                scale=train_noise.scale,
                probability=train_noise.probability,
                feature_scale=train_noise_feature_scale,
                generator=train_noise_generator,
            )
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
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "loss_total": total,
            "loss_data": data_loss,
            "loss_sparsity": sparsity_loss,
        }
        history.append(row)
        progress.set_postfix(
            {
                "loss": f"{total:.4g}",
                "data": f"{data_loss:.4g}",
                "sparse": f"{sparsity_loss:.4g}",
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
    sparse_config.train.run_id = run_id or adaptive_sparsification_run_id(sparse_config)
    if sparse_config.sparsification.method.lower() in {"snap", "graded"}:
        return _snap_encoder_from_checkpoint(sparse_config, checkpoint_path=checkpoint_path)
    seed_everything(sparse_config.train.seed)
    device = resolve_device(sparse_config.train.device)
    data = prepare_training_data(sparse_config)
    feature_names = input_feature_names(sparse_config)
    model = load_model_for_checkpoint(sparse_config, checkpoint_path).to(device)
    if model.encoder is None:
        raise ValueError("Adaptive sparsification requires an enabled encoder.")

    experiment_dir = sparse_config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(sparse_config, experiment_dir)

    source_S = model.encoder_matrix().detach().float().cpu().numpy().copy()
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
        desc=f"stage2 sparsity ({method})",
    )
    if model.encoder.gates is not None:
        model.encoder.collapse_gates()

    dense_S = model.encoder_matrix().detach().float().cpu().numpy().copy()
    dense_state = deepcopy(model.state_dict())
    metric, metric_threshold = adaptive_metric_threshold(sparse_config)
    candidates = active_term_candidates(sparse_config)
    candidate_results: list[dict[str, object]] = []
    candidate_states: list[dict[str, torch.Tensor]] = []
    selected_candidate: dict[str, object] | None = None
    selected_state: dict[str, torch.Tensor] | None = None

    support_search = sparse_config.sparsification.support_search.lower()

    for cap in candidates:
        masks = candidate_support_masks(
            dense_S,
            cap=cap,
            threshold=sparse_config.sparsification.threshold,
            support_search=support_search,
            max_candidates=sparse_config.sparsification.max_support_candidates,
        )
        support_results: list[dict[str, object]] = []
        best_support: tuple[float, dict[str, object]] | None = None
        for support_idx, mask in enumerate(masks):
            model.load_state_dict(dense_state)
            apply_encoder_mask(model, mask)
            sparse_S_before_refit = model.encoder_matrix().detach().float().cpu().numpy().copy()
            desc = f"stage2 masked refit (cap={cap})"
            if len(masks) > 1:
                desc += f" support {support_idx + 1}/{len(masks)}"

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
                desc=desc,
                schedule=sparse_config.sparsification.masked_refit_schedule,
                min_learning_rate=sparse_config.sparsification.masked_refit_min_learning_rate,
            )
            apply_encoder_mask(model, mask)
            final_S = model.encoder_matrix().detach().float().cpu().numpy().copy()

            train_metrics = with_selection_metric(
                evaluate_model(model, data.X_train, data.y_train, device=device, batch_size=8192),
                data.y_train,
                metric=metric,
            )
            test_metrics = with_selection_metric(
                evaluate_model(
                    model, data.X_selection, data.y_selection, device=device, batch_size=8192,
                ),
                data.y_selection,
                metric=metric,
            )
            if len(masks) > 1:
                support_results.append({
                    "active_terms": [
                        [feature_names[j] for j in np.flatnonzero(row)] for row in mask
                    ],
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                })
            score = float(test_metrics[metric])
            if best_support is None or score < best_support[0]:
                best_support = (score, {
                    "mask": mask,
                    "sparse_S_before_refit": sparse_S_before_refit,
                    "final_S": final_S,
                    "refit_history": refit_history,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "state": deepcopy(model.state_dict()),
                })

        assert best_support is not None
        chosen = best_support[1]
        model.load_state_dict(chosen["state"])
        mask = chosen["mask"]
        sparse_S_before_refit = chosen["sparse_S_before_refit"]
        final_S = chosen["final_S"]
        refit_history = chosen["refit_history"]
        train_metrics = chosen["train_metrics"]
        test_metrics = chosen["test_metrics"]
        formula_report = encoder_formula_report(
            model,
            feature_names,
            threshold=sparse_config.sparsification.threshold,
        )
        passes = adaptive_run_passes(
            train_metrics,
            test_metrics,
            metric=metric,
            threshold=metric_threshold,
            max_generalization_gap=sparse_config.adaptive.max_generalization_gap,
        )
        refit_history_path = None
        if sparse_config.sparsification.save_training_logs:
            suffix = "uncapped" if cap == 0 else f"cap{cap:02d}"
            refit_history_path = save_json(
                experiment_dir / f"adaptive_stage2_refit_{suffix}_history.json",
                {
                    "stage": "masked_refit",
                    "method": "masked_refit",
                    "max_active_terms_per_row": cap,
                    "history": refit_history,
                },
            )
        candidate = {
            "max_active_terms_per_row": cap,
            "threshold": sparse_config.sparsification.threshold,
            "metric": metric,
            "metric_threshold": metric_threshold,
            "max_generalization_gap": sparse_config.adaptive.max_generalization_gap,
            "passes": passes,
            "sparse_S_before_refit": sparse_S_before_refit.tolist(),
            "final_sparse_S": final_S.tolist(),
            "mask": mask.astype(int).tolist(),
            "active_counts": mask.sum(axis=1).astype(int).tolist(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "refit_history": refit_history,
            "refit_history_path": None
            if refit_history_path is None
            else str(refit_history_path),
            "formulas": formula_report,
        }
        if support_results:
            candidate["support_candidates"] = sorted(
                support_results,
                key=lambda result: float(result["test_metrics"][metric]),  # type: ignore[index]
            )
        candidate_results.append(candidate)
        candidate_states.append(deepcopy(model.state_dict()))

        if (passes or not sparse_config.sparsification.adaptive_max_active_terms) and (
            selected_candidate is None
        ):
            selected_candidate = candidate
            selected_state = candidate_states[-1]
            if sparse_config.sparsification.adaptive_max_active_terms:
                break

    if selected_candidate is None:
        selected_index = min(
            range(len(candidate_results)),
            key=lambda idx: float(
                candidate_results[idx]["test_metrics"][metric]  # type: ignore[index]
            ),
        )
        selected_candidate = candidate_results[selected_index]
        selected_state = candidate_states[selected_index]

    model.load_state_dict(selected_state)
    selected_cap = int(selected_candidate["max_active_terms_per_row"])
    mask = np.asarray(selected_candidate["mask"], dtype=int)
    sparse_S_before_refit = np.asarray(
        selected_candidate["sparse_S_before_refit"],
        dtype=float,
    )
    final_S = np.asarray(selected_candidate["final_sparse_S"], dtype=float)
    train_metrics = selected_candidate["train_metrics"]
    test_metrics = selected_candidate["test_metrics"]
    formula_report = selected_candidate["formulas"]
    refit_history = selected_candidate["refit_history"]

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
            "max_active_terms_per_row": selected_cap,
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
                "max_active_terms_per_row": selected_cap,
                "history": refit_history,
            },
        )
    summary_path = save_json(
        experiment_dir / sparse_config.sparsification.summary_name,
        {
            "source_checkpoint": str(checkpoint_path),
            "method": method,
            "support_search": support_search,
            "threshold": sparse_config.sparsification.threshold,
            "adaptive_max_active_terms": sparse_config.sparsification.adaptive_max_active_terms,
            "max_active_terms_candidates": candidates,
            "max_active_terms_per_row": selected_cap,
            "selected_max_active_terms_per_row": selected_cap,
            "selection_metric": metric,
            "selection_partition": "validation" if sparse_config.data.kind == "supervised" else "test",
            "selection_metric_threshold": metric_threshold,
            "selection_max_generalization_gap": sparse_config.adaptive.max_generalization_gap,
            "selected_candidate_passes": bool(selected_candidate["passes"]),
            "candidate_results": candidate_results,
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
