# Adaptive Encoder Invariant Discovery

This document describes the second invariant-discovery approach. It is separate
from the existing broad-training, top-k prune/refit, and PySR workflow.

## Goal

Find the smallest encoder output dimension `n` that can fit and validate the
yield surface, then keep that `n` fixed while reducing the number of active
terms inside each learned invariant.

The model form is:

```text
f_hat(sigma) = NN(J1, J2, ..., Jn)
J_i = sum_j S_ij z_j
```

where `z_j` is either the normalized invariant input or, after conversion, the
same expression can be written directly in terms of raw invariants `I_j`.

## Commands

Run the complete adaptive pipeline:

```bash
uv run python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_rotated_hill.toml
```

Run only Stage 1:

```bash
uv run python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_rotated_hill.toml --stage stage1
```

Run Stage 2 from a chosen Stage 1 checkpoint:

```bash
uv run python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_rotated_hill.toml \
  --stage stage2 \
  --selected-n 3 \
  --checkpoint results/adaptive_rotatedhill_n03/checkpoint_best.pt
```

Run Stage 3 from a sparse checkpoint:

```bash
uv run --extra symbolic python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_rotated_hill.toml \
  --stage stage3 \
  --selected-n 3 \
  --checkpoint results/adaptive_rotatedhill_sparse/checkpoint_best.pt
```

## Stage 1: Adaptive Dimension Sweep

The script trains independent full models across encoder output dimensions. The
direction is configurable:

```text
forward:  n = n_min, ..., n_max
backward: n = n_max, ..., n_min
```

Each run uses the same invariant pool, structural tensors, normalization, and
MLP settings. When `n` is smaller than the invariant-pool dimension, encoder
initialization is switched to random initialization to avoid identity bias
toward the first invariant columns.

In forward mode, the selected `n` is the first run whose train and test metric
both pass the configured absolute threshold:

```toml
[adaptive]
search_direction = "forward"
metric = "rmse" # or "mse"
rmse_threshold = 1e-3
mse_threshold = 1e-6
max_generalization_gap = 1e-3
```

In backward mode, the largest `n` is trained first as the reference. The sweep
then tries smaller dimensions and keeps the smallest `n` whose loss remains
within the allowed delta from that max-dim reference:

```toml
[adaptive]
search_direction = "backward"
metric = "rmse" # or "mse"
max_loss_delta = 1e-4
max_relative_loss_delta = 0.05
patience = 2
```

`patience` is the number of consecutive failing smaller dimensions tolerated
before stopping. This is useful because independent neural trainings are not
perfectly monotonic: one dimension can fail while the next lower dimension may
still train well.

Outputs:

- `results/adaptive_rotatedhill/adaptive_stage1_summary.json`
- `results/adaptive_rotatedhill/adaptive_stage1_<metric>_vs_n.png`
- per-dimension runs such as `results/adaptive_rotatedhill_n01/`
- each per-dimension run saves its full training log at
  `results/adaptive_rotatedhill_nXX/history.json`, and the Stage 1 summary
  links those paths for notebook plotting

## Stage 2: Sparsify Terms Inside S

Stage 2 starts from the selected Stage 1 checkpoint and keeps the encoder
output dimension fixed. It supports two sparsification methods:

- `method = "lasso"` adds elementwise `||S||_1` pressure.
- `method = "gated"` adds trainable gates on encoder entries and penalizes gate
  activity.

After sparsity training, small entries are thresholded and a masked refit keeps
those entries exactly zero while other trainable parameters adapt.

Outputs:

- `results/adaptive_rotatedhill_sparse/adaptive_stage2_sparsify.json`
- `results/adaptive_rotatedhill_sparse/adaptive_stage2_mask.json`
- `results/adaptive_rotatedhill_sparse/adaptive_stage2_sparse_history.json`
- `results/adaptive_rotatedhill_sparse/adaptive_stage2_refit_history.json`
- `results/adaptive_rotatedhill_sparse/checkpoint_best.pt`

The Stage 2 summary includes:

- source dense `S`
- sparsity-trained dense `S`
- final sparse `S`
- binary active mask
- active term count per encoded invariant
- formulas for `J_i` in normalized and raw invariant coordinates

## Stage 3: PySR on Reduced Encoded Invariants

Stage 3 runs PySR on:

```text
J1, J2, ..., Jn
```

It does not select original `I_j` columns. The formulas saved by Stage 2 and
Stage 3 explain each `J_i` in terms of the original invariant pool, so the PySR
equation remains interpretable.

Outputs:

- `results/adaptive_rotatedhill_sparse/symbolic_encoded/encoded_invariant_formulas.json`
- `results/adaptive_rotatedhill_sparse/symbolic_encoded/equations.csv`
- `results/adaptive_rotatedhill_sparse/symbolic_encoded/best_equation.txt`
- `results/adaptive_rotatedhill_sparse/symbolic_encoded/metrics.json`

## Notebook

Use:

```text
notebooks/adaptive_encoder_analysis.ipynb
```

The notebook visualizes:

- Stage 1 train/test metric vs `n`
- selected minimum `n`
- dense Stage 1 formulas
- sparse Stage 2 formulas
- before/after encoder heatmaps
- active term counts per `J_i`
- prediction quality
- structure tensor diagnostics
- encoded PySR formulas and metrics
