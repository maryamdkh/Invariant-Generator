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
  --checkpoint results/adaptive_rotatedhill/stage1/n03/checkpoint_best.pt
```

Run Stage 3 from a sparse checkpoint:

```bash
uv run --extra symbolic python scripts/adaptive_encoder_pipeline.py \
  --config configs/adaptive_encoder_rotated_hill.toml \
  --stage stage3 \
  --selected-n 3 \
  --checkpoint results/adaptive_rotatedhill/stage2_sparse/checkpoint_best.pt
```

## Stage 1: Adaptive Dimension Sweep

The script trains independent full models while adding one encoder output at a
time:

```text
n = n_min, n_min + 1, ..., n_max
```

Each run uses the same invariant pool, structural tensors, normalization, and
MLP settings. When `n` is smaller than the invariant-pool dimension, encoder
initialization is switched to random initialization to avoid identity bias
toward the first invariant columns.

The selected `n` is the first run whose train and test metric both pass the
configured absolute threshold:

```toml
[adaptive]
metric = "rmse" # or "mse"
rmse_threshold = 1e-3
mse_threshold = 1e-6
max_generalization_gap = 1e-3
patience = 2
```

`patience` is the number of additional consecutive dimensions that must also
pass before confirming the first passing `n`. For example, `patience = 2`
selects `n` only after `n`, `n + 1`, and `n + 2` all pass. Use `patience = 0`
to select the first passing dimension immediately.

Outputs:

- `results/adaptive_rotatedhill/stage1/adaptive_stage1_summary.json`
- `results/adaptive_rotatedhill/stage1/adaptive_stage1_<metric>_vs_n.png`
- per-dimension runs such as `results/adaptive_rotatedhill/stage1/n01/`
- each per-dimension run saves its full training log at
  `results/adaptive_rotatedhill/stage1/nXX/history.json`, and the Stage 1 summary
  links those paths for notebook plotting

## Stage 2: Sparsify Terms Inside S

Stage 2 starts from the selected Stage 1 checkpoint and keeps the encoder
output dimension fixed. It supports two sparsification methods:

- `method = "lasso"` adds elementwise `||S||_1` pressure.
- `method = "gated"` adds trainable gates on encoder entries and penalizes gate
  activity.

After sparsity training, small entries are thresholded and a masked refit keeps
those entries exactly zero while other trainable parameters adapt.

For direct control over formula size, set:

```toml
[sparsification]
lambda_encoder_l1 = 1e-2
threshold = 1e-3
max_active_terms_per_row = 4
```

The cap is applied after thresholding and keeps the strongest coefficients in
each encoded invariant row. Set it to `0` to disable the cap. To explore the
accuracy/sparsity tradeoff, rerun Stage 2 and Stage 3 with caps such as `4`,
`3`, and `2`; Stage 1 does not need to be rerun.

Outputs:

- `results/adaptive_rotatedhill/stage2_sparse/adaptive_stage2_sparsify.json`
- `results/adaptive_rotatedhill/stage2_sparse/adaptive_stage2_mask.json`
- `results/adaptive_rotatedhill/stage2_sparse/adaptive_stage2_sparse_history.json`
- `results/adaptive_rotatedhill/stage2_sparse/adaptive_stage2_refit_history.json`
- `results/adaptive_rotatedhill/stage2_sparse/checkpoint_best.pt`

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

- `results/adaptive_rotatedhill/stage3_pysr/encoded_invariant_formulas.json`
- `results/adaptive_rotatedhill/stage3_pysr/equations.csv`
- `results/adaptive_rotatedhill/stage3_pysr/best_equation.txt`
- `results/adaptive_rotatedhill/stage3_pysr/metrics.json`

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
