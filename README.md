# Invariant Generator

PyTorch implementation of the invariant-generator pipeline for yield-surface
prediction from stress components.

The default input order is:

```text
[s11, s22, s33, s23, s13, s12]
```

Run training with:

```bash
uv run python scripts/train.py --config configs/default.toml
```

After training, run symbolic regression with PySR on the most important
pre-encoder invariant columns selected from the learned encoder `S` weights:

```bash
uv run --extra symbolic python scripts/train_pysr.py --config configs/default.toml
```

This PySR stage uses invariant values as inputs, not raw stress components and
not the encoded `S I` features. By default it fits the same homogeneous
yield-surface targets used during training. Automatic PySR feature selection
uses scale-aware encoder scores by default:

```text
score_j = ||S[:, j]||_2 * std(I_j)
```

Set `symbolic.feature_selection = "manual"` and provide
`symbolic.selected_invariants` to override this without changing training.

Optional physics constraints are configured separately from invariant selection.
For example, to require the fourth-order structural tensor `A` to be positive
semi-definite in a shear-aware Mandel basis:

```toml
[constraints.A_psd]
enabled = true
mode = "hard"  # hard | penalty | check
target = "fourth_order_A"
basis = "mandel"
```

Constraints do not select invariants or change PySR operators; they only enforce
or report the configured physical property.

The default loss follows the attached notes:

```text
L = L_data + L_param + L_structure + L_enc + L_constraint
```

where `L_data` is the sum of squared errors against the homogeneous scaling
target, and the structure/encoder terms normalize the trainable tensors and
encourage sparsity in the encoder. `L_constraint` is zero unless a constraint
uses penalty mode.
