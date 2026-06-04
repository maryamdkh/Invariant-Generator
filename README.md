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

The default loss follows the attached notes:

```text
L = L_data + L_param + L_structure + L_enc
```

where `L_data` is the sum of squared errors against the homogeneous scaling
target, and the structure/encoder terms normalize the trainable tensors and
encourage sparsity in the encoder.
