# Project Understanding

This repository implements a PyTorch pipeline for learning yield-surface /
effective-stress functions from stress-component data. The model does not feed
raw stresses directly into a neural network. Instead, it first converts each
stress vector into tensor invariants, optionally learns structural tensors and a
sparse invariant encoder, and then trains a small MLP regressor to predict the
yield-surface value.

## Overall Workflow

1. **Configure an experiment**
   - Experiment settings live in `configs/default.toml`.
   - The config controls dataset location, stress format, augmentation,
     selected invariants, encoder settings, MLP size, loss weights, optional
     physics constraints, symbolic-regression settings, and training behavior.

2. **Load and prepare stress data**
   - Data is loaded from an HDF/H5 file, usually from the `stress` dataset key.
   - Inputs are canonicalized to the 6-component Voigt order:
     `[s11, s22, s33, s23, s13, s12]`.
   - Old 2D plane-stress data `[s11, s22, s12]` can be embedded into the same
     6-component format.
   - The data is split into train/test sets, with optional reuse of saved split
     files in `results/splits`.
   - Training data can be scaled by factors `k` using the homogeneity law
     `f(k*sigma) = |k|^p f(sigma)`, producing matching scaled targets.
   - Optional Gaussian input noise can be added to the training stresses.

3. **Generate invariant features**
   - `InvariantPool` converts each stress vector into a symmetric `3x3` stress
     tensor and computes selected invariants `I1` through `I13`.
   - `I1`, `I2`, and `I3` are basic stress tensor trace invariants.
   - `I4` through `I10` use a learnable second-order structural tensor `a`.
   - `I11` through `I13` use a learnable fourth-order structural tensor `A`.
   - If `homogenize = true`, higher-degree invariants are transformed with a
     signed root so each selected feature scales linearly with stress.
   - Optional constraints can be applied to structural tensors without changing
     which invariants are selected. The implemented example is a PSD constraint
     on the fourth-order tensor `A`, checked in a shear-aware Mandel basis.
     In `hard` mode, `A` is parameterized from a PSD Mandel matrix `L L^T`;
     in `penalty` mode, negative eigenvalues add a loss term; in `check` mode,
     eigenvalues are reported but training is unchanged.

4. **Optionally encode invariants**
   - If `[normalization].enabled = true`, the prepared training set is passed
     through the initial invariant pool once before optimization to compute
     per-invariant mean and standard deviation. These statistics are stored as
     non-trainable model buffers and standardize invariant features before the
     encoder/regressor.
   - A linear encoder `S` can map the selected invariant vector to a smaller
     feature vector.
   - The encoder is intended to learn sparse combinations of invariants.
   - For discovery runs, use `encoder.output_dim = 0` with identity
     initialization to avoid privileging early invariant columns. If a smaller
     encoder is needed, use random initialization.
   - Sparsity is encouraged by the encoder loss term. The column-wise penalty
     `sum_j ||S[:, j]||_2` directly encourages dropping whole invariant
     columns.

5. **Predict the yield-surface value**
   - The final model is:
     `stress -> invariant pool -> optional encoder -> MLP -> prediction`.
   - The MLP is configurable and defaults to hidden layers with SiLU activation
     and a Softplus output so predictions remain non-negative.

6. **Train with the structured loss**
   - Training is launched with:
     `uv run python scripts/train.py --config configs/default.toml`.
   - The loss follows:
     `L = L_data + L_param + L_structure + L_enc + L_constraint`.
   - `L_data` is the sum of squared errors against the homogeneous scaling
     target.
   - `L_param` is L2 regularization on the neural regressor parameters.
   - `L_structure` keeps learned structural tensors near unit norm.
   - `L_enc` encourages sparse, controlled-size encoder weights, including the
     optional column-wise invariant-selection penalty.
   - `L_constraint` is zero unless an enabled constraint uses penalty mode.

7. **Save results and evaluate**
   - Training writes results under `results/<run_id>/`.
   - Saved artifacts include a config snapshot, `history.json`, and
     `checkpoint_best.pt`.
   - `checkpoint_latest.pt` is used only as a rolling recovery checkpoint and is
     removed after successful training.
   - Evaluation is run with `scripts/evaluate.py`, which reloads the config and
     checkpoint, computes regression metrics on the prepared test set, and saves
     `evaluation.json` plus test predictions.
   - Training and evaluation also save configured constraint diagnostics,
     invariant feature statistics, and raw/scale-adjusted encoder scores. For
     the PSD `A` constraint, diagnostics include Mandel eigenvalues, minimum
     eigenvalue, tolerance, and pass/fail status.
   - Checkpoint payloads include invariant-normalization statistics when
     normalization is enabled.

8. **Fit an interpretable PySR equation**
   - After neural training, `scripts/train_pysr.py` loads the best checkpoint and
     selects original pre-encoder invariant columns according to
     `[symbolic].feature_selection`.
   - The default automatic ranking is scale-aware on the actual encoder input:
     `score_j = ||S[:, j]||_2 * std(z_j)`, where `z_j` is the feature seen by
     the encoder. If invariant normalization is enabled, `std(z_j)` should be
     close to one, so raw encoder column norms are directly comparable. Raw
     invariant statistics are still saved for interpretation.
   - `feature_selection = "encoder_norm"` keeps the older raw column-norm
     ranking, while `feature_selection = "manual"` uses
     `[symbolic].selected_invariants` exactly as provided by the user.
   - PySR target values use `[symbolic].target_transform`, currently
     `identity` or `square`. This is user-controlled and is not changed by
     physics constraints.
   - The PySR inputs are invariant values, not raw stress components and not the
     encoded `S I` features.
   - Outputs are saved under `results/<run_id>/symbolic/`, including selected
     invariants, selection scores, diagnostics, the equation table, best equation
     text, and train/test metrics.

## Main Code Map

- `src/invariant_generator/config.py`: dataclass-based config loading.
- `src/invariant_generator/data.py`: HDF loading, stress canonicalization,
  splitting, noise, and homogeneous augmentation.
- `src/invariant_generator/invariants.py`: tensor conversion and invariant
  generation, including learnable structural tensors.
- `src/invariant_generator/constraints.py`: Mandel-basis tensor/matrix mapping,
  PSD parameterization, eigenvalue checks, and penalty utilities.
- `src/invariant_generator/diagnostics.py`: constraint diagnostics, invariant
  feature statistics, and encoder score summaries.
- `src/invariant_generator/model.py`: invariant pool, optional sparse encoder,
  and MLP regressor assembled into the full model.
- `src/invariant_generator/losses.py`: structured objective matching the notes.
- `src/invariant_generator/train.py`: full training loop, logging, checkpointing,
  learning-rate scheduling, early stopping, and diagnostics.
- `src/invariant_generator/evaluation.py`: prediction and regression metrics.
- `src/invariant_generator/symbolic.py`: post-training invariant selection and
  PySR symbolic regression.
- `scripts/train.py` and `scripts/evaluate.py`: CLI entry points.
- `scripts/train_pysr.py`: CLI entry point for post-training PySR.
- `scripts/prune_refit.py`: select top-ranked invariants from a trained
  encoder and refit a reduced model from scratch.
- `scripts/rotatedhill_benchmark.py`: run rotated-hill sanity variants for
  unconstrained, PSD-only, and PSD-plus-standardized settings.
- `tests/`: checks for data preparation, invariant behavior, loss terms, and
  training artifact behavior, constraint behavior, Mandel mapping, and symbolic
  feature selection.
