# Feynman input-expansion pilot

This pilot uses equation I.6.20a from [Udrescu and Tegmark, *AI Feynman*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/):

\[
f(\theta)=\exp(-\theta^2/2)/\sqrt{2\pi}.
\]

The exponent in the published equation is **negative**. The generated pilot
replaces \(\theta\) with \(a+b\). It samples 2,000 noiseless input/output rows
from the published formula; these are **newly generated data**, not the paper's
original numerical data table. `scripts/prepare_feynman_gaussian.py` also
generates one-input and product-substitution files for future experiments.
The current config runs the sum substitution only. The generator writes the
formula to JSON for evaluation; training and PySR read only `X` and `y`.
The feature map includes the raw inputs, one-level products and ordered ratios;
it does not recursively build expressions. A configurable feature cap prevents
large input tables from silently creating an oversized map.

Run from the repository root with the existing `.venv`:

```bash
.venv/bin/python scripts/prepare_feynman_gaussian.py --samples 2000 --seed 2026
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum.toml --stage stage1
```

Read `results/feynman_i_6_20a_sum/stage1/adaptive_stage1_summary.json`.
With the current data, config and seed, Stage 1 selected `n=1`. Run Stage 2:

```bash
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum.toml --stage stage2 --selected-n 1 --checkpoint results/feynman_i_6_20a_sum/stage1/n01/checkpoint_best.pt
```

Read `results/feynman_i_6_20a_sum/stage2_sparse/adaptive_stage2_sparsify.json`
for the selected mask and the explicit \(J\) formula. Stage 1 and Stage 2
select using validation rows; the 200 test rows remain unused until Stage 3.
Then start PySR interactively:

```bash
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum.toml --stage stage3 --selected-n 1 --checkpoint results/feynman_i_6_20a_sum/stage2_sparse/checkpoint_best.pt
```

The config caps this first PySR search at 15 minutes. Its terminal displays
the evolving Hall of Fame and loss; press `q`, then Enter, to end it earlier
and let PySR return its results. For a longer search, change
`[symbolic].timeout_hours` in the config *before* Stage 3. Do not change the
dataset, split, transformation or encoder config between stages.

After Stage 3, inspect these files in `results/feynman_i_6_20a_sum/stage3_pysr/`:

- `encoded_invariant_formulas.json`: candidate features and learned \(J\).
- `equations.csv`: PySR's Pareto equations and search losses.
- `best_equation.txt`: PySR's selected equation in \(J\).
- `equation_in_original_inputs.txt`: the selected equation after substituting \(J\).
- `metrics.json`: train, validation and untouched-test errors.
- `original_input_evaluation.json`: test relative L2 error and exact symbolic
  recovery, checked by simplifying the difference from the known formula.
- `heldout_predictions.npz`: test inputs, targets and predictions.

The PySR search also writes
`results/feynman_i_6_20a_sum/stage3_pysr/pysr_search/hall_of_fame.csv`.
Its loss measures fit on the search data; the final test error and exact
formula check answer different questions. The [AI Feynman paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/)
counts mathematically equivalent formulas as exact recovery.

## v1 local check

Stages 1 and 2 ran on the 2,000-row pilot. Stage 1 selected one latent
variable with validation RMSE `0.002648`. Stage 2 selected three active terms
with validation RMSE `0.005707` and found approximately
`J1 = 2.45326*a + 2.5179*b + 2.89536*a*b`. This is a valid predictive
compression, but it is not the expected `a+b` substitution. A separate
20-iteration PySR smoke test on a 300-row version verified Stage 3 output
generation; it did **not** recover the equation. The 2,000-row Stage 3 search
is left for the interactive run above.

## v2: diagnosis and rerun

The v1 miss came from the procedure, not the hypothesis or the model: `a+b`
is in the feature pool and a one-variable function of it fits the data to
~1e-10. Three things kept v1 from finding it:

1. **Stage 1 training.** With one shared learning rate the MLP adapts to
   whatever `J` it receives, so the encoder direction barely moves and stays
   near an `a*b`-dominated start (val nrmse 0.015-0.035 over seeds 42-49).
   `[train].encoder_learning_rate_scale = 30` reaches the `a+b` family from
   every tested seed (val nrmse ~0.001). `[adaptive].restarts` keeps the best
   of several seeds per `n`. The absolute `1e-6` MSE deltas for early stopping
   and LR decay were also sized for yield-surface targets of ~1, not ~0.08.
2. **Stage 2 support choice.** Keeping the largest coefficients picked
   `{b, a*b}` at cap 2, never `{a, b}`. `support_search = "exhaustive"`
   refits every support with `cap` terms and keeps the best on validation.
3. **Thresholds.** Absolute RMSE 0.006 let a 3%-wrong `J` pass.
   `metric = "nrmse"` (rmse / std(y)) gives a scale-free threshold.

Stage 3 reports `numerical_recovery` (relative L2 below 1e-6 on fresh points
over the whole input domain) and an advisory `symbolic_recovery`; see
[`FEYNMAN_BENCHMARK.md`](FEYNMAN_BENCHMARK.md). Strict SymPy equality stays
reported but cannot pass with fitted floats such as 0.39893 for
1/sqrt(2*pi).

All new options default to the v1 behaviour; the rotated-Hill configs are
unchanged and produce identical Stage 1 and Stage 2 results.

```bash
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum_v2.toml --stage stage1
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum_v2.toml --stage stage2 --selected-n 1 --checkpoint results/feynman_i_6_20a_sum_v2/stage1/n01/seed043/checkpoint_best.pt
.venv/bin/python scripts/adaptive_encoder_pipeline.py --config configs/adaptive_encoder_feynman_gaussian_sum_v2.toml --stage stage3 --selected-n 1 --checkpoint results/feynman_i_6_20a_sum_v2/stage2_sparse/checkpoint_best.pt
```

Results (`results/feynman_i_6_20a_sum_v2/`, same data and split as v1):

| Stage | Result |
|---|---|
| 1 | `n=1`; restarts 42/43/44 reach val nrmse 0.0014/0.0011/0.0017 (threshold 0.01) |
| 2 | `J1 = 1.86884*a + 1.86906*b`; `{a,b}` val nrmse 0.0054, next support `{a, b*b}` 0.110 |
| 3 | `0.39893*exp(-0.14314*J1^2)`: the structure of `exp(-(a+b)^2/2)/sqrt(2*pi)`, but the float encoder coefficients differ by 1e-4, so the fresh-sample relative L2 is 3.1e-5 and it does not meet the 1e-6 recovery criterion |

The same v2 config on the product file (`theta=a*b`) selected `n=1` and
`J1 = 3.459*a*b` (val nrmse 0.0013, next support 0.714); Stage 3 was not run.

This is one feasibility pilot, not a benchmark result. A publication claim
would require original and expanded equations, a direct PySR baseline under
the same search budget, multiple seeds, exact-recovery rates, and held-out
errors. [AI Feynman already tests sum, product and ratio variable reductions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/),
so those transformations alone are not a new contribution.
