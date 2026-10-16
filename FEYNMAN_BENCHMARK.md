# Feynman benchmark: adaptive encoder as a front end for PySR

Goal: not to compete with PySR, but to hand it a smaller problem. The
adaptive encoder finds the few combinations of the inputs that the target
depends on; PySR then searches over those instead of over all raw inputs.

## How the yield-surface pipeline generalizes

The adaptive core is unchanged: input features -> (normalization) -> linear
encoder `J = S z` -> MLP, with Stage 1 (smallest `n`), Stage 2 (make `S`
interpretable) and Stage 3 (PySR on `J`). For yield surfaces the feature step
is the stress-invariant pool. For Feynman equations it is `log(x_i)`:

```text
J_i = sum_j B_ij log x_j = log( prod_j x_j^B_ij )
```

so each encoded variable is a power-law group of the inputs, the analogue of a
dimensionless group. Stage 2 (`method = "snap"`) puts the rows of `B` into
reduced row-echelon form, rescales each row and snaps its exponents to simple
fractions, keeping the smallest denominator whose frozen-encoder refit still
passes the validation threshold. Stage 3 gives PySR the monomials
`exp(J_i) = prod_j x_j^B_ij` with the exact snapped exponents, then expands the
result back into the original inputs.

Opt-in pieces, all defaulting to the yield-surface behaviour:

| Setting | Purpose |
|---|---|
| `[input_transform] kind = "log"` | log features (`LogFeaturePool`) |
| `[model] standardize_target = true` | fixed affine output map fitted on training targets |
| `[train] encoder_learning_rate_scale` | encoder learns faster than the MLP adapts around it |
| `[adaptive] metric = "nrmse"`, `restarts` | scale-free threshold, several initializations per `n` |
| `[sparsification] method = "snap"` | exponent snapping instead of sparsity masks |
| `[symbolic] encoded_feature_transform = "exp"` | PySR sees monomials, not logs |

The rotated-Hill configs set none of these and give identical Stage 1 and
Stage 2 results with this code.

## Data and evaluation

- Formulas and ranges: PMLB metadata of the Feynman Symbolic Regression
  Database (the SRBench copy), `data/feynman_tables/pmlb_metadata/`. For the
  pilot equations the parsed formulas reproduce PMLB's official data to
  <= 7e-13 relative error.
- Data: 3,000 fresh noiseless samples per equation, uniform in the published
  ranges; 80/10/10 train/validation/test split. Stage 1/2 select on validation.
- Both arms use the same PySR settings and time limit (`configs/feynman_benchmark.toml`).
  PySR stops early once its MSE is below `1e-12 * var(y)`.
- Recovery (headline): `numerical_recovery`, relative L2 error below 1e-6 on
  10,000 fresh points drawn over the whole input domain. On the pilot every
  candidate passing it was read by hand and is structurally correct (fitted
  constants such as 0.119366 = 3/(8*pi)); the nearest wrong-form
  approximation had 6e-6. `symbolic_recovery` (candidate / truth simplifies to
  a constant after rounding fitted floats) is advisory: it had no false
  positives and 3 false negatives on the pilot.
- `n*` in the tables is an analysis-only quantity computed from the known
  formula (rank of the gradients with respect to `log x`); the pipeline never
  sees it.

## Commands

```bash
.venv/bin/python scripts/prepare_feynman_benchmark.py --verify
.venv/bin/python scripts/run_feynman_benchmark.py --results-dir results/feynman_benchmark/pilot
.venv/bin/python scripts/summarize_feynman_benchmark.py results/feynman_benchmark/pilot/benchmark_summary.json
```

The pilot selection is rule-based: main-set equations with >= 6 inputs and
bonus equations with >= 4 inputs, with strictly positive input ranges (21
equations).

## Pilot results (21 equations, one seed, 15 min PySR per arm)

`results/feynman_benchmark/pilot/benchmark_summary.json`. Rows marked
`[fallback]` are equations where Stage 1 found no reduction, so the pipeline
hands the raw problem to PySR and reports the PySR-alone result.

| equation | inputs | n* (analysis) | Stage 1 n | groups (exponents) | pipeline recovered | PySR alone recovered |
|---|---|---|---|---|---|---|
| III_9_52 | 6 | 3 | None | - | no (4.0e-04) [fallback] | no (4.0e-04) |
| II_36_38 | 8 | 2 | 2 | (1, 0, -1, -1, 1, -1, -2, 1); (-1, -1, 1, 1, 0, 0, 0, 0) | **yes** (0.0e+00) | **yes** (1.2e-16) |
| II_6_15a | 6 | 2 | 2 | (-1, 1, -5, 0, 1, 1); (0, 0, 0, 1, -1, 0) | **yes** (4.9e-08) | no (3.8e-03) |
| I_11_19 | 6 | 3 | 4 | (1/3, 0, 0, 0, 1, 0); (0, 0, 1, 0, 0, 1); (1, 0, 0, 1, 0, 0); (-1/3, 1, 0, 0, 0, 0) | **yes** (0.0e+00) | **yes** (2.4e-14) |
| I_32_17 | 6 | 2 | None | - | no (2.9e-04) [fallback] | no (2.9e-04) |
| I_40_1 | 6 | 2 | 2 | (1, 0, 0, 0, 0, 0); (0, 1, 1, -1, 1, -1) | **yes** (2.0e-16) | no (8.1e-05) |
| I_9_18 | 9 | 6 | 6 | dense | no (3.1e-02) | no (4.0e-02) |
| test_1 | 7 | 2 | None | - | no (2.2e-04) [fallback] | no (2.2e-04) |
| test_11 | 4 | 4 | None | - | no (1.8e-02) [fallback] | no (1.8e-02) |
| test_12 | 5 | 3 | 1 | (1, -2, 1, 1, 0) | no (7.3e-04) | no (6.0e-06) |
| test_16 | 6 | 4 | 4 | (1, 2, 0, 0, 0, 0); (1, 0, 0, 0, -2, 2); (0, 0, 0, 1, 0, 1); (1, 0, -2, 0, 0, 0) | no (nan) | no (1.8e-02) |
| test_17 | 6 | 3 | 2 | (1, 2, 0, 0, 2, 0); (1, 2, 0, 2, 0, -2) | no (1.1e-03) | **yes** (3.2e-14) |
| test_18 | 5 | 2 | None | - | **yes** (2.3e-09) [fallback] | **yes** (2.3e-09) |
| test_19 | 6 | 3 | None | - | **yes** (2.2e-09) [fallback] | **yes** (2.2e-09) |
| test_3 | 4 | 4 | None | - | **yes** (9.4e-10) [fallback] | **yes** (9.4e-10) |
| test_4 | 5 | 3 | 3 | (0, 1, -1, 0, 0); (1, 0, -1, 0, 0); (0, 0, 1, -1, 1) | **yes** (8.3e-09) | **yes** (1.3e-10) |
| test_5 | 4 | 2 | 2 | (-3, 1, 0, 1); (0, 0, 1, -1) | **yes** (1.1e-11) | no (6.2e-03) |
| test_6 | 7 | 1 | 1 | (2, 2, -1, -2, -2, -4, 1) | **yes** (5.1e-13) | no (2.2e-03) |
| test_7 | 5 | 2 | 2 | (1, 1, 0, 0, 0); (0, 0, 1, 2, -2) | **yes** (6.2e-09) | **yes** (8.7e-10) |
| test_8 | 4 | 3 | 3 | (1, 0, 0, 0); (0, 0, 0, 1); (0, 1, 2, 0) | **yes** (2.4e-07) | **yes** (1.1e-16) |
| test_9 | 5 | 2 | None | - | no (2.5e-02) [fallback] | no (2.5e-02) |

Recovered (relative error < 1e-6 on fresh points): pipeline 12/21, PySR alone 9/21

Groups are exponent rows over the inputs in the order of `data/feynman/<id>.json`.

### What the pilot shows

- **When Stage 1 + 2 find the groups, PySR almost always finishes the job.**
  Stage 1 reduced 13 equations; the pipeline recovered 9 of them, PySR alone
  6. With snapped exponents and at least as many groups as needed (n >= n*),
  PySR recovered 9 of 10 (the miss, `test_16`, has four groups inside a
  square root).
- **The groups are the physical ones**, with exact integer or simple
  fractional exponents: `m g x / (kb T)` (I.40.1),
  `epsilon^2 L^2 E_n / (m Z_1^2 Z_2^2 q^4)` (test_6),
  `p_d y z / (epsilon r^5)` and `x / y` (II.6.15a), `G m2 / d^3` and `m1 / m2`
  (test_5).
- **PySR's search gets much shorter.** Where both arms recovered the
  equation, PySR on the groups took 1, 2, 15, 18 and 35 s versus 17, 4, 49,
  189 and 261 s on raw inputs. Stage 1 + 2 add about 1-2 min of network
  training per equation, independent of how hard the formula is.
- **Pipeline-only recoveries:** II.6.15a, I.40.1, test_5, test_6 (PySR on the
  groups: 27, 2, 108, 1 s; PySR alone failed in 15 min). PySR alone did
  recover I.40.1 and test_5 in the earlier 2-minute smoke run, so with one
  seed two of these four reflect PySR's run-to-run variance. II.6.15a and
  test_6 were run once with PySR alone (not in the smoke run), so their
  margin is also unconfirmed until repeated.
- **PySR-alone-only recovery:** test_17, where Stage 1 accepted an
  approximate reduction (n = 2 < n* = 3).

### Where the pipeline falls short

1. **Stage 1's acceptance rule.** A fixed `nrmse < 0.01` threshold is both
   too strict and too loose. It missed six real reductions (III.9.52, I.32.17,
   test_1, test_9, test_18, test_19), three of them at 0.0101-0.0111 just above
   the threshold, because the MLP cannot fit large-dynamic-range targets
   (resonances, 1/sin^4) to 1%. It also accepted approximate under-reductions
   (test_12: n = 1 for n* = 3; test_17). Stage 1 correctly found no reduction
   for test_3 and test_11 (n* = number of inputs).
2. **Dense rows when snapping fails** (I.9.18, 6 groups): PySR then works on
   float-exponent groups and does not recover the formula.
3. **Single seed.** PySR's recovery varies between runs; the comparison needs
   several seeds per arm before any rate is reported.

### Next steps

1. Fit Stage 1 in a better-conditioned target space (e.g. `log y` for
   positive targets: monotone, so the groups are unchanged) and replace the
   fixed threshold with a relative rule (accept `n` when `n + 1` gives no
   large further improvement). Choose these on equations outside this pilot.
2. Repeat both arms with 5+ seeds and report recovery rates with spread.
3. Run all 110 positive-domain equations (the other 9 need a sign-aware
   feature map).
