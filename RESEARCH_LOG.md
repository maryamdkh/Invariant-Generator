# Research log

Chronological record of experiments, decisions and results, kept for writing
the paper. Each entry: goal, what was done, evidence (numbers), decision, and
where the code and outputs are. Probe scripts are in `experiments/probes/`,
their saved outputs in `experiments/outputs/`, and pipeline results under
`results/` (git-ignored, regenerable).

Conventions: `n*` is the true number of groups computed from the known
formula (analysis only). `nrmse` = rmse / std(y). "Recovered" (from entry 9 on)
= relative L2 error < 1e-6 on 10,000 fresh samples over the input domain.

---

## 1. Starting point (2026-10-15)

- Adaptive encoder pipeline for rotated Hill: Stage 1 smallest encoder width
  `n`, Stage 2 sparse encoder `S`, Stage 3 PySR on `J = S z`.
- First Feynman pilot (I.6.20a, `theta = a + b`, pairwise features
  `a, b, a^2, ab, b^2, a/b, b/a`, 2,000 rows): Stage 1 `n = 1`, Stage 2
  `J1 = 2.45a + 2.52b + 2.90ab`, i.e. not `a + b`. 300-row PySR smoke run gave
  `0.085/b` (test relative L2 0.32). Docs: `FEYNMAN_PILOT.md` (v1 section).

## 2. Diagnosis of the v1 miss (2026-10-15)

- Test: best smooth 1-D fit `g(J)` (degree-12 Chebyshev) for candidate `J`s.

  | J | RMSE of best g(J) |
  |---|---|
  | a + b | 7.6e-11 |
  | a^2 + 2ab + b^2 | 2.3e-13 |
  | 0.97a + 1.03b | 2.3e-3 |
  | learned v1 J | 5.5e-3 |
  | ab | 1.3e-2 |

- Conclusion: hypothesis and architecture fine; `a+b` is exactly reachable.
  Causes: (i) Stage 1/2 thresholds (rmse 0.006/0.015) cannot separate a
  3%-wrong `J` from the right one; (ii) Stage 2 kept largest coefficients and
  never tried `{a, b}`; (iii) L1 on `S` is cancelled by MLP rescaling.
- Probe `experiments/probes/support_probe.py`: refit every 1-3 term support
  from the v1 Stage 1 checkpoint (63 refits). `{a, b}` best at val RMSE 7.5e-4,
  next pair 8.6e-3 (11x); learned weights give J = 2.58(a + b).
  Output: `experiments/outputs/support_probe_rows.json`.

## 3. Stage 1 optimization probes (2026-10-15)

- `experiments/probes/stage1_probe.py`: dense `n = 1` fit depends on the
  random start, not on normalization. Seeds 42-49 (standard and scale_only)
  reach val nrmse 0.015-0.035 (`ab`-dominated valley); seed 7 reaches 0.0023.
  8000 epochs slowly drift toward the right family (0.0093). Output:
  `experiments/outputs/stage1_normalization_seed_probe.txt`.
- `experiments/probes/stage1_lr_probe.py`: encoder learning rate scaled
  relative to the MLP.

  | encoder LR scale | val nrmse, seeds 42-46 | g(J) floor |
  |---|---|---|
  | 1 | 0.015-0.035 | ~1e-3 |
  | 10 | 0.002-0.008 | 1e-4-6e-4 |
  | 30 | 0.0008-0.0014 | 7e-6-1e-4 |

- Decision: two-timescale training (`train.encoder_learning_rate_scale`).

## 4. v2 changes and rerun of the a+b pilot (2026-10-15)

- Added (all opt-in): `adaptive.metric = "nrmse"`, `adaptive.restarts`,
  `train.encoder_learning_rate_scale`, `sparsification.support_search =
  "exhaustive"`, `masked_refit_schedule = "cosine"`; config
  `configs/adaptive_encoder_feynman_gaussian_sum_v2.toml`.
- Results (`results/feynman_i_6_20a_sum_v2/`): Stage 1 `n = 1` (restarts
  val nrmse 0.0014/0.0011/0.0017); Stage 2 `J1 = 1.86884a + 1.86906b`
  (`{a,b}` 0.0054 vs next 0.110); Stage 3 `0.39893*exp(-0.14314*J1^2)`,
  fresh-sample relative L2 3.1e-5 (right structure; float encoder
  coefficients differ by 1e-4, so not "recovered" under the 1e-6 criterion of
  entry 9). Stage 3 log: `experiments/outputs/feynman_sum_v2_stage3.log`.
- Product substitution (`theta = a*b`), same config: `J1 = 3.459ab`
  (val nrmse 0.0013, next support 0.714). Stage 3 not run.

## 5. Yield-surface safety check (repeated after every core change)

- `experiments/probes/hill_regression.py`: tiny rotated-Hill run (40 epochs)
  with original vs new code. Stage 1 checkpoint keys, weights and metrics
  identical; Stage 2 `final_sparse_S`, mask, dense S and metrics identical;
  only extra summary key `support_search`. Last verified after the log-feature
  and target-scaler changes (entry 7). Full test suite passes (78 tests at
  the time of entry 12).

## 6. Feynman data and a priori structure (2026-10-15)

- Source: PMLB metadata (SRBench copy of the AI Feynman database), 119
  problems (100 main + 19 bonus): `data/feynman_tables/pmlb_metadata/`.
  Parser `src/invariant_generator/feynman.py`; generator
  `scripts/prepare_feynman_benchmark.py` (3,000 noiseless uniform samples).
- Verification: parsed formulas reproduce PMLB's official data tables to max
  relative error <= 7e-13 on the 21 pilot problems.
- Log reducibility (rank of row-normalized gradients w.r.t. log x from the
  formula; `log_reducibility`): 9 problems have non-positive inputs; of the
  other 110, 98 have `n* < d` and 64 drop >= 2 inputs.
  Saved: `data/feynman_tables/log_reducibility.json`.

## 7. Log-feature pipeline (2026-10-15)

- Added (opt-in): `input_transform.kind = "log"` (`LogFeaturePool`),
  `model.standardize_target`, `sparsification.method = "snap"` (RREF of
  encoder rows, per-row rescale, snap to 1/D for D in [1,2,3,4,6], freeze S,
  refit MLP, smallest passing D), `symbolic.encoded_feature_transform =
  "exp"` (PySR gets monomials with exact exponents), Stage 3 expansion with
  positive symbols.
- Probe `experiments/probes/log_stage1_probe.py` (Stage 1 only): correct
  `n` on test_6 (n*=1), test_5 (2), I.40.1 (2); largest subspace angle
  3.3e-4, 4.1e-3, 9.0e-4 rad.
- Probe `experiments/probes/snap_probe.py`: all three snapped at D = 1 to the
  physical groups: `eps^2 L^2 E_n/(m Z1^2 Z2^2 q^4)`; `G m2/d^3`, `m1/m2`;
  `n_0`, `m g x/(kb T)`.

## 8. Benchmark runner and smoke test (2026-10-15)

- `src/invariant_generator/feynman_benchmark.py`,
  `scripts/run_feynman_benchmark.py`, config `configs/feynman_benchmark.toml`
  (frozen settings; PySR: + - * /, square sqrt exp log sin cos, maxsize 35,
  30 populations x 50, 10 threads, early stop at MSE < 1e-12 var(y)).
- Smoke (2 equations, 2 min PySR): I.40.1 pipeline recovered
  `n_0 exp(-g m x/(T kb))` in 26 s of PySR; PySR alone recovered it in 40 s.
  test_5: PySR alone recovered in 77 s; pipeline only approximated in 2 min
  (groups `G m2/d^3`, `m1/m2` make `m1 + m2` awkward: basis choice matters).
  Found a recovery-check false negative (`kb + 3.7e-8`), fixed. Log:
  `experiments/outputs/feynman_benchmark_smoke.log`.

## 9. 21-equation pilot (2026-10-15/16)

- Selection rule (code: `select_pilot_problems`): main-set problems with
  >= 6 inputs and bonus problems with >= 4 inputs, positive ranges -> 21.
- Run: one seed, 15 min PySR per arm, ~5 h total.
  `results/feynman_benchmark/pilot/benchmark_summary.json`.
- Recovery-check fixes after the run (no rerun needed; re-scored with
  `scripts/rescore_feynman_benchmark.py`): (a) a dataset variable `theta` was
  shadowed by the substitution placeholder; (b) fraction snapping misread
  8.37758 (= 8 pi/3) as 67/8; (c) adopted numerical recovery (rel. L2 < 1e-6
  on fresh points) as the headline; every candidate passing it was read by
  hand and is structurally correct (nearest wrong-form candidate: 6e-6). The
  rounding-based symbolic check is advisory (0 false positives, 3 false
  negatives). Prototypes: `experiments/probes/check_proto*.py`.
- Result: pipeline 12/21, PySR alone 9/21 (3 of the 12 are fallbacks to PySR
  alone where Stage 1 found no reduction). Where Stage 1 reduced (13): 9 vs 6.
  With snapped groups and n >= n*, PySR recovered 9 of 10. PySR search time
  where both recovered: 1, 2, 15, 18, 35 s (groups) vs 17, 4, 49, 189, 261 s
  (raw). Pipeline-only: II.6.15a, I.40.1, test_5, test_6 (I.40.1 and test_5
  were solved by PySR alone in the smoke run: seed variance). PySR-only:
  test_17 (Stage 1 accepted n = 2 < n* = 3). Full table and analysis:
  `FEYNMAN_BENCHMARK.md`.
- Failure modes: fixed 1% Stage 1 threshold missed 6 real reductions
  (III.9.52, I.32.17, test_1, test_9, test_18, test_19; three at
  0.0101-0.0111) and accepted 2 approximate ones (test_12, test_17); dense
  rows when snapping fails (I.9.18); single seed.

## 10. Literature positioning (2026-10-16)

- Closest work: AI Feynman (dimensional analysis); dimensionless learning
  (Bakarji 2022, Xie 2022, IT-pi 2025, Hi-pi 2025, arXiv:2411.15919) mostly
  uses units; FePySR (arXiv:2605.12704, 2026) neural features before PySR
  (must be a baseline); mechanics discovery (EUCLID 2022, neural-polynomial
  CMAME 2024, ICNN yield functions). PySR reported ~65.8% exact recovery on
  noiseless AI Feynman (arXiv:2503.19043 table). Plan and protocol:
  `PAPER_PLAN.md`.

## 11. Week-1 method revisions (2026-10-16, in progress)

- `sparsification.snap_basis = "sparsest"`: sparsest integer basis of the
  snapped row space (enumerate integer combinations, primitive vectors, greedy
  matroid selection). I.11.19 rows `(x1^1/3 y2, ...)` -> `x1 y1, x2 y2, x3 y3`.
  For test_5 it keeps `m1/m2` (sparsity alone does not prefer `G m1/d^3`).
- `adaptive.acceptance = "relative_to_reference"`: train a full-width model
  without the bottleneck; accept n when its metric is within
  `relative_factor` (default 2) of the reference (floor `relative_floor`).
  `adaptive.reference_init` seeds restart 0 from the top gradient directions
  (`encoder.init_weights`).
- Portfolio arm (`run_portfolio`): split the PySR budget between groups and
  raw inputs; choose by validation RMSE only.

## 12. Gradient-spectrum n estimate (2026-10-16, in progress)

- Probe `experiments/probes/gradient_rank_probe.py`: one full-width MLP per
  equation on log features; SVD of row-normalized d f / d log x.
- Raw-y targets (`experiments/outputs/gradient_spectrum_raw_y.txt`): largest
  gap gives n* on 9/21; count of singular values > 1e-2 gives n* on 11/21.
  The noise floor of the gradient spectrum tracks the MLP's fit (val nrmse
  2e-3 to 3e-2).
- log-y targets: running; first three equations show much cleaner gaps
  (II.6.15a: 1, 0.05, then 0.002).
