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
- log-y targets (`experiments/outputs/gradient_spectrum_log_y.txt`; 3
  equations skipped because y <= 0 occurs: test_3, test_9, test_19): fits
  improve (val nrmse mostly 1e-3 to 5e-3) and spectra show clean gaps; both
  the largest-gap rule and the 1e-2 cutoff give n* on 14/18. Misses: I.32.17
  and test_17 (gap rule), III.9.52 (poor fit), I.40.1, test_6 (cutoff), test_11
  and test_12 (weak directions: test_12 has singular values 1, 0.0075, 0.0016,
  i.e. nearly a one-group function on the sampled domain).
- Decision: do not tune a spectral cutoff on 18 development equations. Use
  log y in Stage 1-2 when y > 0, accept n relative to the full-width
  reference fit, initialize one restart from the top gradient directions, and
  report the spectrum as a diagnostic. Stage 3 and the PySR-alone arm keep
  fitting the original y.

## 13. Checkpoint commit (2026-10-16)

- Branch `feynman-generalization`, commit `8b9955c` (all work up to entry 12).

## 14. Stage 1-2 revisions implemented (2026-10-16)

- `data.target_transform = "log"` (supervised): Stages 1-2 train on log y
  after the split (split unchanged); `symbolic.target_transform = "exp"` maps
  back so PySR fits y; Stage 3 test errors and the PySR early-stop threshold
  use the y scale. The runner falls back to identity when some y <= 0.
  The PySR-alone arm always fits y.
- Runner: `--seed` (training seed and PySR random_state; data split fixed),
  `--portfolio-raw-share`.
- Config `configs/feynman_benchmark_v2.toml` = pilot config + log target,
  relative acceptance (factor 2, floor 1e-3), reference-initialized restart,
  sparsest basis. PySR settings identical to the pilot, so pilot PySR-alone
  results (seed 42) are reused as the baseline arm of the seed-42 dev run.
- Checks: 79 tests pass; rotated-Hill Stage 1 weights/keys and Stage 2
  outputs identical to the original code.

## 15. v2 smoke test and seed-42 development run (2026-10-16)

- Smoke (`results/feynman_benchmark/v2_smoke/`, 2 min PySR, pipeline and
  portfolio arms): I.32.17, missed by the pilot's fixed threshold, now gets
  n = 2 = n* (reference val nrmse 0.0016, gradient spectrum 1, 0.037, 0.0015;
  n = 1: 0.033 rejected, n = 2: 0.0010 accepted). test_5 groups in the
  sparsest basis: `m1/m2`, `d^3/(G m1)`. At 2 min neither reached the 1e-6
  criterion (4.7e-5, 1.1e-5). Portfolio reused Stages 1-2 as intended.
- Launched seed-42 development run (`results/feynman_benchmark/v2_dev_seed42/`,
  config `configs/feynman_benchmark_v2.toml`, 15 min PySR per arm; portfolio
  splits 7.5/7.5 min). Baseline arm seeded from the pilot (identical PySR
  settings and seed), marked `reused_from` in the summary.

## 16. Cluster tooling (2026-10-16)

- Clusters available: Vera and Arrhenius (SLURM). Added `cluster/`:
  `setup_env.sh` (uv env, Julia/PySR precompile in `.julia_depot/`, data for
  all 119 equations), `make_tasks.py` (one task per equation x seed;
  `--positive-only` gives 110 equations), `feynman_array.sbatch` (all arms per
  task, own results folder, `--numprocs` = SLURM cores), and
  `scripts/merge_feynman_benchmark.py` (merges per-task summaries; recovery per
  arm across seeds and per equation). Dry run: 119 datasets generated, 550
  tasks for 5 seeds, merge reproduces the pilot's 12 vs 9.

## 17. Venue, references and yield benchmarks (2026-10-16)

- Author's decisions: target a CS/ML venue (method-first framing; yield
  surfaces become an application section); prefer recent literature for
  positioning; cluster jobs are submitted by the author (two-factor login),
  using `cluster/`.
- References in `references/`: Cazacu & Barlat (2001) CB2001
  (`generalized_drucker.pdf`): J2^0 (Eq. 5, a1-a6), J3^0 (Eq. 4, b1-b11),
  criterion (J2^0)^3 - c (J3^0)^2 = k^2 (Eq. 8); coefficients for 6016-T4
  (c = 1.4) and 2090-T3 (a1 = 0.586, a2 = 1.05, a3 = 0.823, a4 = 0.96,
  b1 = 1.44, b2 = 0.061, b3 = -1.302, b4 = -0.281, b5 = -0.375, b10 = 0.445,
  c = 1.285; plane stress); linear-transformation variant (Eqs. 12-13).
  Barlat et al. (2005) Yld2004-18p (`...S0749641904001160...`). Cazacu,
  Plunkett & Barlat (2006) CPB06 (`...S0749641905001257...`).
- Literature check for the CS framing: LLM-SRBench (ICML 2025 Oral, 239
  problems, anti-memorization) is now a standard SR benchmark; LLM-SR and
  LaSR are current LLM-guided SR systems. Recent ML yield work: convex neural
  yield discovery (arXiv:2606.19375), "AI Meets Plasticity" survey
  (arXiv:2602.01215). Plan rewritten: `PAPER_PLAN.md`.
- Yield benchmark decision: Hill48 (exact), CB2001 2090-T3 (two encoder rows
  over an orthotropic polynomial integrity basis: a quadratic and a cubic
  invariant), Yld2004-18p (non-polynomial stress test).

## 18. Yield-surface benchmarks: criteria, data, feature map (2026-10-16)

- `src/invariant_generator/yield_criteria.py`: CB2001 (Eqs. 4, 5, 8) and
  Yld2004-18p (Eqs. 5, 14, 15; Table 2, a = 8) as equivalent stresses
  (degree-1 homogeneous, = 1 for uniaxial x tension). Presets:
  `cb2001_2090T3`, `cb2001_6016T4` (plane-stress fits; unidentified a5, a6,
  b6-b9, b11 set to 1), `yld2004_2090T3`, `yld2004_6111T4`.
- Validation (`tests/test_yield_criteria.py`): isotropic limits equal
  classical J2/J3 and Hershey to ~1e-15; all presets positive, homogeneous and
  pressure-insensitive on random 3D stresses; directional yield stresses match
  the papers' figures (2090-T3, Barlat 2005 Fig. 7: CB2001 within 0.05,
  Yld2004 within 0.02; 6016-T4, CB2001 Fig. 4: within 0.01). A transcription
  error in the 6016-T4 coefficients (a3, a4) was caught by these checks and
  fixed.
- Data: `scripts/prepare_yield_benchmark.py` -> `data/yield_benchmarks/`
  (2,000 surface points per criterion, Mandel format like rotatedhill.hdf).
- Feature map `input_transform.kind = "orthotropic_polynomial"`: 29 degree-2/3
  monomials of the orthotropic integrity basis (CB2001 Eqs. 2-3). J2^0 and
  J3^0 of CB2001 are exact linear combinations (residual ~1e-14; 9 and 20
  terms), so the exact answer is n = 2. Config `configs/yield_cb2001_2090T3.toml`;
  probe `experiments/probes/yield_stage1_probe.py` (running).
- LLM-SRBench: gated on Hugging Face (401 without accepted terms); the author
  needs to accept the terms and download it to `data/llm_srbench/`.
- Incident: the seed-42 dev run (detached process) stopped during III.9.52's
  portfolio arm, coinciding with a working-directory change of the agent's
  shell; relaunched as a tracked task, resuming from its summary.

## 19. CB2001 (2090-T3) Stage 1 result (2026-10-16)

- `experiments/probes/yield_stage1_probe.py configs/yield_cb2001_2090T3.toml`
  (output `experiments/outputs/yield_cb2001_2090T3_stage1.txt`, 694 s):
  full-width reference nrmse 0.023 (gradient spectrum 1, 0.12, 0.068, ...);
  n = 1: 0.069 (rejected), n = 2: 0.008 (accepted; threshold 0.047).
- The two learned encoder rows span the generating invariants: principal-angle
  cosines with span{J2^0, J3^0} = 0.999999 and 0.999993. The pipeline
  recovered both CB2001 generalized invariants (9 and 20 monomial terms) from
  yield-surface points alone.
- Note: the reference model fit worse than the n = 2 bottleneck, so the
  relative rule's threshold (2x reference) was loose here; the decision was
  still correct (n = 1 is 8.7x worse than n = 2).
- Next for this benchmark: Stage 2 with a degree-graded basis (one purely
  quadratic and one purely cubic row) and Stage 3 PySR for
  f = ((J2)^3 - c (J3)^2)^(1/6).

## 20. Degree-graded Stage 2 for polynomial feature maps (2026-10-16)

- `sparsification.method = "graded"` (`graded_encoder_rows`): for features
  graded by polynomial degree, find combinations of the learned rows supported
  on a single degree (left singular vectors of the off-degree block), freeze
  them, refit the MLP. Test: recovers J2^0 and J3^0 exactly from an arbitrary
  mix of the two (`tests/test_yield_criteria.py`).
- First CB2001 run: graded split residual 0.0013 (row space splits cleanly
  into one quadratic and one cubic row), but the frozen refit reached val
  nrmse 0.052 > 0.01 and Stage 2 fell back to the dense rows; PySR (15 min)
  then returned a long expression in the mixed rows that is NaN on some fresh
  directions. Cause: graded rows normalized by their largest coefficient give
  J on very different scales. Fix: rescale each graded row to unit spread of
  J on training data (row scale is free); refit epochs 2000 -> 4000. Rerun
  in progress (`results/yield_benchmarks/cb2001_pipeline_probe2.log`).
- Open consistency issue: Stage 2 acceptance uses the absolute metric
  threshold, while v2 Stage 1 uses the reference-relative rule. Left
  unchanged during the running seed-42 dev run; to align before the
  cluster run.

## 21. LLM-SRBench (LSR-Transform) integration (2026-10-16)

- Data: author obtained access (gated, manual approval) and downloaded
  nnheui/llm-srbench to `data/llm_srbench/` (228 MB). Layout: parquet tables
  (name, symbols, symbol_descs, symbol_properties, expression) and
  `lsr_bench_data.hdf5` with `lsr_transform/<name>/{train, test}` (~50k/12.5k
  rows) and `lsr_synth/<domain>/<name>/{train, test, ood_test}`; float32;
  columns [output, inputs...] in `symbols` order (verified: formula vs data
  median relative error 2e-8 to 2e-7, i.e. float32 precision; a few
  near-singular problems reach relative L2 up to 0.6 from float32 inputs).
- LSR-Synth (129 problems) not converted: ground truths contain named
  constants without values (e.g. F0, beta, omega0) and one malformed constant
  (CRK0 `..._z`), so only numeric test/OOD error can be scored there.
- `scripts/prepare_llm_srbench.py` -> `data/llm_srbench_converted/`: per
  problem 3,000 seeded rows of the official train split (X, y) and 10,000
  rows of the official test split (X_eval). 111 problems; 91 with positive
  inputs (log features apply); 67 with positive targets.
- Evaluation: candidates are scored against the ground-truth formula in
  float64 on X_eval (`eval_key` in the problem JSON), not against float32
  outputs. Problems with non-positive inputs skip the log pipeline
  (`skipped`) and count as PySR alone. Runner: `--data-dir`.
- Smoke run (2 problems, 2 min PySR) in progress: `results/llm_srbench/smoke/`.

## 22. CB2001 (2090-T3) end to end (2026-10-16)

- Rerun with unit-spread graded rows (`experiments/outputs/yield_cb2001_2090T3_stages23.txt`):
  Stage 2 graded basis accepted (off-degree residual 0.0013; val nrmse
  0.0047): J1 = quadratic invariant with 9 terms, J2 = cubic invariant with 20
  terms, i.e. the support of CB2001's J2^0 and J3^0.
- Stage 3 (PySR, 15 min, 4 threads): sqrt(J1 (cbrt(0.647 - 0.817 J2^2/J1^3)
  - 0.068)); true form sqrt(J1 cbrt(1 - c' J2^2/J1^3)) x const. Right nested
  structure (J3^2/J2^3 inside a cube root inside a square root) with one
  spurious -0.068 term. Relative L2 vs the true criterion on 5,000 fresh
  stress directions: 2.0e-4.
- Remaining error has two sources: approximate encoder coefficients (learned,
  ~1e-3) and one structural slip in PySR. Planned method step: a final joint
  refit of all constants (encoder rows and formula constants) on the training
  data once a structure is found ("constant polishing").
- Smoke result after fixing a path bug (problem names with dots, e.g.
  `II.36.38_1_0`, broke `Path.with_suffix` when locating the problem JSON, so
  scoring was skipped; regression test added): II.36.38_1_0 recovered exactly
  by both arms, PySR 1.2 s on the groups (`T f kb/mom`, `M alpha/(eps c^2)`)
  vs 33 s on raw inputs; II.6.15a_1_0 not recovered at 2 min by either
  (pipeline 6.4e-4, PySR alone 1.2e-2). `scripts/rescore_feynman_benchmark.py`
  gained `--data-dir`.

## 23. Interim: seed-42 dev run, 5 of 21 equations (2026-10-16 13:05)

- Recovered by every arm: II.36.38, I.11.19 (PySR on groups 2 s each).
  Not recovered by any arm: III.9.52 (Stage 1 n = 2 < n* = 3), II.6.15a,
  I.32.17 (Stage 1 now finds n = n* = 2, unlike the pilot, but PySR misses).
- II.6.15a: pilot pipeline recovered it; v2 finds equally valid groups in a
  different basis (`x/y`, `eps r^5/(p_d x z)`; the sparsest-basis score ties
  with the pilot's y-based group) and PySR reaches only 1.3e-5 in 15 min.
  With one seed this is not separable from PySR run-to-run variance, which
  is why the protocol needs several seeds.

## 24. Move to the cluster (2026-10-16)

- Commit `6ac3eae` on branch `feynman-generalization`, pushed to GitHub
  (origin maryamdkh/Invariant-Generator). Publisher PDFs in `references/` are
  git-ignored (kept local for copyright reasons).
- Plan: the author pulls the branch on Arrhenius, runs `cluster/setup_env.sh`
  and submits the 21-equation seed-42 dev run as a SLURM array (CPU only; no
  GPU needed: small MLPs, PySR is CPU-bound). The local dev run keeps going
  until the cluster jobs are confirmed running.
- Cluster bring-up issues and fixes: (1) an old `.venv` copied between
  machines failed with "Exec format error"; (2) PySR/Julia failed with
  `GLIBCXX_3.4.30 not found` (old /lib64/libstdc++): fixed by putting Julia's
  bundled lib directory first on LD_LIBRARY_PATH (`450dd52`); (3) first array
  (job 3921807) failed in 4 s on node n48: Arrhenius GPU nodes are aarch64
  while the login node is x86_64; scripts now keep per-architecture uv, venv
  and Julia depot (`4ef3288`) and the GPU-node environment is built inside an
  interactive GPU job.
- (4) With Julia's lib directory on LD_LIBRARY_PATH, Julia's loader failed
  with `libLLVM.so.20.1jl: cannot open` on both architectures (the file
  exists; the folder rename away from a space in the path did not help).
  Replaced by LD_PRELOAD of Julia's bundled libstdc++.so.6 only.
- Cluster dev run (job array 3927133, 21 tasks, Arrhenius GPU partition,
  account naiss2026-4-599-gpu) started successfully: tasks running Stage 1 at
  ~85 epochs/s on CPU. Local seed-42 dev run stopped (its partial results for
  6 equations remain in `results/feynman_benchmark/v2_dev_seed42/`).
- Job array 3927133 finished: 20/21 tasks COMPLETED (5-57 min each); task 10
  (feynman_test_12) OUT_OF_MEMORY after 41 min under the default memory
  limit. Rerun with --mem=64G; `#SBATCH --mem=64G` added to the array script.

## 25. Cluster dev run (v2, seed 42) results and diagnosis (2026-10-17)

- Recovered (relative L2 < 1e-6 on fresh points): pipeline 9/21, portfolio
  13/21, PySR alone 11/21 (`experiments/outputs/v2_dev_seed42_cluster_summary.json`).
  Portfolio >= PySR alone on every equation; adds I.32.17 and test_17.
- Stage 1 picks n = n* on 16/19 reducible equations (pilot: 10/19) and
  correctly finds no reduction on test_3, test_11. Misses: III.9.52, test_12,
  test_19 (n = 2 < n* = 3).
- Regressions vs the pilot where both chose the same, correct n:
  II.6.15a, test_4, test_5 (sparsest basis gave equivalent but less natural,
  sometimes inverted groups: test_5 `d^3/(G m1)`, `m1/m2` instead of
  `G m2/d^3`, `m1/m2`; test_4 `m/E_n` instead of `m/U`; II.6.15a x-based
  inverted group) and test_8 (snapping found the exact groups theta, E_n,
  m c^2, but the frozen refit reached val nrmse 0.0144 > the fixed 0.01, so
  Stage 2 fell back to dense float rows; Stage 1 fit was 0.0019).
- Decision (v3, `configs/feynman_benchmark_v3.toml`): `snap_basis = "rref"`
  (sparsest-basis hypothesis rejected: 3 regressions, no gains) and
  `snap_acceptance = "relative_to_dense"` (accept a snapped basis when its
  frozen refit is within 2x of the unsnapped rows' frozen refit, which
  isolates snapping from refit convergence). Check running locally on test_8
  and test_5 from the cluster's Stage 1 checkpoints
  (`experiments/probes/stage23_from_checkpoint.py`).
- v3 local check from the cluster Stage 1 checkpoints
  (`experiments/outputs/v3_check_test8_test5.log`): test_5 snapped to the
  natural rref groups `G m2/d^3`, `m1/m2` and was recovered in 18 s of PySR
  (rel. L2 1.8e-11; v2 failed). test_8: the relative rule accepted the exact
  groups (snapped refit 0.0196 vs unsnapped 0.0180, threshold 0.036; v2's
  fixed 0.01 rejected them); PySR reached 2.8e-5 in 15 min, not recovered
  (the pilot recovered it from the same groups in 35 s: PySR variance).

## 26. Cluster dev run v3 (seed 42) and decision to freeze (2026-10-17)

- Job array 3963382, `configs/feynman_benchmark_v3.toml`. Recovered: pipeline
  12/21, portfolio 11/21, PySR alone 13/21.
- The pipeline regained exactly the three equations targeted by the v3
  fixes (II.6.15a, test_4, test_5); test_8 still missed (PySR variance, see
  entry 25).
- PySR alone, with identical settings and seed in all three runs, recovered
  9 (pilot), 11 (v2) and 13 (v3) of the 21: multithreaded PySR is not
  reproducible run to run, so single-seed differences of +-2 equations are
  noise (e.g. test_9, test_12, test_19 flip between runs). Arms cannot be
  ranked from one seed; the protocol's 5 seeds are required.
- Decision: freeze v3 for the full runs (Feynman: 110 positive-input
  equations x 5 seeds; LLM-SRBench LSR-Transform: 111 x 5), pending a check of
  the remaining GPU-hour allocation (~0.5 GPU-h per task observed).

## 27. Full Feynman run, v3, seed 42 (2026-10-17)

- 110 positive-input AI Feynman equations, 110/110 tasks completed, ~30.5
  GPU-hours (0.28 GPU-h/task). `experiments/outputs/v3_full_seed42_summary.json`.
- Recovered: pipeline 85, portfolio 88, PySR alone 86 (dev 21: 10/14/11;
  held-out 89: 75/74/75). Pairwise vs PySR alone: pipeline +6/-7,
  portfolio +8/-6. No recovery difference beyond run-to-run noise.
- Stage 1 structure discovery: n = n* on 93 of 102 reducible equations,
  n < n* on 7, missed reduction on 2; correctly no reduction on all 8 with
  n* = d.
- Speed (73 equations recovered by both, pipeline reduced): PySR median 11 s
  on groups vs 3 s on raw inputs (mean 44 s vs 75 s; groups faster on 18/73).
  Stage 1+2 overhead median 192 s, so wall-clock favours PySR alone on these.
- Interpretation: at a 15-minute budget, PySR alone already solves ~78% of
  the AI Feynman set, mostly in seconds, so this benchmark leaves little room
  for a front end; it does not support a claim of higher recovery. The
  clearly positive result is unit-free structure discovery (n* found on 91%
  of reducible equations, exact groups). Where the method should matter:
  harder benchmarks (LLM-SRBench), noise, small budgets, and many-variable
  problems where SR's search space explodes.

## 28. Target noise for supervised benchmarks (2026-10-17)

- New opt-in `data.target_noise` (+ `target_noise_seed`, default 42): y +
  N(0, (s std(y))^2) on the train and validation rows; test rows and fresh
  recovery points stay noiseless; the draw is independent of the training
  seed. Default 0 adds an exact zero, so v3 runs are unchanged; the surface
  (Hill) path is not touched. CLI `--target-noise`, sbatch `TARGET_NOISE`.
  Test: `test_target_noise_touches_train_and_validation_only`; full suite
  passes.
- Smoke run, I.40.1 `n_0 exp(-g m x/(T kb))`, s = 0.01, v3, 2-min PySR
  budget, seed 42: Stage 1 n = 2, snapped exact groups (n_0, g m x/(T kb)),
  Stage 2 val nrmse 0.00999 (at the noise floor); PySR on the groups
  returned the exact formula (rel. L2 0). PySR alone returned
  `n_0 exp(-1.000382 g m x/(T kb))` (rel. L2 3.5e-4), not recovered under
  the 1e-6 criterion. One equation only, but this is the mechanism to test:
  exact snapped exponents absorb the noise that otherwise lands in fitted
  constants.
- Scoring note: with noise the 1e-6 criterion mostly measures whether the
  constants are exact; also report fresh-point R^2 and a symbolic check with
  constants rounded to ~3 digits (post hoc from `expanded_equation`).
