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

## 29. Submitted: LLM-SRBench and noisy Feynman, v3, seed 42 (2026-10-17)

- LLM-SRBench LSR-Transform, 111 problems (91 positive-input), job array
  3983674, `RUN=llmsr_v3`. Conversion on the cluster matched the Mac (worst
  median formula-vs-data error 2.0e-7). Needed pandas/pyarrow, now declared
  as the `benchmarks` extra (`4b5e97d`).
- AI Feynman with 1% target noise, 110 positive-input equations, job array
  3983693, `RUN=v3_noise1pct`, `TARGET_NOISE=0.01`.
- Both at commit `4b5e97d` (code identical to `e98e031` apart from the
  dependency declaration).

## 30. Project rules file and reusable Hill check (2026-10-17)

- `CLAUDE.md` at the repo root: project goal, the rule that rotated Hill
  must stay unchanged, workflow (research log, cluster steps, seeds).
  Loaded automatically by every Claude Code session on this folder.
- `experiments/probes/hill_old_vs_new.py [OLD_REF]`: runs a tiny rotated
  Hill Stage 1 + Stage 2 on OLD_REF (git worktree) and on the working tree,
  compares data hash, Stage 1 weights/metrics and Stage 2 S/mask/metrics.
  Against `afbee09` (before target noise): IDENTICAL on all keys.
- `reports/claude_project_brief.md`: one-page summary for the author's
  claude.ai project instructions.
- Cluster status shortly after submission: both arrays running
  (LLM-SRBench throttled by the 40-task cap, noise array by the account's
  GPU run-minutes limit `AssocGrpGRESRunMinutes`).

## 31. Results: LLM-SRBench and noisy Feynman, v3, seed 42 (2026-10-17)

LLM-SRBench LSR-Transform (`results/feynman_benchmark/llmsr_v3/`):
- 91 positive-input problems (the comparison set): recovered (rel. L2 <
  1e-6) pipeline 63, portfolio 63, PySR alone 57. Pairwise vs PySR alone:
  pipeline +7/-1. Stage 1 reduced 83/91 (n = 1: 30, 2: 46, 3: 7), all 83
  snapped. On the 52 recovered by both, PySR median 11 s on groups vs 8 s
  raw (mean 33 vs 58 s); Stage 1+2 median 191 s.
- Pipeline-only: I.32.17_2_0, II.11.27_3_0, II.11.3_4_1, II.13.23_1_0,
  II.24.17_1_1, II.6.15a_1_0, III.4.33_3_0 (PySR alone near misses, rel. L2
  2e-6 to 2e-2). PySR-only: II.36.38_7_0 (d = 8, Stage 1 chose n = 1).
- First benchmark beyond Hill where the pipeline is ahead (+6 net), but one
  seed; Feynman showed +-2 run-to-run variance. Needs seeds 43-46.
- Bug: PySR alone crashed on the 20 non-positive-input problems (the run
  applied the log feature map's positivity check to the raw-input arm).
  Fixed in `run_pysr_baseline` (raw-input arm uses a neutral feature map;
  data identical on positive problems, checked). The pipeline arm is skipped
  on those 20, so the comparison set is unaffected. III.15.12_0_0: Stage 3
  overflow (inf encoded feature); PySR alone also missed it.

Noisy Feynman, 1% target noise (`results/feynman_benchmark/v3_noise1pct/`,
scored by `scripts/score_noisy_benchmark.py`, `noisy_scores.json`):

| arm | exact 1e-6 | R2 > 0.999 | symbolic (3 digits) |
|---|---|---|---|
| pipeline | 7 | 102 | 55 |
| portfolio | 17 | 108 | 72 |
| PySR alone | 26 | 106 | 70 |

- Under noise both arms mostly find the same structure with the same noisy
  constants (e.g. 0.159139 h n for h n/(2 pi)); exact 1e-6 is not meaningful.
- Stage 1 chose the same n as the noiseless run on 94/110; same snapped
  exponents on 57/102.
- Of the 21 symbolic losses vs PySR alone, 13 had the correct groups
  (same exponents as the clean run): PySR returned the group plus a tiny
  wiggle, e.g. `J1 - 0.11 sin(19.6 J1)`. Cause: PySR's model_selection
  "best" picks max score -dlog(loss)/dcomplexity among loss <= 1.5 min, and
  the first hall-of-fame entry (`J1`, complexity 1, here the truth) has
  score 0 by definition, so it can never be picked; noise gives wiggles a
  small positive score. Hits the pipeline only (raw inputs never have a
  one-term answer). Other losses: 5 different n, 3 different exponents.
- Offline re-pick (`experiments/probes/reselect_simplest.py`, same rule
  for both arms, first entry scored against the constant model var(y)):
  symbolic pipeline 55 -> 67, PySR alone 70 -> 70; R2 > 0.999 unchanged
  (102 vs 106). With the fix the arms are level within single-seed noise:
  no advantage from the pipeline under 1% noise on Feynman.
- Decision: the selection fix is a real Stage 3 improvement (to add as an
  opt-in `symbolic.model_selection` option, Hill default unchanged). Noise
  alone is not where the pipeline wins on Feynman; next evidence comes from
  LLM-SRBench seeds and the scaling experiment.
- `cluster/make_tasks.py --positive-only` now also reads `inputs_positive`
  (LLM-SRBench manifests have no input ranges).

## 32. LLM-SRBench, 5 seeds (2026-10-18)

- Seeds 43-46: job array 4018521 (364 tasks = 91 positive-input problems x
  4 seeds, arms pipeline + PySR alone, v3 config, commit `37b71cb`). All
  tasks returned results; merged with seed 42 in
  `results/feynman_benchmark/llmsr_v3/merged_summary.json`.
- Recovered (rel. L2 < 1e-6), 91 problems:

  | seed | 42 | 43 | 44 | 45 | 46 | mean +- sd |
  |---|---|---|---|---|---|---|
  | pipeline | 63 | 66 | 59 | 58 | 60 | 61.2 +- 2.9 |
  | PySR alone | 57 | 59 | 57 | 55 | 58 | 57.2 +- 1.3 |
  | difference | +6 | +7 | +2 | +3 | +2 | +4.0 |

- Pipeline ahead on all 5 seeds. Paired over the 455 problem x seed pairs:
  pipeline-only 34, PySR-only 14 (exact binomial p = 0.0055; treats pairs
  as independent, so optimistic). Per-problem recovery rate (problem as
  unit): 13 problems better, 5 worse, 73 tied; Wilcoxon p = 0.058. Mean
  recovery rate 0.673 vs 0.629. Solved on all 5 seeds: 54 vs 50; on at
  least one: 69 vs 67.
- Consistent pipeline gains: II.11.3_4_1 (5/5 vs 0/5), III.4.33_3_0 (5/5 vs
  0/5), I.24.6_0_0 (5/5 vs 2/5), I.32.17_2_0 (5/5 vs 2/5), II.24.17_1_1 (4/5
  vs 1/5), II.11.27_3_0, I.34.14_2_0 (4/5 vs 2/5).
- Consistent loss: II.36.38_7_0 (0/5 vs 5/5): y = c^2 eps (H mom - T f kb)
  /(alpha mom) has two groups, but Stage 1 accepted n = 1 at val nrmse 0.048
  on every seed (same snapped row each time): the acceptance threshold was
  too loose here; a systematic Stage 1 failure. I.32.17_4_3 (0/5 vs 2/5):
  n = 2 groups plausible, PySR cannot assemble the nested square roots from
  them.
- Stability: Stage 1 chose the same n on all 5 seeds for 81/91 problems.
- Speed (259 problem x seed pairs recovered by both, reduced): PySR median
  11 s on groups vs 7 s raw (mean 29 vs 45 s); Stage 1+2 median 194 s.
- Interpretation: on the harder LLM-SRBench problems the pipeline gives a
  consistent gain of ~4 problems (~4.4 points, 61.2 vs 57.2 of 91) over PySR
  alone at the same PySR budget, on every seed; the gain comes from a set of
  problems where PySR alone gets close but not exact. Borderline under the
  conservative per-problem test (p = 0.058), clear under the paired-trials
  test. Contrast with AI Feynman (no difference: PySR alone already solves
  ~78% in seconds).

## 33. Positioning and comparison methods for the paper (2026-10-25)

- Goal: fix the core contribution and the baselines before writing.
- Read Cranmer et al. 2020 (arXiv:2006.11287, symbolic distillation) and the
  PySR v2.7.2 docs/README. Finding: "NN with a compact latent, then SR on the
  latent" is already published (Cranmer 2020; its raw-Eureqa experiment shows
  raw SR failing on high-D), and the PySR README recommends it for high-D
  data. PySR's own tools (`select_k_features`, units, templates) select or
  constrain inputs but never combine them.
- Core contribution, as agreed with the author: (1) a latent space that is
  closed form by construction (sparse combination of a feature library), so
  only one SR step z -> y is needed; (2) learnable physical structure
  (PSD structural tensors a, A) inside the library, which makes 6D rotated
  Hill solvable for PySR; (3) automatic choice of n (weakest; Cranmer's L1
  model also finds the latent size). Differs from SINDy: the library is a
  bottleneck in front of a nonlinear regressor, not the final model.
- Reviewer risk noted: Hill is sigma:A:sigma, which is in the library, so
  CB2001/Yld2004 (answer not a single library term) must sit next to Hill.
- Comparison methods decided: A ours, B PySR alone, C distillation, D SINDy
  on the same library, E PySR on the full library, F' learned vs true A,
  I Operon +- encoder (run last) required; G `select_k_features` and H AI
  Feynman optional. Fixed A = I ablation dropped (predictable failure).
  Table in `PAPER_PLAN.md`.
- Missing and critical: B on rotated Hill (multi-seed), C everywhere,
  synthetic scaling experiment.

## 34. Experiment matrix for the paper (2026-10-25)

- Goal: one table of datasets x methods to run and analyse; in
  `PAPER_PLAN.md` ("Experiment matrix").
- Datasets agreed with the author: Hill ladder (von Mises, aligned Hill,
  rotated Hill 2D with 3 inputs, rotated Hill 3D with 6 inputs; author
  reports PySR alone recovers all but rotated 3D, results not yet in the
  repo/report), CB2001, Yld2004-18p, synthetic (a) rotated quadratic forms
  and (b) power-law groups, Feynman, LLM-SRBench, a small noise section.
- Noise kept as a short robustness section: it tests our Stage 1 threshold
  and Stage 2 snapping, and SR benchmarks report it; not a headline.
- Risk noted: rotated Hill (y = sqrt(sigma:A:sigma)) is solvable by linear
  regression of y^2 on the 21 quadratic stress monomials, so baseline D
  (SINDy) may solve it. CB2001 and synthetic (a) with nonlinear outer
  functions must carry the "needs a nonlinear step" argument.
- Run order: B on Hill ladder; A + structure recovery on Hill ladder;
  synthetic generators; distillation baseline C; D, E; Feynman 5 seeds;
  noise; Operon.

## 35. Yield datasets inventory and quadratic-fit check (2026-10-25)

- `data/vonMises25.hdf`, `data/vonMises100.hdf`: same von Mises surface at
  yield stress 25 and 100 (Mandel; sigma_eq exactly 25.000 / 100.000). Use
  one in the Hill ladder and the other as a scale check.
- `data/hill.hdf`, `data/rotatedhill.hdf`: 1000 Mandel points each; rotated
  Hill has the same tensor eigenvalues as aligned Hill (a pure rotation).
  All four files contain only points with every component >= 0 (positive
  orthant); consider regenerating on the full sphere for the final runs.
- `data/dx56d_miller.hdf` (402 pts, DX56D steel, CPFEM, Fraunhofer Fordatis
  DOI 10.24406/fordatis/225.2) and `data/single_crystal_gold_ep002.hdf`
  (294 pts, single-crystal gold CPFEM, Zenodo DOI 10.5281/zenodo.21981226,
  yield at 0.2% plastic strain): added 2026-09-19/20, see
  `EXTERNAL_DATASETS.md`. No known closed form; candidates for a realistic
  data row. DOIs not yet verified.
- Rotated Hill 2D (3 inputs) dataset no longer exists; regenerate from the
  recovered tensor (plane-stress block s11, s22, s12).
- Probe `experiments/probes/hill_quadratic_fit.py` (output
  `experiments/outputs/hill_quadratic_fit.json`): linear least squares on the
  21 quadratic stress monomials fits every file exactly (max residual
  <= 2e-13); rotated Hill uses all 21 terms, aligned Hill and von Mises 9.
- Consequence: a SINDy-style linear baseline solves rotated Hill exactly.
  Rotated Hill alone cannot show that a nonlinear step is needed; it shows
  PySR-on-raw-inputs failing and the learned tensor collapsing 21 terms into
  one invariant. The "needs more than a linear library" argument must come
  from CB2001, Yld2004 and the CPFEM data. The fitted M is the ground truth
  for structure recovery (F').

## 36. Experiment tracker (2026-10-26)

- Added `EXPERIMENTS.md`: one row per (dataset, method), IDs
  `<dataset>-<method>` (e.g. `RH3-B`), status, seeds, commit, result, log
  entry. Hill ladder split into VM25, VM100 (scale check), HILL, RH2, RH3.
  CPFEM rows (DX, AU) listed as proposed. Prerequisites P1-P8 (data
  regeneration, baselines C/D/E, synthetic generators, Operon, quota-safe job
  script, common `summary.csv` and table builder).
- Each run will live in `experiments/runs/<ID>/` (README with command and
  commit, frozen config, `summary.csv` with fixed columns).

## 37. P1: Hill ladder on the full sphere (2026-10-26)

- The original Hill-family files contain only points with every component
  >= 0 (entry 35). The author is checking with the supervisor why; until
  then they stay untouched, and all paper experiments use new files with
  points in all directions.
- `scripts/prepare_hill_ladder.py`: recovers M (sigma^T M sigma = 1, Mandel)
  from each original file by least squares on the 21 quadratic monomials
  (residual <= 2e-13), samples 1000 directions uniform on the unit sphere
  (seed 2026, as `prepare_yield_benchmark.py`) and scales them onto the
  surface. Output `data/hill_ladder_full/<name>_full.hdf` + JSON with M:
  vonMises25, vonMises100, hill, rotatedhill (mandel_3d) and rotatedhill2d
  (plane_stress_2d, [s11, s22, s12]: plane-stress section of the rotated-Hill
  tensor; the old 2D file is lost, so this is the same material, not
  necessarily the old 2D data). Max |f - 1| <= 2e-15; 98% of 6D rows and 88%
  of 2D rows have a negative entry. Same directions for all 6D files, so
  vonMises100 = 4 x vonMises25 exactly (scale check).
- Configs `configs/hill_ladder/{vm25,vm100,hill,rh2,rh3}.toml`: copies of
  `configs/adaptive_encoder_rotated_hill.toml` with only data and output
  paths changed. Smoke test (Stage 1, 300 epochs, CPU) runs on rh3 and rh2.
  On this Mac `device = "auto"` picks MPS, which lacks eigh; use CPU locally.
- No shared code changed (new script, data, configs).

## 38. P7: job script within the file quota (2026-10-26)

- Arrhenius project quota: 250,000 files (the old results directory held
  ~51,000; one LLM-SRBench task left ~100 files).
- `cluster/feynman_array.sbatch`: new `KEEP_ARTIFACTS` (default `pack`).
  After a task, its folder keeps `benchmark_summary.json` and one
  `artifacts.tar.gz` without `*.pt`, `*.pkl`, `*.bak`; `summary` keeps the
  summary only, `full` keeps everything. Tested on a copy of a real task
  folder: 2 files remain, archive has 51 entries, none of the excluded types.
  The merge script reads only `benchmark_summary.json`, so it is unaffected.
- `cluster/README.md`: quota note; only the aarch64 venv/depot are needed;
  `make_tasks.py` runs with the system `python3` on the login node.

## 39. P8: common summary format and table builder (2026-10-26)

- `src/invariant_generator/experiment_summary.py` (new module, no existing
  code changed): fixed `summary.csv` columns (as in `EXPERIMENTS.md`), a
  writer that rejects unknown columns, a reader, and a converter from
  `run_feynman_benchmark.py` output (one arm). Pipeline rows without a
  reduction report the PySR-alone result, as the merge script does; with
  target noise `recovered` is the symbolic match.
- `scripts/export_run_summary.py <run_dir> --arm ... --experiment-id ...
  --dataset ... --method ... --commit ...` writes
  `experiments/runs/<ID>/summary.csv`.
- `scripts/build_tables.py`: per-experiment recovered count per seed
  (mean +- sd) and a dataset x method table from all `summary.csv` files.
- Check on the old runs (scratch, not committed): LLM-SRBench pipeline
  [63, 66, 59, 58, 60] = 61.2 +- 2.9, PySR alone 57.2 +- 1.3, Feynman v3
  pipeline 85: identical to entries 27 and 32.
- Yield experiments (adaptive_encoder_pipeline.py) need their own converter;
  added with the yield runner.

## 40. Comparison arms C, D, E; yield runner; synthetic data (P2-P5) (2026-10-26)

- Yield runner (new): `src/invariant_generator/yield_benchmark.py`,
  `scripts/run_yield_benchmark.py`, `cluster/yield_array.sbatch` (task
  lines `<config> <seed>`, same KEEP_ARTIFACTS cleanup). One
  `benchmark_summary.json` per task in the Feynman layout, so export and
  merge scripts apply. Every arm is scored on 5000 fresh surface points
  (new directions on the true surface, scaled by k in the augmentation range)
  against the generating criterion: Hill ladder from the JSON tensor M,
  CB2001/Yld2004 from `yield_criteria.CRITERIA`; recovered = relative L2 <
  1e-6. Datasets without ground truth (DX, AU) are rejected for now.
  Truth functions reproduce every data file (max |f - 1| ~1e-15).
- Arms (opt-in, defaults unchanged): pipeline (A), baseline (B; zero
  columns dropped, so RH2 PySR sees s11, s22, s12), distillation (C),
  sparse_linear (D), library_pysr (E). The same arms exist for supervised
  benchmarks via `run_feynman_benchmark.py --arms ...`
  (`src/invariant_generator/supervised_baselines.py`; hook in
  `feynman_benchmark.run_benchmark`).
  - C (`src/invariant_generator/distillation.py`): MLP (2 x 128 ReLU) -> 8
    latent -> MLP, L1 1e-2 on latent activations, keep up to 4 components
    with std > 1% of the largest, PySR for each z_j = h_j(x) and for
    y = g(z); PySR time split over the k + 1 searches; no constant refit.
  - D: STLSQ on a degree-3 polynomial library of the raw inputs, target y^m
    (m = 1, 2, 3), (m, threshold) chosen on a 10% holdout. Stronger than
    "SINDy on our invariant library" (which, with fixed tensors, is
    isotropic); chosen because it is the honest worst case for Hill.
  - E: PySR on the pipeline's feature library without the encoder; for
    stress invariants the structural tensors are fixed to identity and
    vanishing/duplicate columns dropped, leaving I1, I2, I3, I13.
- Smoke results (1-minute PySR budgets, not results):
  - D recovers VM25, HILL, RH2, RH3 exactly (9, 9, 6, 21 active terms,
    m = 2, fresh rel. L2 ~3e-11) and fails on CB2001 (2.1e-2); on SA it
    recovers sqrt(q1) (d = 4) and fails on q1/(1 + q2) (7.6e-2).
  - B on RH2 in ~4.5 min: 4.5e-2 (not recovered). E on RH3: 8.4e-2.
  - C on RH3: all latent components shrank to std ~1e-3 (L1 scale
    collapse); composed formula 0.30. C needs a dev run (e.g. normalise the
    latent or L1 relative to decoder weight norms) before the experiment.
  - A (pipeline) on RH2 with 500/300 epochs runs end to end (path check).
  - Supervised on SB d = 4 (x1^2/x4 + x2^2 x3): E recovered; C 0.15; D 1.3e-2.
- Synthetic data (P5): `scripts/prepare_synthetic_benchmark.py` ->
  `data/synthetic/{sa,sb,sbd}` (gitignored, 37 MB; regenerate on the
  cluster). SA: 70 problems, d = 2-12, sqrt(q1) (n* = 1, SINDy control),
  q1/(1 + q2), sqrt(q1) + q2 (n* = 2). SB: 50 problems, d = 4-12, k = 1-3
  power-law groups. SBD: SB + 3 distractors. `log_reducibility` = n*.
  `configs/synthetic_sa.toml`: v3 with identity targets, product features
  (x_i, x_i x_j) and gated Stage 2 with caps up to 90 (dense rows); first
  version, needs a dev run. SB uses `configs/feynman_benchmark_v3.toml`.
- Not done: P6 (Operon). Yld2004 has no config yet (data from
  `prepare_yield_benchmark.py`). The PySR time limit overshoots (2 min
  limit took ~4.5 min), so budgets are approximate.
- Existing tests pass. No Hill-critical file changed (new modules; one new
  opt-in branch in `feynman_benchmark.py`), so the Hill old-vs-new check was
  not required.

## 41. Operon postponed; yield runs keep the author's PySR settings (2026-10-26)

- pyoperon (0.6.0-0.6.2) has wheels for x86_64 Linux, macOS arm64 and
  Windows only; none for Linux aarch64 (Arrhenius GPU nodes), not on
  conda-forge either. The macOS wheels fail to import (they link to
  libz/libxxhash in the maintainers' CI environment). Added an `operon`
  extra (`pyoperon==0.6.1`, x86_64 Linux only) so the dependency is
  recorded; the Operon arms are not written until there is an x86_64 Linux
  machine to test on. P6 and all method-I rows postponed by the author.
- `uv sync` with explicit extras removed two undeclared packages from the
  local env (huggingface-hub, truststore); no project code imports them.
- Author's workflow: PySR runs use the tuned yield configs and are stopped by
  hand (q + Enter) when the loss has converged. The yield runner no longer
  overrides the config's PySR stop condition (it used var(y) * 1e-12).
  `cluster/interactive_env.sh` sets up the environment inside an interactive
  job. For the comparison, stopping times go in each run README.

## 42. VM25-A: first paper run, von Mises 25 with our pipeline (2026-10-26)

- Run folder `experiments/runs/VM25-A/` (README, config, summary.csv).
  Arrhenius interactive node, seed 42; PySR stopped by hand after ~1 min.
- Stage 1 n = 1; Stage 2 one active term J1 = -0.1626 * I11; Stage 3
  y = -0.4256 * J1 (complexity 4, loss 2.1e-10). Learned A vs the true von
  Mises tensor: cosine 0.9999999995; eigenvalues 5e-5 (true 0) and
  0.99994-1.0 (true 1). Fresh relative L2 1.31e-5 (test 1.34e-5).
- Not "recovered" under the 1e-6 threshold, although every structural
  element is right. The error is the learned tensor's: the best constant times
  sqrt(sigma:A_learned:sigma) is off by exactly 1.30e-5, so PySR stopping
  time is irrelevant. PySR alone fits the 9 (or 21) quadratic coefficients
  directly in float64, so the strict threshold favours it on the Hill ladder.
- Decision needed (author): how to score yield recovery before more runs:
  keep 1e-6 and add a structure criterion, or add an opt-in float64 refit of
  the learned tensor at the end of Stage 2/3.
- Code: `expanded_equation` now set for the yield pipeline arm; seed found
  from the full path in `experiment_summary.benchmark_rows`.

## 43. VM25-B and VM25-D (2026-10-26)

- VM25-B (PySR alone, Arrhenius, stopped by hand after 136 s, loss 5.1e-15):
  exact von Mises form found at complexity 32-36, fresh relative L2 7.4e-8,
  recovered. Expected (ladder rung 1).
- VM25-D (sparse linear, run locally, 0.03 s): y^2 = 9 quadratic terms,
  fresh relative L2 2.9e-11, recovered.
- Von Mises so far: B and D recovered; A has the right structure but 1.3e-5
  (entry 42). Next on the cluster: VM25-E; VM25-C waits for the distillation
  dev fix.

## 44. Time to recovery for hand-stopped PySR runs (2026-10-26)

- Author's request: log when a run recovered, not only when it was stopped.
- PySR 1.5.10 writes hall_of_fame.csv only at the end of the search (checked:
  no file during a 3-min run), so polling it cannot time recovery. It does
  evaluate the early-stop condition throughout the search, so
  `invariant_generator.pysr_watch.LossWatcher` extends the config's rule: the
  first time loss < threshold, Julia touches a marker file; the stop rule
  itself (e.g. loss < 1e-17) is unchanged. Time to recovery = marker mtime -
  start of fit (includes Julia start-up, as pysr_seconds does).
- Threshold = PySR training loss equivalent to relative L2 1e-6: 1e-12 for
  relative-squared weights (yield configs), 1e-12 * mean(y^2) for MSE.
  It is the training loss, so it approximates the fresh-point criterion.
- Wired into the yield arms baseline, pipeline (time to recovery includes
  Stages 1-2) and library_pysr; new `summary.csv` column
  `time_to_recovery_s`. Test: y = 2 x0 x1 with stop rule loss < 1e-30 and a
  40 s limit: marker at 3.6 s, run returned at 43.6 s.
- Not recorded for VM25-A/B (run before this change).

## 45. VM25-E and VM25-F (2026-10-26)

- VM25-E (PySR on the fixed isotropic library I1, I2, I3, I13; stopped by
  hand after 62 s): exact von Mises form sqrt(0.0024 F2^2 - 0.0008 F1^2) at
  complexity 12 (loss 1.3e-13); selected equation fresh relative L2 1.7e-8,
  recovered. Raw components (VM25-B) needed complexity 32-36.
- VM25-F recorded from VM25-A (tensor cosine 0.9999999995).
- Timing: the author will rerun these with the time-to-recovery logging;
  until then no time comparisons.
- Von Mises row: A structure right (1.3e-5), B, D, E recovered; C waits for
  the distillation fix; I postponed.

## 46. Hill ladder: early-stopping patience 100 -> 50 (2026-10-26)

- Author's request: Stage 1/2 training waited too long before stopping.
  Patience counts evaluations, one every log_every = 50 epochs, so 100 meant
  5000 epochs without improvement; now 50 (2500 epochs; the LR plateau
  patience of 20 evaluations still halves the LR first).
- Changed only `configs/hill_ladder/*.toml`; `configs/adaptive_encoder_rotated_hill.toml`
  (the Hill reference) is unchanged. VM25-A ran with 100; the author will
  rerun it later.

## 47. VM100-A (scale check) and a fix to the time-to-recovery marker (2026-10-26)

- VM100-A (patience 50, PySR stopped by hand after 85 s): n = 1, J1 =
  -0.0400 * I11, y = J1 / -2.2928; tensor cosine 0.9999999995 (as VM25-A);
  fresh relative L2 1.35e-5 (VM25-A 1.31e-5). Scale check passed.
- Bug found: the marker fired at 19 s although the final best loss was
  2.2e-10. The Hill configs use batching = true, so the stop condition sees
  minibatch losses. Fix in `pysr_watch.LossWatcher`: a marker counts only if
  the final hall of fame (full-data losses) is below the threshold; otherwise
  `time_to_recovery` is empty and `marker_ignored_minibatch` is true. With
  batching the recorded time is the first minibatch hit, so approximate.

## 48. VM100-B (2026-10-26)

- PySR alone on von Mises 100, stopped by hand after 222 s (loss 3.9e-12):
  exact von Mises form with constants 1.7320542 (sqrt 3), 0.3333320 (1/3),
  3333.37 (10000/3); fresh relative L2 2.2e-6, so not recovered under 1e-6.
  At scale 25 (VM25-B) it reached 7e-8 in 136 s.
- Observation: three of five yield runs so far have the exact structure with
  errors between 2e-6 and 1.4e-5 (VM25-A, VM100-A, VM100-B). The 1e-6
  threshold alone hides that; a "structure recovered" column (exact form up
  to constants) would separate right-but-imprecise from wrong formulas. Kept
  as is for now (author); revisit before the tables go into the paper.

## 49. HILL-A, HILL-F; yield recovery threshold 1e-6 -> 1e-4 (2026-10-27)

- HILL-A (rerun after a lost connection; run folder
  `experiments/runs/HILL-A/`): Stage 1 n = 1 (964 s), Stage 2 one active term
  J1 = -1.58264 * I11 (694 s), Stage 3 y = -0.43579 * J1 (complexity 4,
  loss 1.6e-10, stopped by hand after 243 s). Fresh relative L2 1.14e-5
  (test 1.30e-5).
- HILL-F with the new `experiments/probes/structure_recovery.py` (output
  `experiments/outputs/structure_recovery_hill.json`): learned A vs the true
  Hill tensor cosine 0.9999999996; eigenvalues / max 2.6e-5 (true 0) and
  0.53790, 0.59765, 0.65742, 0.79295, 1 (true 0.53789, 0.59766, 0.65742,
  0.79297, 1). Error floor of c * sqrt(sigma:A_learned:sigma) on random
  directions 9.5e-6, so the 1.1e-5 is the learned tensor's float32
  precision. Same script on VM25-A reproduces cosine 0.9999999995.
- Decision (author): a PySR loss of ~1e-10 already means the equation was
  found; the PySR community usually treats loss < 1e-7 as found. With
  `inverse_target_squared` weights the loss is ~ (relative error)^2, so
  1e-7 is a relative error of ~3e-4. Yield recovery threshold set to
  relative L2 < 1e-4 (loss ~1e-8, a margin below 1e-7), applied the same to
  every method: `RECOVERY_THRESHOLD` in `yield_benchmark.py`, the
  time-to-recovery marker now fires at loss 1e-8. Equation datasets keep
  1e-6 (exponents snapped to exact rationals there). Caveat for the paper:
  loss thresholds in the PySR docs are for the plain MSE in the target's
  units, so the paper must state the criterion as a relative error.
- Re-marked: VM25-A (1.3e-5), VM100-A (1.35e-5), VM100-B (2.2e-6) now
  recovered; VM25-B/D/E unchanged (recovered). The structure-score column
  (F') stays as the second, structural criterion.

## 50. HILL-B and HILL-D (2026-10-27)

- HILL-B (PySR alone, 6 raw components; `experiments/runs/HILL-B/`):
  stopped by hand after 1116 s at loss 4.7e-5, complexity 38; the author had
  seen this plateau in earlier runs. Fresh relative L2 8.8e-3 (test 7.4e-3):
  not recovered. The equation has the Hill shape (squared stress differences
  plus squared shears); its quadratic form matches the true M within 0.1% in
  the normal block and 0.3% for s23, but s12 and s13 share one coefficient
  (truth 0.22 and 0.20 in Mandel), which is the whole error. Our pipeline
  (HILL-A) gets the exact form at complexity 4.
- First difference between A and B on the Hill ladder: on von Mises (VM25-B)
  PySR alone recovered. One seed only; more seeds needed before a claim.
- HILL-D (sparse linear, run locally, 0.03 s): exact 9-term quadratic,
  4.7e-11. As for VM25-D, expected: Hill surfaces are quadratic forms.

## 51. HILL-E (2026-10-27)

- PySR on the fixed library (identity tensors: I1, I2, I3, I13), no
  encoder; `experiments/runs/HILL-E/`. Stopped by hand after 1330 s, loss
  plateau 3.1e-3; selected y = -0.0939 F1^2/F4 + F2/2.032, fresh relative L2
  6.1e-2. Not recovered.
- Expected by construction: identity-tensor invariants are isotropic and
  cannot represent an anisotropic surface. Hill ladder so far (seed 42):
  A recovered (structure exact), B one constant short (7.4e-3), D exact
  (linear fit of a quadratic form), E impossible (6%). E vs A isolates the
  learned structural tensor (contribution 2); on isotropic VM25 E recovered.

## 52. RH2-A and RH2-F (2026-10-27)

- Rotated Hill, plane stress (s11, s22, s12), our pipeline;
  `experiments/runs/RH2-A/`. Stage 1 n = 1 (591 s), Stage 2 J1 = -1.19932 *
  I11 (694 s), Stage 3 y = J1 / -1.7472 (complexity 4, loss 1.2e-12, PySR
  reached 1e-8 after 20 s, stopped by hand at 54 s). Fresh relative L2
  1.18e-6: recovered. Time to recovery 1306 s, almost all encoder.
- RH2-F: `structure_recovery.py` now compares only the s11, s22, s12 block
  when the truth is plane stress (3x3). Cosine 0.99999999999, eigenvalues
  within 4e-6. Ten times more precise than the 6D runs (3x3 block vs 6x6).
- RH2-D (sparse linear, run locally, 0.01 s): exact 6-term quadratic,
  4.5e-11.

## 53. RH2-B (2026-10-27)

- PySR alone on rotated Hill 2D (s11, s22, s12); `experiments/runs/RH2-B/`.
  Stopped by hand after 468 s at loss 2.1e-8, complexity 32. Fresh relative
  L2 1.39e-4 (test 1.52e-4): just above the 1e-4 threshold, not recovered.
- Same failure pattern as HILL-B: right quadratic shape, but s11^2 and s22^2
  share one coefficient (truth differs by 0.12%). Borderline: under a loss
  < 1e-7 rule it would count as found; under the relative-error criterion
  fixed in entry 49 it does not. Kept as is (criterion not changed per run);
  more seeds will decide. RH2-A: 1.1e-6 at complexity 4.
