# Experiments tracker

Master table of every experiment for the paper. Plan and rationale:
`PAPER_PLAN.md` ("Experiment matrix"); the narrative goes in
`RESEARCH_LOG.md`, and each entry names the experiment IDs it concerns.
The interpretation of every result (what it means, why, caveats, which
contribution it supports), written for the paper, goes in
`experiments/ANALYSIS.md`, updated after every run.

## How to use

- **ID** = `<dataset>-<method>`, e.g. `RH3-B` = rotated Hill 3D, PySR alone.
- **Status**: `planned` -> `running` -> `done`, or `failed` (say why in the
  folder README) or `blocked` (name the prerequisite, e.g. P2).
- Each run gets a folder `experiments/runs/<ID>/` (committed):
  - `README.md`: goal, exact command(s), config, commit, seeds, expected
    outcome, conclusion;
  - `config.toml`: the frozen config used;
  - `summary.csv`: one row per (instance, seed), columns below.
  Heavy outputs (models, PySR checkpoints) are not kept.
- Every result must be reproducible from the command and commit in its README.
- Single-seed differences of +-2 equations are PySR noise; claim effects only
  over 5 seeds. All arms get the same PySR budget.

### `summary.csv` columns (same for every experiment)

`experiment_id, dataset, instance, method, seed, commit, recovered,
test_rel_l2, test_r2, equation, complexity, n_selected, n_true,
structure_score, time_encoder_s, time_sr_s, time_to_recovery_s, notes`

- `instance`: surface name, equation name, or `d=6`, etc.
- `recovered`: yield datasets: relative L2 < 10^-3.5 (~3.2e-4) on fresh
  points (PySR loss ~1e-7 with relative weights; RESEARCH_LOG 68); equation datasets:
  relative L2 < 1e-6 or symbolic match; for noisy runs, symbolic match only.
- `time_sr_s`: PySR wall time until it returned (stopped by hand or by its
  own limit). `time_to_recovery_s`: time until PySR's best training loss first
  reached the recovery level (loss ~ threshold^2 relative): the config's stop
  rule is extended to drop a marker file at that level (the stop rule itself
  is unchanged; `invariant_generator.pysr_watch`); for our
  pipeline it includes Stages 1-2. Empty if never reached or not recorded
  (runs before 2026-10-26 evening).
- `structure_score`: F' metric (e.g. cosine of learned vs true tensor or
  rows; exponent match); empty if not applicable.

## Methods

| Code | Method | Notes |
|---|---|---|
| A | Ours + PySR | full pipeline (Stage 1-3) |
| B | PySR alone | raw inputs |
| C | Symbolic distillation (Cranmer 2020) | MLP with L1 latent, PySR on the latent; needs P2 |
| D | Sparse linear library regression (SINDy-style) | fixed library, no network; needs P3 |
| E | PySR on the feature library, no encoder | tensor inputs: structural tensors fixed (identity); needs P4 |
| F' | Structure recovery | learned tensor / rows / exponents / n vs ground truth; computed from A |
| G | PySR `select_k_features` | only meaningful with distractor inputs (optional) |
| H | AI Feynman | cite reported numbers, not run (optional) |
| I | Operon with and without our encoder | needs P6 |

## Datasets

| Code | Dataset | Inputs | Source / generator | Status |
|---|---|---|---|---|
| VM25 | von Mises, yield stress 25 | 6 | `data/hill_ladder_full/vonMises25_full.hdf`, config `configs/hill_ladder/vm25.toml` | exists |
| VM100 | von Mises, yield stress 100 (scale check) | 6 | `data/hill_ladder_full/vonMises100_full.hdf`, `configs/hill_ladder/vm100.toml` | exists |
| HILL | Hill, aligned axes | 6 | `data/hill_ladder_full/hill_full.hdf`, `configs/hill_ladder/hill.toml` | exists |
| RH2 | Rotated Hill, plane stress (s11, s22, s12) | 3 | `data/hill_ladder_full/rotatedhill2d_full.hdf` (plane-stress section of the RH3 tensor), `configs/hill_ladder/rh2.toml` | exists |
| RH3 | Rotated Hill, 3D | 6 | `data/hill_ladder_full/rotatedhill_full.hdf`, `configs/hill_ladder/rh3.toml` | exists |
| CB | CB2001, 2090-T3 | 6 | `data/yield_benchmarks/cb2001_2090T3.hdf` (`scripts/prepare_yield_benchmark.py`), `configs/cb/cb2001_2090T3.toml` | exists |
| YLD | Yld2004-18p | 6 | `scripts/prepare_yield_benchmark.py` | exists |
| DX | DX56D steel, CPFEM (proposed) | 6 | `data/dx56d_miller.hdf`, see `EXTERNAL_DATASETS.md` | exists; row not yet agreed |
| AU | Single-crystal gold, CPFEM (proposed) | 6 | `data/single_crystal_gold_ep002.hdf` | exists; row not yet agreed |
| SA | Synthetic (a): rotated quadratic forms, d = 2, 3, 4, 6, 8, 10, 12 (70 problems) | 2-12 | `scripts/prepare_synthetic_benchmark.py` -> `data/synthetic/sa` (gitignored, regenerate), config `configs/synthetic_sa.toml` | exists |
| SB | Synthetic (b): power-law groups, d = 4-12, k = 1-3 groups (50 problems) | 4-12 | same script -> `data/synthetic/sb`, config `configs/feynman_benchmark_v3.toml` with `--data-dir` | exists |
| SBD | Synthetic (b) plus 3 distractor inputs (50 problems) | 7-15 | same script -> `data/synthetic/sbd` | exists |
| FEY | Feynman (110 equations) | 1-9 | `scripts/prepare_feynman_benchmark.py` | exists |
| LLM | LLM-SRBench LSR-Transform (91 positive-input problems) | 2-10 | `scripts/prepare_llm_srbench.py` | exists (gated data) |
| NRH3 | Rotated Hill 3D with input noise, 1-2 levels | 6 | from RH3 | to generate |
| NFEY | Feynman subset with target noise, 1-2 levels | 1-9 | `--target-noise` | exists |

Note: the original Hill-family files (`data/vonMises25.hdf`, `vonMises100.hdf`,
`hill.hdf`, `rotatedhill.hdf`) contain only points with every component >= 0.
They are kept untouched (the reason for positive-only data is being checked
with the supervisor). All paper experiments use the full-sphere files above.

## Prerequisites

| ID | Task | Status |
|---|---|---|
| P1 | Generator for the Hill ladder on the full sphere (VM25, VM100, HILL, RH2, RH3) from the recovered tensors; new files | done: `scripts/prepare_hill_ladder.py`, `data/hill_ladder_full/`, `configs/hill_ladder/` (RESEARCH_LOG 37) |
| P2 | Implement distillation baseline (C) | done, needs a dev run (latent collapse in scale): `invariant_generator/distillation.py`; arm `distillation` (RESEARCH_LOG 40) |
| P3 | Implement sparse linear library baseline (D) | done: arm `sparse_linear` (STLSQ, y^m = polynomial of raw inputs, degree 3, m = 1, 2, 3) (RESEARCH_LOG 40) |
| P4 | Opt-in "no encoder" mode: PySR on the feature library (E) | done: arm `library_pysr` (tensor library: a = A = identity, duplicates dropped) (RESEARCH_LOG 40) |
| P5 | Synthetic generators SA, SB, SBD with ground truth metadata | done: `scripts/prepare_synthetic_benchmark.py`, `configs/synthetic_sa.toml` (SA config needs a dev run) (RESEARCH_LOG 40) |
| P6 | Operon backend (I) | postponed: pyoperon 0.6.x runs only on x86_64 Linux (no aarch64 build; macOS wheels do not load); `operon` extra added, arms not written (RESEARCH_LOG 41) |
| P7 | Job script keeps only summaries; deletes models and PySR checkpoints (file quota) | done: `KEEP_ARTIFACTS` in `cluster/feynman_array.sbatch` (RESEARCH_LOG 38) |
| P8 | Common `summary.csv` writer and a script that builds paper tables from `experiments/runs/` | done: `invariant_generator/experiment_summary.py`, `scripts/export_run_summary.py`, `scripts/build_tables.py` (RESEARCH_LOG 39) |

Any change to shared code passes `experiments/probes/hill_old_vs_new.py`
(prints `IDENTICAL`) before it is committed.

## How to run

- Yield datasets (Hill ladder, CB, YLD): `scripts/run_yield_benchmark.py`
  (locally) or `cluster/yield_array.sbatch` (task lines `<config> <seed>`).
  Arms: `pipeline` (A), `baseline` (B), `distillation` (C), `sparse_linear`
  (D), `library_pysr` (E). Recovered = relative L2 < 3.2e-4 on 5000 fresh
  surface points against the generating criterion.
- Equation datasets (FEY, LLM, SA, SB, SBD, NFEY): `scripts/run_feynman_benchmark.py`
  / `cluster/feynman_array.sbatch` with the same arm names (plus `portfolio`).
- Then `scripts/export_run_summary.py` -> `experiments/runs/<ID>/summary.csv`,
  and `scripts/build_tables.py`.
- Every arm gets the same PySR time limit (`--timeout-minutes`); distillation
  splits it over its k + 1 searches.
- Yield runs keep the config's own PySR settings (e.g. 24 h limit, stop at
  loss < 1e-17) and are usually stopped by hand (`q` + Enter) once the loss
  has converged. For the paper comparison, record in the run README when and
  why each run was stopped; the PySR time is saved as `pysr_seconds`. A
  hand-stopped arm must not get less time than the arm it is compared with.
- Interactive runs on Arrhenius: inside `srun ... --pty bash`, run
  `source cluster/interactive_env.sh`, then `"$PYTHON" scripts/...`.

## Experiments

Req: R = required, o = optional. Status for all rows starts at `planned`.

### 1. Hill ladder

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| VM25-A | von Mises 25 | Ours | R | done | 1/5 (42) | `32a258a` | recovered (1.3e-5); I11, tensor cosine 0.9999999995 | 42, 49 |
| VM25-B | von Mises 25 | PySR alone | R | done | 1/5 (42) | 5d7c710+ | recovered (7.4e-8), 136 s PySR | 43 |
| VM25-C | von Mises 25 | Distillation | R | done | 1/5 (42) | `5e21412` | not recovered (3.9e-2); latent = von Mises invariant, warped; outer cubic | 59 |
| VM25-D | von Mises 25 | Sparse linear | R | done | 1/5 (42) | `bcd8e18` | recovered (2.9e-11), 9 terms | 43 |
| VM25-E | von Mises 25 | PySR + library | R | done | 1/5 (42) | 501c5f9+ | recovered (1.7e-8); exact form at complexity 12 | 45 |
| VM25-F | von Mises 25 | Structure recovery | R | done | 1/5 (42) | `32a258a` | n = 1, I11, tensor cosine 0.9999999995 | 42 |
| VM25-I | von Mises 25 | Operon +- encoder | o | blocked (P6) | | | | |
| VM100-A | von Mises 100 (scale check) | Ours | R | done | 1/5 (42) | `2c33f0e` | recovered (1.35e-5); same structure as VM25-A | 47, 49 |
| VM100-B | von Mises 100 (scale check) | PySR alone | o | done | 1/5 (42) | `9248eab` | recovered (2.2e-6), exact form; 222 s | 48, 49 |
| HILL-A | Hill aligned | Ours | R | done | 1/5 (42) | `1b22c97` | recovered (1.3e-5); n = 1, I11, tensor cosine 0.9999999996 | 49 |
| HILL-B | Hill aligned | PySR alone | R | done | 1/5 (42) | `dcb66db` | not recovered (7.4e-3, loss 4.7e-5, 1116 s); Hill shape, s12/s13 share one constant | 50 |
| HILL-C | Hill aligned | Distillation | R | done | 1/5 (42) | `3ae90d6` | not recovered (1.4e-2); latent = Hill form, constants off; outer linear | 61 |
| HILL-D | Hill aligned | Sparse linear | R | done | 1/5 (42) | `dcb66db` | recovered (4.7e-11), 9 terms | 50 |
| HILL-E | Hill aligned | PySR + library | R | done | 1/5 (42) | `733a827` | not recovered (5.8e-2, loss plateau 3.1e-3); isotropic library cannot express Hill | 51 |
| HILL-F | Hill aligned | Structure recovery | R | done | 1/5 (42) | `1b22c97` | n = 1, I11, cosine 0.9999999996; 5 eigenvalues within 2e-5 | 49 |
| HILL-I | Hill aligned | Operon +- encoder | o | blocked (P6) | | | | |
| RH2-A | Rotated Hill 2D | Ours | R | done | 1/5 (42) | `22e0d10` | recovered (1.1e-6); n = 1, I11; PySR 20 s to loss 1e-8 | 52 |
| RH2-B | Rotated Hill 2D | PySR alone | R | done | 1/5 (42) | `def3529` | recovered at 3.2e-4 (fresh 1.4e-4, loss 2.1e-8, 468 s); one constant short | 53, 68 |
| RH2-C | Rotated Hill 2D | Distillation | R | done | 1/5 (42) | `9274664` | not recovered (1.1e-2); latent = quadratic, constants off; outer sqrt | 62 |
| RH2-D | Rotated Hill 2D | Sparse linear | R | done | 1/5 (42) | `b229b38` | recovered (4.5e-11), 6 terms | 52 |
| RH2-E | Rotated Hill 2D | PySR + library | R | done | 1/5 (42) | `4e4e007` | not recovered (6.6e-2, loss plateau 2.5e-3); isotropic library | 54 |
| RH2-F | Rotated Hill 2D | Structure recovery | R | done | 1/5 (42) | `22e0d10` | cosine 0.99999999999 (plane-stress block) | 52 |
| RH2-I | Rotated Hill 2D | Operon +- encoder | o | blocked (P6) | | | | |
| RH3-A | Rotated Hill 3D | Ours | R | done | 1/5 (42) | `de42785` | recovered (1.1e-5); n = 1, I11; PySR 20 s to loss 1e-8 | 55 |
| RH3-B | Rotated Hill 3D | PySR alone | R | done | 1/5 (42) | `7c0822f` | not recovered (5.8e-2, loss plateau 2.2e-3, 1056 s); no rotation terms | 56 |
| RH3-B-max | Rotated Hill 3D | PySR alone, maxsize 150 (cap check; `configs/hill_ladder/rh3_maxsize150.toml`) | R | done | 1/5 (42) | `6e229e7` | not recovered (3.4e-2) after 92 min; never beyond complexity 80 | 65 |
| RH3-C | Rotated Hill 3D | Distillation | R | done | 1/5 (42) | `1079f13` | not recovered (5.4e-2); latent search fails like B (no rotation terms) | 63 |
| RH3-D | Rotated Hill 3D | Sparse linear | R | done | 1/5 (42) | `de42785` | recovered (4.1e-11), 21 terms | 55 |
| RH3-E | Rotated Hill 3D | PySR + library | R | done | 1/5 (42) | `1618c9b` | not recovered (5.7e-2, plateau 2.8e-3); returns von Mises form | 57 |
| RH3-F | Rotated Hill 3D | Structure recovery | R | done | 1/5 (42) | `de42785` | cosine 0.9999999997; eigenvalues within 2e-5 | 55 |
| RH3-I | Rotated Hill 3D | Operon +- encoder | o | blocked (P6) | | | | |

Known: a linear fit on the 21 quadratic monomials recovers every surface in
this group exactly (RESEARCH_LOG 35), so D is expected to succeed here.

### 2. Anisotropic criteria without a single-invariant answer

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| CB-A | CB2001 | Ours | R | done | 1/5 (42) | `8466a4d` | recovered (fresh 2.9e-4 < 3.2e-4); exact form at complexity 13, n = 2; floor = learned J3^0 | 22, 67, 68 |
| CB-B | CB2001 | PySR alone | R | done | 1/5 (42) | `35a378d` | not recovered (2.4e-2, loss plateau 3.6e-4); quadratic only, no J3^0 | 69 |
| CB-C | CB2001 | Distillation | R | planned | | | | |
| CB-D | CB2001 | Sparse linear | R | done | 1/5 (42) | `ecaa45a` | recovered (1.6e-9): y^6 is a degree-6 polynomial; 138 terms | 66 |
| CB-E | CB2001 | PySR + library | R | planned | | | | |
| CB-F | CB2001 | Structure recovery (rows vs J2^0, J3^0) | R | done | 1/5 (42) | `8466a4d` | cosines 0.99999993 (J2^0), 0.999994 (J3^0) | 67 |
| CB-I | CB2001 | Operon +- encoder | o | blocked (P6) | | | | |
| YLD-A | Yld2004-18p | Ours | R | planned | | | | |
| YLD-B | Yld2004-18p | PySR alone | R | planned | | | | |
| YLD-C | Yld2004-18p | Distillation | R | planned | | | | |
| YLD-D | Yld2004-18p | Sparse linear | o | planned | | | | |
| YLD-E | Yld2004-18p | PySR + library | o | planned | | | | |
| YLD-I | Yld2004-18p | Operon +- encoder | o | blocked (P6) | | | | |

### 3. Crystal-plasticity data (proposed, not yet agreed)

No known closed form: report accuracy and compactness, not recovery.

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| DX-A | DX56D | Ours | o | planned | | | | |
| DX-B | DX56D | PySR alone | o | planned | | | | |
| DX-C | DX56D | Distillation | o | planned | | | | |
| DX-D | DX56D | Sparse linear | o | planned | | | | |
| AU-A | Gold | Ours | o | planned | | | | |
| AU-B | Gold | PySR alone | o | planned | | | | |
| AU-C | Gold | Distillation | o | planned | | | | |
| AU-D | Gold | Sparse linear | o | planned | | | | |

### 4. Synthetic scaling

One experiment covers all d values; `instance` in `summary.csv` holds d and
the equation index (about 10 equations per d, 5 seeds).

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| SA-A | Synthetic (a) | Ours | R | planned | | | | |
| SA-B | Synthetic (a) | PySR alone | R | planned | | | | |
| SA-C | Synthetic (a) | Distillation | R | planned | | | | |
| SA-D | Synthetic (a) | Sparse linear | R | planned | | | | |
| SA-E | Synthetic (a) | PySR + library | R | planned | | | | |
| SA-F | Synthetic (a) | Structure recovery (A vs true) | R | blocked (SA-A) | | | | |
| SA-I | Synthetic (a) | Operon +- encoder | R | blocked (P6) | | | | |
| SB-A | Synthetic (b) | Ours | R | planned | | | | |
| SB-B | Synthetic (b) | PySR alone | R | planned | | | | |
| SB-C | Synthetic (b) | Distillation | R | planned | | | | |
| SB-D | Synthetic (b) | Sparse linear | o | planned | | | | |
| SB-E | Synthetic (b) | PySR + library | R | planned | | | | |
| SB-F | Synthetic (b) | Structure recovery (n, exponents) | R | blocked (SB-A) | | | | |
| SB-I | Synthetic (b) | Operon +- encoder | R | blocked (P6) | | | | |
| SBD-A | Synthetic (b) + distractors | Ours | o | planned | | | | |
| SBD-G | Synthetic (b) + distractors | PySR select_k_features | o | planned | | | | |

### 5. Equation benchmarks

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| FEY-A | Feynman | Ours | R | planned (old: seed 42, 85/110) | | | | 27 |
| FEY-B | Feynman | PySR alone | R | planned (old: seed 42, 86/110) | | | | 27 |
| FEY-C | Feynman | Distillation | R | planned | | | | |
| FEY-D | Feynman | Sparse linear | o | planned | | | | |
| FEY-E | Feynman | PySR + library | o | planned | | | | |
| FEY-F | Feynman | Structure recovery (n vs true groups) | R | blocked (FEY-A) (old: 93/102, 8/8) | | | | 27 |
| FEY-G | Feynman | PySR select_k_features | o | planned | | | | |
| FEY-H | Feynman | AI Feynman (cited, not run) | o | planned | | | | |
| FEY-I | Feynman | Operon +- encoder | o | blocked (P6) | | | | |
| LLM-A | LLM-SRBench | Ours | R | planned (old: 61.2 +- 2.9 of 91) | | | | 32 |
| LLM-B | LLM-SRBench | PySR alone | R | planned (old: 57.2 +- 1.3 of 91) | | | | 32 |
| LLM-C | LLM-SRBench | Distillation | o | planned | | | | |
| LLM-F | LLM-SRBench | Structure recovery | o | blocked (LLM-A) | | | | |
| LLM-I | LLM-SRBench | Operon +- encoder | o | blocked (P6) | | | | |

### 6. Noise robustness

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| NRH3-A | Rotated Hill 3D, noisy | Ours | R | planned | | | | |
| NRH3-B | Rotated Hill 3D, noisy | PySR alone | R | planned | | | | |
| NRH3-C | Rotated Hill 3D, noisy | Distillation | o | planned | | | | |
| NRH3-F | Rotated Hill 3D, noisy | Structure recovery | o | blocked (NRH3-A) | | | | |
| NFEY-A | Feynman subset, noisy | Ours | R | planned | | | | |
| NFEY-B | Feynman subset, noisy | PySR alone | R | planned | | | | |
