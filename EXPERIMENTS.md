# Experiments tracker

Master table of every experiment for the paper. Plan and rationale:
`PAPER_PLAN.md` ("Experiment matrix"); the narrative goes in
`RESEARCH_LOG.md`, and each entry names the experiment IDs it concerns.

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
structure_score, time_encoder_s, time_sr_s, notes`

- `instance`: surface name, equation name, or `d=6`, etc.
- `recovered`: exact recovery (relative L2 < 1e-6 on fresh points, or
  symbolic match); for noisy runs, symbolic match only.
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
| CB | CB2001, 2090-T3 | 6 | `scripts/prepare_yield_benchmark.py` | exists |
| YLD | Yld2004-18p | 6 | `scripts/prepare_yield_benchmark.py` | exists |
| DX | DX56D steel, CPFEM (proposed) | 6 | `data/dx56d_miller.hdf`, see `EXTERNAL_DATASETS.md` | exists; row not yet agreed |
| AU | Single-crystal gold, CPFEM (proposed) | 6 | `data/single_crystal_gold_ep002.hdf` | exists; row not yet agreed |
| SA | Synthetic (a): rotated quadratic forms with nonlinear outer functions, d = 2, 3, 4, 6, 8, 10, 12 | 2-12 | to write (P5) | to generate |
| SB | Synthetic (b): power-law groups, d = 4-12, k = 1-3 groups | 4-12 | to write (P5) | to generate |
| SBD | Synthetic (b) plus 2-4 distractor inputs | 6-16 | to write (P5) | to generate |
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
| P2 | Implement distillation baseline (C) | planned |
| P3 | Implement sparse linear library baseline (D) | planned |
| P4 | Opt-in "no encoder" mode: PySR on the feature library (E) | planned |
| P5 | Synthetic generators SA, SB, SBD with ground truth metadata | planned |
| P6 | Operon backend (I) | planned |
| P7 | Job script keeps only summaries; deletes models and PySR checkpoints (file quota) | done: `KEEP_ARTIFACTS` in `cluster/feynman_array.sbatch` (RESEARCH_LOG 38) |
| P8 | Common `summary.csv` writer and a script that builds paper tables from `experiments/runs/` | planned |

Any change to shared code passes `experiments/probes/hill_old_vs_new.py`
(prints `IDENTICAL`) before it is committed.

## Experiments

Req: R = required, o = optional. Status for all rows starts at `planned`.

### 1. Hill ladder

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| VM25-A | von Mises 25 | Ours | R | planned | | | | |
| VM25-B | von Mises 25 | PySR alone | R | planned | | | | |
| VM25-C | von Mises 25 | Distillation | R | blocked (P2) | | | | |
| VM25-D | von Mises 25 | Sparse linear | R | blocked (P3) | | | | |
| VM25-E | von Mises 25 | PySR + library | R | blocked (P4) | | | | |
| VM25-F | von Mises 25 | Structure recovery | R | blocked (VM25-A) | | | | |
| VM25-I | von Mises 25 | Operon +- encoder | o | blocked (P6) | | | | |
| VM100-A | von Mises 100 (scale check) | Ours | R | planned | | | | |
| VM100-B | von Mises 100 (scale check) | PySR alone | o | planned | | | | |
| HILL-A | Hill aligned | Ours | R | planned | | | | |
| HILL-B | Hill aligned | PySR alone | R | planned | | | | |
| HILL-C | Hill aligned | Distillation | R | blocked (P2) | | | | |
| HILL-D | Hill aligned | Sparse linear | R | blocked (P3) | | | | |
| HILL-E | Hill aligned | PySR + library | R | blocked (P4) | | | | |
| HILL-F | Hill aligned | Structure recovery | R | blocked (HILL-A) | | | | |
| HILL-I | Hill aligned | Operon +- encoder | o | blocked (P6) | | | | |
| RH2-A | Rotated Hill 2D | Ours | R | planned | | | | |
| RH2-B | Rotated Hill 2D | PySR alone | R | planned | | | | |
| RH2-C | Rotated Hill 2D | Distillation | R | blocked (P2) | | | | |
| RH2-D | Rotated Hill 2D | Sparse linear | R | blocked (P3) | | | | |
| RH2-E | Rotated Hill 2D | PySR + library | R | blocked (P4) | | | | |
| RH2-F | Rotated Hill 2D | Structure recovery | R | blocked (RH2-A) | | | | |
| RH2-I | Rotated Hill 2D | Operon +- encoder | o | blocked (P6) | | | | |
| RH3-A | Rotated Hill 3D | Ours | R | planned | | | | |
| RH3-B | Rotated Hill 3D | PySR alone | R | planned | | | | |
| RH3-C | Rotated Hill 3D | Distillation | R | blocked (P2) | | | | |
| RH3-D | Rotated Hill 3D | Sparse linear | R | blocked (P3) | | | | |
| RH3-E | Rotated Hill 3D | PySR + library | R | blocked (P4) | | | | |
| RH3-F | Rotated Hill 3D | Structure recovery | R | blocked (RH3-A) | | | | |
| RH3-I | Rotated Hill 3D | Operon +- encoder | o | blocked (P6) | | | | |

Known: a linear fit on the 21 quadratic monomials recovers every surface in
this group exactly (RESEARCH_LOG 35), so D is expected to succeed here.

### 2. Anisotropic criteria without a single-invariant answer

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| CB-A | CB2001 | Ours | R | planned (old: n = 2, rows vs J2^0, J3^0 cosine 0.999999, 1 seed) | | | | 22 |
| CB-B | CB2001 | PySR alone | R | planned | | | | |
| CB-C | CB2001 | Distillation | R | blocked (P2) | | | | |
| CB-D | CB2001 | Sparse linear | R | blocked (P3) | | | | |
| CB-E | CB2001 | PySR + library | R | blocked (P4) | | | | |
| CB-F | CB2001 | Structure recovery (rows vs J2^0, J3^0) | R | blocked (CB-A) | | | | |
| CB-I | CB2001 | Operon +- encoder | o | blocked (P6) | | | | |
| YLD-A | Yld2004-18p | Ours | R | planned | | | | |
| YLD-B | Yld2004-18p | PySR alone | R | planned | | | | |
| YLD-C | Yld2004-18p | Distillation | R | blocked (P2) | | | | |
| YLD-D | Yld2004-18p | Sparse linear | o | blocked (P3) | | | | |
| YLD-E | Yld2004-18p | PySR + library | o | blocked (P4) | | | | |
| YLD-I | Yld2004-18p | Operon +- encoder | o | blocked (P6) | | | | |

### 3. Crystal-plasticity data (proposed, not yet agreed)

No known closed form: report accuracy and compactness, not recovery.

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| DX-A | DX56D | Ours | o | planned | | | | |
| DX-B | DX56D | PySR alone | o | planned | | | | |
| DX-C | DX56D | Distillation | o | blocked (P2) | | | | |
| DX-D | DX56D | Sparse linear | o | blocked (P3) | | | | |
| AU-A | Gold | Ours | o | planned | | | | |
| AU-B | Gold | PySR alone | o | planned | | | | |
| AU-C | Gold | Distillation | o | blocked (P2) | | | | |
| AU-D | Gold | Sparse linear | o | blocked (P3) | | | | |

### 4. Synthetic scaling

One experiment covers all d values; `instance` in `summary.csv` holds d and
the equation index (about 10 equations per d, 5 seeds).

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| SA-A | Synthetic (a) | Ours | R | blocked (P5) | | | | |
| SA-B | Synthetic (a) | PySR alone | R | blocked (P5) | | | | |
| SA-C | Synthetic (a) | Distillation | R | blocked (P2, P5) | | | | |
| SA-D | Synthetic (a) | Sparse linear | R | blocked (P3, P5) | | | | |
| SA-E | Synthetic (a) | PySR + library | R | blocked (P4, P5) | | | | |
| SA-F | Synthetic (a) | Structure recovery (A vs true) | R | blocked (SA-A) | | | | |
| SA-I | Synthetic (a) | Operon +- encoder | R | blocked (P5, P6) | | | | |
| SB-A | Synthetic (b) | Ours | R | blocked (P5) | | | | |
| SB-B | Synthetic (b) | PySR alone | R | blocked (P5) | | | | |
| SB-C | Synthetic (b) | Distillation | R | blocked (P2, P5) | | | | |
| SB-D | Synthetic (b) | Sparse linear | o | blocked (P3, P5) | | | | |
| SB-E | Synthetic (b) | PySR + library | R | blocked (P4, P5) | | | | |
| SB-F | Synthetic (b) | Structure recovery (n, exponents) | R | blocked (SB-A) | | | | |
| SB-I | Synthetic (b) | Operon +- encoder | R | blocked (P5, P6) | | | | |
| SBD-A | Synthetic (b) + distractors | Ours | o | blocked (P5) | | | | |
| SBD-G | Synthetic (b) + distractors | PySR select_k_features | o | blocked (P5) | | | | |

### 5. Equation benchmarks

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| FEY-A | Feynman | Ours | R | planned (old: seed 42, 85/110) | | | | 27 |
| FEY-B | Feynman | PySR alone | R | planned (old: seed 42, 86/110) | | | | 27 |
| FEY-C | Feynman | Distillation | R | blocked (P2) | | | | |
| FEY-D | Feynman | Sparse linear | o | blocked (P3) | | | | |
| FEY-E | Feynman | PySR + library | o | blocked (P4) | | | | |
| FEY-F | Feynman | Structure recovery (n vs true groups) | R | blocked (FEY-A) (old: 93/102, 8/8) | | | | 27 |
| FEY-G | Feynman | PySR select_k_features | o | planned | | | | |
| FEY-H | Feynman | AI Feynman (cited, not run) | o | planned | | | | |
| FEY-I | Feynman | Operon +- encoder | o | blocked (P6) | | | | |
| LLM-A | LLM-SRBench | Ours | R | planned (old: 61.2 +- 2.9 of 91) | | | | 32 |
| LLM-B | LLM-SRBench | PySR alone | R | planned (old: 57.2 +- 1.3 of 91) | | | | 32 |
| LLM-C | LLM-SRBench | Distillation | o | blocked (P2) | | | | |
| LLM-F | LLM-SRBench | Structure recovery | o | blocked (LLM-A) | | | | |
| LLM-I | LLM-SRBench | Operon +- encoder | o | blocked (P6) | | | | |

### 6. Noise robustness

| ID | Dataset | Method | Req | Status | Seeds | Commit | Result | Log |
|---|---|---|---|---|---|---|---|---|
| NRH3-A | Rotated Hill 3D, noisy | Ours | R | planned | | | | |
| NRH3-B | Rotated Hill 3D, noisy | PySR alone | R | planned | | | | |
| NRH3-C | Rotated Hill 3D, noisy | Distillation | o | blocked (P2) | | | | |
| NRH3-F | Rotated Hill 3D, noisy | Structure recovery | o | blocked (NRH3-A) | | | | |
| NFEY-A | Feynman subset, noisy | Ours | R | planned | | | | |
| NFEY-B | Feynman subset, noisy | PySR alone | R | planned | | | | |
