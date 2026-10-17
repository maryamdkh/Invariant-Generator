# Handoff (2026-10-17)

Read this first in a new session, then `RESEARCH_LOG.md` (entries 1-27, the
full record with numbers) and `PAPER_PLAN.md`.

## Goal

PhD project. The adaptive encoder (Stage 1 smallest bottleneck n, Stage 2
interpretable encoder rows, Stage 3 PySR on the encoded variables) was built
for anisotropic yield surfaces (rotated Hill). We generalize it into a
method-first paper for a CS/ML venue: an adaptive interpretable bottleneck
that finds a few closed-form variables and hands them to symbolic regression
(not competing with PySR; "setting the scene" for it). Yield surfaces are an
application section. Hard rule: the yield-surface core must keep working
unchanged; everything new is opt-in config, verified with
`experiments/probes/hill_regression.py` (old vs new code identical).

## Code state

Branch `feynman-generalization` on GitHub (maryamdkh/Invariant-Generator),
latest commits `6030fff` (v3) and later log updates. Key pieces:
- Feature maps: `input_transform.kind` = stress_invariants (Hill) | pairwise |
  log (power-law groups) | orthotropic_polynomial (29 degree-2/3 invariants).
- Stage 1 (`adaptive.py`): nrmse metric, restarts, encoder LR scale,
  `acceptance = "relative_to_reference"` (full-width reference model +
  gradient spectrum), log-y targets (`data.target_transform = "log"`).
- Stage 2 (`sparsify.py`): `method = "snap"` (exact rational exponents,
  rref basis, `snap_acceptance = "relative_to_dense"`) or `"graded"`
  (homogeneous rows for polynomial maps).
- Stage 3 (`adaptive_symbolic.py`): monomial inputs, expansion to original
  inputs, recovery = relative L2 < 1e-6 on fresh/official-test points.
- Benchmarks: Feynman (`scripts/prepare_feynman_benchmark.py`, PMLB
  formulas), LLM-SRBench LSR-Transform (`scripts/prepare_llm_srbench.py`,
  data gated, downloaded to `data/llm_srbench/`), yield criteria
  (`yield_criteria.py`: CB2001, Yld2004-18p, validated vs the papers).
- Runner: `scripts/run_feynman_benchmark.py` (arms pipeline, portfolio,
  baseline = PySR alone); merge `scripts/merge_feynman_benchmark.py`.
- Frozen config: `configs/feynman_benchmark_v3.toml`.

## Cluster (Arrhenius), author submits (two-factor login)

Repo at `/nobackup/proj/disk/naiss2026-4-599/personal/maryam/projects/symbolic_regression/Invariant-Generator`
(no spaces in the path). Account `naiss2026-4-599-gpu`, partition `gpu`,
`--gpus=1` (jobs use CPUs; GPU idle). Login node x86_64, GPU nodes aarch64:
per-architecture `.venv-<arch>`, `.julia_depot-<arch>`; use
`.venv-x86_64` on the login node. `cluster/setup_env.sh` (run on login and
via `srun ... --pty bash cluster/setup_env.sh` on a GPU node),
`cluster/make_tasks.py`, `cluster/feynman_array.sbatch` (env vars CONFIG,
RUN, TASKS, MANIFEST, DATA_DIR; 16 cores, 64G, 1.5 h). Julia needs LD_PRELOAD
of its bundled libstdc++ (handled in scripts). Cost ~0.28 GPU-h per task.
Results are copied to the Mac with rsync (run on the Mac).

## Results so far

- Feynman full run, v3, seed 42 (110 equations): recovered pipeline 85,
  portfolio 88, PySR alone 86; no difference beyond noise (PySR alone varies
  +-2 between identical runs). PySR alone solves most in ~3 s; Stage 1+2
  overhead ~3 min. Stage 1 finds the true number of groups on 93/102
  reducible equations and correctly no reduction on all 8 others.
- CB2001 yield (2090-T3): Stage 1 n = 2, learned rows span the true J2^0 and
  J3^0 (cosine 0.999999); graded Stage 2 + PySR found the right nested root
  structure, error 2e-4 (one spurious constant).
- LLM-SRBench: integrated and smoke-tested (II.36.38_1_0 recovered, PySR 1.2 s
  on groups vs 33 s raw). 91/111 problems have positive inputs.

## Next steps (agreed)

1. LLM-SRBench seed 42 on the cluster (~30 GPU-h): see `cluster/README.md`,
   section LLM-SRBench (needs `data/llm_srbench/` copied to the cluster,
   `RUN=llmsr_v3 CONFIG=configs/feynman_benchmark_v3.toml MANIFEST=data/llm_srbench_converted/manifest.json DATA_DIR=data/llm_srbench_converted`).
2. Feynman with noise, seed 42: implemented (`data.target_noise`,
   RESEARCH_LOG entry 28); submit with `TARGET_NOISE=0.01 RUN=v3_noise1pct`
   (cluster/README.md, section "Feynman with target noise"). No early stop
   with noise, so ~2-3x the noiseless cost. Still to write: post-hoc noisy
   scoring (fresh R^2, 3-digit symbolic check) in the merge step.
3. Scaling experiment (new code): synthetic equations with an increasing
   number of inputs (e.g. 4 to 12) that depend on a few power-law groups;
   expect PySR alone to degrade with d while the pipeline does not. Likely
   the key figure for a CS venue.
4. Then multi-seed runs where an effect shows; optional: joint constant
   refit after Stage 3 (CB2001), Stage 1 overhead reduction.
