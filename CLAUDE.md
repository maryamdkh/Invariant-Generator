# Invariant-Generator: rules for Claude

PhD project (author: Maryam). Read `HANDOFF.md` first, then the latest
entries of `RESEARCH_LOG.md`, before changing anything.

## What the project is

An adaptive interpretable encoder in front of symbolic regression:
Stage 1 finds the smallest bottleneck n, Stage 2 makes the encoder rows
interpretable (exact exponents / invariants), Stage 3 runs PySR on the few
encoded variables. It was built for anisotropic yield surfaces, and on
**rotated Hill it beats PySR alone: that is the core result.** The Feynman,
LLM-SRBench, noise and scaling work exists only to show the pipeline
generalizes. We are a front end for PySR, not a competitor.

## Non-negotiable: rotated Hill must keep working unchanged

- Every new behaviour is opt-in config whose default reproduces the old
  behaviour. Never change a default that the Hill configs rely on.
- After any change to shared code (`config.py`, `data.py`, `adaptive.py`,
  `sparsify.py`, `symbolic.py`, `adaptive_symbolic.py`, `model.py`,
  `train.py`, input features), run before committing:
  `uv run python experiments/probes/hill_old_vs_new.py <last-good-commit>`
  It must print `IDENTICAL`. Passing tests alone is not enough.
- If Hill output changes, stop and tell the author; do not commit.

## Workflow

- Record every step (goal, what was done, numbers, decision, paths) as a
  numbered entry in `RESEARCH_LOG.md`; keep `HANDOFF.md` current. Probe
  scripts go in `experiments/probes/`, their outputs in
  `experiments/outputs/`, never only in a scratch folder.
- Work on branch `feynman-generalization`; commit and push so the cluster
  can `git pull`.
- Cluster (Arrhenius, SLURM, two-factor login): the author runs all cluster
  commands; Claude prepares scripts in `cluster/` and gives exact
  copy-paste steps. Repo there:
  `/nobackup/proj/disk/naiss2026-4-599/personal/maryam/projects/symbolic_regression/Invariant-Generator`.
  Submit with `-A naiss2026-4-599-gpu -p gpu --gpus=1`; login node uses
  `.venv-x86_64`. Code reaches the cluster by `git pull`; gitignored data
  (e.g. `data/llm_srbench/`) and results move with rsync from the Mac using
  the SSH alias `arrhenius`. See `cluster/README.md`.
- Single-seed differences of +-2 equations are PySR noise; claim an effect
  only after multiple seeds.
- Paper: CS/ML venue, method-first; yield surfaces are an application
  section. Prefer recent citations.
