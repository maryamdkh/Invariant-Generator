# Experiments

Probe scripts and saved outputs referenced in `RESEARCH_LOG.md`.

- `probes/`: one-off experiments. Most take a scratch directory as their
  first argument (where they write models) and are run from the repository
  root with `.venv/bin/python experiments/probes/<script>.py <scratch_dir> ...`.
- `outputs/`: their printed results and logs (progress bars stripped).

Pipeline results live under `results/` (git-ignored, regenerable from the
configs in `configs/`).
