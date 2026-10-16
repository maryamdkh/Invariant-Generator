# Running the Feynman benchmark on a SLURM cluster (Vera, Arrhenius)

One array task = one (equation, seed); it runs all arms (pipeline,
portfolio, PySR alone) with the same PySR budget and writes its own results
folder, so tasks never share files.

## 1. Copy the repository (from your laptop)

```bash
rsync -av --exclude .venv --exclude .uv-cache --exclude results --exclude data/feynman_tables/pmlb_data ./ <user>@<cluster>:<project-dir>/Invariant-Generator/
```

## 2. One-time setup on the login node

```bash
cd <project-dir>/Invariant-Generator
bash cluster/setup_env.sh
```

This installs the Python environment with uv, downloads Julia and
precompiles PySR inside the project (`.julia_depot/`), and regenerates the
noiseless Feynman data for all 119 equations (`data/feynman/`).

## 3. Make the task list and submit

```bash
.venv/bin/python cluster/make_tasks.py --positive-only --seeds 42,43,44,45,46
mkdir -p cluster/logs
sbatch -A <project-account> --array=1-$(wc -l < cluster/tasks.txt)%40 cluster/feynman_array.sbatch
```

`%40` caps concurrent tasks; adjust to your allocation. Each task asks for
16 cores and at most 1.5 h (Stage 1-2 plus up to 15 min PySR per arm, with
early stopping when the formula is found). Override with environment
variables, e.g. `RUN=v2 CONFIG=configs/feynman_benchmark_v2.toml
TIMEOUT_MINUTES=15 sbatch ...`.

Budget: 110 equations x 5 seeds = 550 tasks; upper bound about
550 x 1.5 h x 16 cores ~ 13,000 core-hours, expected well below that because
easy equations stop early. Start with one seed (110 tasks) to measure.

## 4. Collect and merge

```bash
rsync -av <user>@<cluster>:<project-dir>/Invariant-Generator/results/feynman_benchmark/v2/ results/feynman_benchmark/v2/
.venv/bin/python scripts/merge_feynman_benchmark.py results/feynman_benchmark/v2
```

Partition names, module systems and per-cluster limits differ between Vera
and Arrhenius; add `#SBATCH -p <partition>` if your project requires one.

## LLM-SRBench (LSR-Transform)

Copy the downloaded dataset (`data/llm_srbench/`, gated: access must be
approved on Hugging Face) to the cluster, then on the login node:

```bash
.venv/bin/python scripts/prepare_llm_srbench.py
.venv/bin/python cluster/make_tasks.py --manifest data/llm_srbench_converted/manifest.json --seeds 42,43,44,45,46 --output cluster/tasks_llmsr.txt
RUN=llmsr_v2 MANIFEST=data/llm_srbench_converted/manifest.json DATA_DIR=data/llm_srbench_converted TASKS=cluster/tasks_llmsr.txt \
  sbatch -A <project-account> --array=1-$(wc -l < cluster/tasks_llmsr.txt)%40 cluster/feynman_array.sbatch
```

All 111 problems are included; the 20 with non-positive inputs run the
pipeline arm as PySR alone (reported as `skipped`).
