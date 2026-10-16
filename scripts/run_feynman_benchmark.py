"""Run the Feynman benchmark: adaptive pipeline vs PySR on the raw inputs.

Prepare data first with scripts/prepare_feynman_benchmark.py. Results are
written to <results-dir>/benchmark_summary.json after every equation and arm,
so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feynman_benchmark.toml")
    parser.add_argument("--manifest", type=Path, default=Path("data/feynman/manifest.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/feynman_benchmark/pilot"))
    parser.add_argument("--equations", default="", help="Comma-separated subset of equation ids.")
    parser.add_argument("--arms", default="pipeline,baseline")
    parser.add_argument("--timeout-minutes", type=float, default=None,
                        help="Override [symbolic].timeout_hours for every arm.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the training seed and the PySR random_state.")
    parser.add_argument("--portfolio-raw-share", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override [data].data_dir, e.g. data/llm_srbench_converted.")
    parser.add_argument("--numprocs", type=int, default=None,
                        help="Override [symbolic].numprocs (PySR threads), e.g. the SLURM core count.")
    args = parser.parse_args()

    # PySR's Julia runtime must initialize before Torch is imported.
    import_module("pysr")
    from invariant_generator.config import PROJECT_ROOT, load_config
    from invariant_generator.feynman_benchmark import run_benchmark

    base = load_config(args.config)
    if args.timeout_minutes is not None:
        base.symbolic.timeout_hours = args.timeout_minutes / 60.0
    if args.numprocs is not None:
        base.symbolic.numprocs = args.numprocs
    if args.data_dir is not None:
        base.data.data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    if args.seed is not None:
        # The data split stays fixed ([data].random_state); only training
        # initialization and the PySR search change between seeds.
        base.train.seed = args.seed
        base.symbolic.random_state = args.seed
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.equations:
        wanted = args.equations.split(",")
        entries = [entry for entry in entries if entry["equation_id"] in wanted]
    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    summary = run_benchmark(
        base, entries, results_dir=results_dir, arms=tuple(args.arms.split(",")),
        portfolio_raw_share=args.portfolio_raw_share,
    )
    print(f"[BENCH] summary: {summary}")


if __name__ == "__main__":
    main()
