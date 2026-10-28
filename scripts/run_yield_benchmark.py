"""Run yield-surface methods on one dataset (see invariant_generator.yield_benchmark).

    uv run python scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
        --results-dir results/yield_benchmark/dev/seed42/rh3 --arms pipeline,baseline \
        --seed 42 --timeout-minutes 15

Results go to <results-dir>/benchmark_summary.json (resumable).
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--arms", default="pipeline,baseline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the training seed and the PySR random_state (data split fixed).")
    parser.add_argument("--timeout-minutes", type=float, default=None,
                        help="PySR time limit per arm (same for every arm).")
    parser.add_argument("--numprocs", type=int, default=None)
    parser.add_argument("--device", default=None, help="Override [train].device, e.g. cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Override [train].epochs (smoke tests).")
    parser.add_argument("--sparse-max-power", type=int, default=3,
                        help="sparse_linear (D): fit y^m for m = 1..this, library degree max(3, m).")
    args = parser.parse_args()

    import_module("pysr")  # Julia before Torch
    from invariant_generator.config import PROJECT_ROOT, load_config
    from invariant_generator.yield_benchmark import run_yield_benchmark

    config = load_config(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
        config.symbolic.random_state = args.seed
    if args.timeout_minutes is not None:
        config.symbolic.timeout_hours = args.timeout_minutes / 60.0
    if args.numprocs is not None:
        config.symbolic.numprocs = args.numprocs
    if args.device is not None:
        config.train.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    summary = run_yield_benchmark(config, results_dir=results_dir, arms=tuple(args.arms.split(",")),
                                  sparse_max_power=args.sparse_max_power)
    print(f"[YIELD] summary: {summary}")


if __name__ == "__main__":
    main()
