from __future__ import annotations

import argparse

from invariant_generator.benchmark import run_benchmark_suite
from invariant_generator.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-evaluating synthetic invariant benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/self_eval_benchmark.toml",
        help="Path to the benchmark TOML config.",
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional formula name from [benchmark].formula_names, e.g. "
            "'single_H2'. If omitted, run the full configured suite."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "quality", "pipeline", "evaluate", "all"],
        default="all",
        help="Benchmark stage to run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_benchmark_suite(
        config,
        stage=args.stage,
        case=args.case,
        config_path=args.config,
    )
    print(f"[INFO] Benchmark root:    {result['root']}")
    print(f"[INFO] Benchmark summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
