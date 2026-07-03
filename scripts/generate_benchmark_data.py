from __future__ import annotations

import argparse

from invariant_generator.benchmark import (
    benchmark_formula_names,
    generate_benchmark_dataset,
)
from invariant_generator.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark stress-surface datasets."
    )
    parser.add_argument("--config", default="configs/self_eval_benchmark.toml")
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional formula name from [benchmark].formula_names, e.g. "
            "'single_H2'. If omitted, generate all configured formulas."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    names = [args.case] if args.case is not None else benchmark_formula_names(config)
    for name in names:
        dataset = generate_benchmark_dataset(config, name)
        print(f"[INFO] {name}: dataset={dataset.dataset_path}")
        print(f"[INFO] {name}: pipeline config={dataset.pipeline_config_path}")
        print(f"[INFO] {name}: metadata={dataset.metadata_path}")
        print(f"[INFO] {name}: quality={dataset.quality_path}")


if __name__ == "__main__":
    main()
