from __future__ import annotations

import argparse
from pathlib import Path

from invariant_generator.benchmark import (
    benchmark_formula_names,
    evaluate_benchmark_dataset_quality,
)
from invariant_generator.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated benchmark formula and dataset quality."
    )
    parser.add_argument("--config", default="configs/self_eval_benchmark.toml")
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional formula name from [benchmark].formula_names, e.g. "
            "'single_H2'. If omitted, evaluate all configured formulas."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    names = [args.case] if args.case is not None else benchmark_formula_names(config)
    for name in names:
        quality = evaluate_benchmark_dataset_quality(config, name)
        output_path = Path(str(quality["metadata_path"])).with_name(
            "formula_dataset_quality.json"
        )
        print(f"[INFO] {name}: quality={output_path}")
        print(
            "[INFO] "
            f"{name}: homogeneity_l2="
            f"{quality['formula_homogeneity']['relative_l2_error']:.6g}, "
            f"surface_max_abs="
            f"{quality['dataset_surface']['surface_value_max_abs_error']:.6g}"
        )


if __name__ == "__main__":
    main()
