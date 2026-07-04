from __future__ import annotations

import argparse
from pathlib import Path

from invariant_generator.config import load_config
from invariant_generator.report import create_adaptive_analysis_notebook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a per-run adaptive pipeline analysis notebook."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config used for the adaptive pipeline run.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Optional adaptive run directory. Defaults to "
            "[train].results_dir/[adaptive].results_subdir from the config."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional notebook path. Defaults to <results-dir>/analysis.ipynb.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    notebook_path = create_adaptive_analysis_notebook(
        config,
        config_path=Path(args.config),
        results_dir=None if args.results_dir is None else Path(args.results_dir),
        output_path=None if args.output is None else Path(args.output),
    )
    print(f"[INFO] Analysis notebook: {notebook_path}")


if __name__ == "__main__":
    main()
