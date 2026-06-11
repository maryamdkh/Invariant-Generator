from __future__ import annotations

import argparse
from pathlib import Path

from invariant_generator.config import load_config
from invariant_generator.symbolic import train_symbolic_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PySR symbolic model on selected invariant features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to the TOML config used for the trained model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to results/<run_id>/checkpoint_best.pt.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None
    result = train_symbolic_from_config(
        config,
        checkpoint_path=checkpoint_path,
        config_path=args.config,
    )

    print(f"[INFO] Output dir:          {result.output_dir}")
    print(f"[INFO] Selected invariants: {', '.join(result.selected_invariants)}")
    print(f"[INFO] Best equation:       {result.best_equation}")
    print(f"[INFO] Metrics:             {result.metrics_path}")
    print(f"[INFO] Equations:           {result.equations_path}")


if __name__ == "__main__":
    main()
