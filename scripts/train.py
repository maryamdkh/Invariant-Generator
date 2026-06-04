from __future__ import annotations

import argparse

from invariant_generator.config import load_config
from invariant_generator.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the invariant-generator model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to a TOML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_from_config(config)

    print()
    print(f"[INFO] Best epoch:       {result.best_epoch}")
    print(f"[INFO] Best test MSE:    {result.best_test_mse:.8g}")
    print(f"[INFO] Best checkpoint:  {result.best_checkpoint}")
    print(f"[INFO] Final checkpoint: {result.final_checkpoint}")
    print(f"[INFO] History:          {result.history_path}")


if __name__ == "__main__":
    main()
