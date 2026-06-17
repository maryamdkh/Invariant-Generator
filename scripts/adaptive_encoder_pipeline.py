from __future__ import annotations

import argparse
from pathlib import Path

from invariant_generator.adaptive import config_for_adaptive_n, run_adaptive_sweep
from invariant_generator.adaptive_symbolic import train_encoded_symbolic_from_config
from invariant_generator.config import load_config
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the adaptive encoder invariant-discovery pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/adaptive_encoder_rotated_hill.toml",
        help="Path to the adaptive TOML config.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint for stage2 or stage3. Stage2 expects a stage1 checkpoint; stage3 expects a sparse checkpoint.",
    )
    parser.add_argument(
        "--selected-n",
        type=int,
        default=None,
        help="Encoder output dimension for stage2/stage3 when not running stage1 in this command.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    selected_n = args.selected_n
    selected_checkpoint = Path(args.checkpoint) if args.checkpoint is not None else None

    if args.stage in {"all", "stage1"}:
        sweep = run_adaptive_sweep(config)
        print(f"[INFO] Stage 1 summary: {sweep.summary_path}")
        if sweep.plot_path is not None:
            print(f"[INFO] Stage 1 plot:    {sweep.plot_path}")
        print(f"[INFO] Selected n:      {sweep.selected_n}")
        print(f"[INFO] Selected ckpt:   {sweep.selected_checkpoint}")
        selected_n = sweep.selected_n
        selected_checkpoint = sweep.selected_checkpoint
        if args.stage == "stage1":
            return

    if selected_n is None:
        raise ValueError("--selected-n is required when stage1 is not run in this command.")
    if selected_checkpoint is None:
        raise ValueError("--checkpoint is required when stage1 is not run in this command.")

    stage_config = config_for_adaptive_n(config, selected_n)
    sparse_checkpoint = selected_checkpoint

    if args.stage in {"all", "stage2"}:
        sparse = sparsify_encoder_from_checkpoint(
            stage_config,
            checkpoint_path=selected_checkpoint,
        )
        print(f"[INFO] Stage 2 summary: {sparse.summary_path}")
        print(f"[INFO] Stage 2 mask:    {sparse.mask_path}")
        print(f"[INFO] Stage 2 ckpt:    {sparse.checkpoint_path}")
        sparse_checkpoint = sparse.checkpoint_path
        if args.stage == "stage2":
            return

    if args.stage in {"all", "stage3"}:
        symbolic_config = config_for_adaptive_n(config, selected_n)
        symbolic_config.train.run_id = symbolic_config.sparsification.run_id
        result = train_encoded_symbolic_from_config(
            symbolic_config,
            checkpoint_path=sparse_checkpoint,
            config_path=args.config,
        )
        print(f"[INFO] Stage 3 output:   {result.output_dir}")
        print(f"[INFO] Formulas:         {result.formulas_path}")
        print(f"[INFO] Best equation:    {result.best_equation}")
        print(f"[INFO] Metrics:          {result.metrics_path}")


if __name__ == "__main__":
    main()
