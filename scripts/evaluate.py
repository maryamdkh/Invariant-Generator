from __future__ import annotations

import argparse
from pathlib import Path

import torch

from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.diagnostics import (
    constraint_diagnostics,
    encoder_score_diagnostics,
    invariant_feature_statistics,
)
from invariant_generator.evaluation import evaluate_model, predict_numpy
from invariant_generator.model import InvariantYieldModel
from invariant_generator.utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an invariant-generator checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to the TOML config used to build the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to results/<run_id>/checkpoint_best.pt.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.train.device)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else config.experiment_dir / "checkpoint_best.pt"
    )

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(
        model,
        data.X_test,
        data.y_test,
        device=device,
        batch_size=8192,
    )
    predictions = predict_numpy(
        model,
        data.X_test,
        device=device,
        batch_size=8192,
    )
    invariant_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        batch_size=8192,
    )
    encoder_input_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        batch_size=8192,
        normalized=True,
    )

    output_path = save_json(
        config.experiment_dir / "evaluation.json",
        {
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "constraint_diagnostics": constraint_diagnostics(model, config.constraints),
            "invariant_feature_statistics": invariant_stats,
            "encoder_input_feature_statistics": encoder_input_stats,
            "encoder_score_diagnostics": encoder_score_diagnostics(
                model,
                config.invariants.selected,
                feature_std=encoder_input_stats["std"],
            ),
            "invariant_normalization": model.invariant_normalization_state(),
        },
    )
    predictions_path = config.experiment_dir / "predictions_test.pt"
    torch.save(
        {
            "prediction": torch.as_tensor(predictions),
            "target": torch.as_tensor(data.y_test),
            "stress": torch.as_tensor(data.X_test),
        },
        predictions_path,
    )

    print(f"[INFO] Checkpoint:  {checkpoint_path}")
    print(f"[INFO] Metrics:     {metrics}")
    print(f"[INFO] Saved JSON:  {output_path}")
    print(f"[INFO] Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
