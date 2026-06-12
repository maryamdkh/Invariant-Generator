from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from invariant_generator.config import load_config
from invariant_generator.data import prepare_training_data
from invariant_generator.diagnostics import (
    encoder_score_diagnostics,
    invariant_feature_statistics,
)
from invariant_generator.model import InvariantYieldModel
from invariant_generator.symbolic import select_top_invariants_from_encoder
from invariant_generator.train import train_from_config
from invariant_generator.utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune invariants by encoder score and refit from scratch."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--feature-selection", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.train.device)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else config.experiment_dir / "checkpoint_best.pt"
    )
    top_k = args.top_k if args.top_k is not None else config.symbolic.top_k
    feature_selection = args.feature_selection or config.symbolic.feature_selection

    data = prepare_training_data(config)
    model = InvariantYieldModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    S = model.encoder_matrix()
    if S is None:
        raise ValueError("Prune-and-refit requires a trained encoder matrix S.")

    encoder_input_stats = invariant_feature_statistics(
        model,
        data.X_train,
        config.invariants.selected,
        device=device,
        normalized=True,
    )
    selection = select_top_invariants_from_encoder(
        S,
        config.invariants.selected,
        top_k=top_k,
        feature_selection=feature_selection,
        feature_stds=np.asarray(encoder_input_stats["std"], dtype=np.float64),
    )

    refit_config = deepcopy(config)
    refit_config.invariants.selected = selection.names
    refit_config.train.run_id = args.run_id or f"{config.train.run_id}_pruned_top{top_k}"

    result = train_from_config(refit_config)
    summary_path = save_json(
        result.experiment_dir / "prune_refit.json",
        {
            "source_checkpoint": str(checkpoint_path),
            "feature_selection": selection.feature_selection,
            "selected_invariants": selection.names,
            "selected_indices": selection.indices,
            "selected_scores": selection.scores,
            "selected_raw_scores": selection.raw_scores,
            "selected_scaled_scores": selection.scaled_scores,
            "encoder_score_diagnostics": encoder_score_diagnostics(
                model,
                config.invariants.selected,
                feature_std=encoder_input_stats["std"],
            ),
            "refit_run_id": refit_config.train.run_id,
            "best_epoch": result.best_epoch,
            "best_test_mse": result.best_test_mse,
        },
    )

    print(f"[INFO] Selected invariants: {', '.join(selection.names)}")
    print(f"[INFO] Refit results:       {result.experiment_dir}")
    print(f"[INFO] Summary:             {summary_path}")


if __name__ == "__main__":
    main()
