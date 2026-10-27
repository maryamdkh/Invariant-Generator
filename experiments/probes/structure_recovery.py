"""Structure recovery (F'): learned fourth-order tensor A vs the true M.

    uv run python experiments/probes/structure_recovery.py \
        --config configs/hill_ladder/hill.toml \
        --checkpoint results/yield_benchmark/hill_ladder/seed42/hill/hill_full/stage2_sparse/checkpoint_best.pt \
        --truth data/hill_ladder_full/hill_full.json

Prints the Frobenius cosine between the learned A (Mandel 6x6) and the true
M_mandel, both sets of eigenvalues scaled by their largest, and the relative
L2 of the best c * sqrt(sigma:A:sigma) against sqrt(sigma:M:sigma) on 5000
random directions (the error floor any equation in I11 alone can reach).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from invariant_generator.config import load_config
from invariant_generator.constraints import fourth_order_to_mandel_matrix
from invariant_generator.sparsify import load_model_for_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.train.device = "cpu"
    model = load_model_for_checkpoint(config, args.checkpoint)
    tensor = model.invariant_pool.effective_fourth_order_tensor().detach().double()
    learned = fourth_order_to_mandel_matrix(tensor).numpy()
    truth = np.asarray(json.loads(args.truth.read_text(encoding="utf-8"))["M_mandel"], dtype=float)

    cosine = float(np.sum(learned * truth) / (np.linalg.norm(learned) * np.linalg.norm(truth)))
    eig_learned = np.linalg.eigvalsh(learned)
    eig_truth = np.linalg.eigvalsh(truth)

    rng = np.random.default_rng(2026)
    directions = rng.normal(size=(5000, 6))
    q_learned = np.sqrt(np.clip(np.einsum("ni,ij,nj->n", directions, learned, directions), 0, None))
    q_truth = np.sqrt(np.einsum("ni,ij,nj->n", directions, truth, directions))
    scale = float(q_learned @ q_truth / (q_learned @ q_learned))
    floor = float(np.linalg.norm(scale * q_learned - q_truth) / np.linalg.norm(q_truth))

    result = {
        "checkpoint": str(args.checkpoint),
        "truth": str(args.truth),
        "cosine": cosine,
        "eigenvalues_learned_over_max": (eig_learned / eig_learned.max()).tolist(),
        "eigenvalues_true_over_max": (eig_truth / eig_truth.max()).tolist(),
        "best_scale": scale,
        "relative_l2_floor": floor,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
