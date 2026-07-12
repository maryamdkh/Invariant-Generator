from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from invariant_generator.adaptive import (
    adaptive_results_dir,
    adaptive_sparsification_run_id,
    config_for_adaptive_n,
    run_adaptive_sweep,
)
from invariant_generator.adaptive_symbolic import train_encoded_symbolic_from_config
from invariant_generator.config import Config, load_config
from invariant_generator.report import create_adaptive_analysis_notebook
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint
from invariant_generator.utils import save_json


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return [float(item) for item in values]


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    return [int(item) for item in values]


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _variant_config(
    base_config: Config,
    *,
    base_results_subdir: str,
    scale: float,
    probability: float,
    seed: int,
) -> Config:
    config = deepcopy(base_config)
    config.noise.enabled = False
    config.train.seed = int(seed)
    config.train_input_noise.enabled = float(scale) > 0.0 and float(probability) > 0.0
    config.train_input_noise.scale = float(scale)
    config.train_input_noise.probability = float(probability)
    config.train_input_noise.random_state = int(seed)
    config.adaptive.results_subdir = str(
        Path(base_results_subdir)
        / "noise_sweep"
        / f"scale_{_float_tag(scale)}"
        / f"prob_{_float_tag(probability)}"
        / f"seed_{seed}"
    )
    return config


def _row_from_outputs(
    config: Config,
    *,
    scale: float,
    probability: float,
    seed: int,
    selected_n: int | None,
    selected_checkpoint: Path | None,
    stage1_summary: Path | None,
    stage2_summary: Path | None,
    symbolic_metrics: Path | None,
    best_equation_path: Path | None,
) -> dict[str, object]:
    stage1 = _load_json(stage1_summary)
    stage2 = _load_json(stage2_summary)
    symbolic = _load_json(symbolic_metrics)
    selected_run = None
    if stage1 is not None:
        for run in stage1.get("runs", []):
            if run.get("selected"):
                selected_run = run
                break

    return {
        "scale": float(scale),
        "probability": float(probability),
        "seed": int(seed),
        "results_dir": str(adaptive_results_dir(config)),
        "train_input_noise": {
            "enabled": config.train_input_noise.enabled,
            "scale": config.train_input_noise.scale,
            "probability": config.train_input_noise.probability,
            "random_state": config.train_input_noise.random_state,
            "relative_to_feature_std": config.train_input_noise.relative_to_feature_std,
        },
        "selected_n": selected_n,
        "selected_checkpoint": None
        if selected_checkpoint is None
        else str(selected_checkpoint),
        "stage1_summary": None if stage1_summary is None else str(stage1_summary),
        "stage1_selected_train_metrics": None
        if selected_run is None
        else selected_run.get("train_metrics"),
        "stage1_selected_test_metrics": None
        if selected_run is None
        else selected_run.get("test_metrics"),
        "stage2_summary": None if stage2_summary is None else str(stage2_summary),
        "stage2_train_metrics": None if stage2 is None else stage2.get("train_metrics"),
        "stage2_test_metrics": None if stage2 is None else stage2.get("test_metrics"),
        "stage2_formulas": None if stage2 is None else stage2.get("formulas"),
        "symbolic_metrics": None
        if symbolic_metrics is None
        else str(symbolic_metrics),
        "symbolic_test_metrics": None if symbolic is None else symbolic.get("test"),
        "best_equation_path": None
        if best_equation_path is None
        else str(best_equation_path),
        "best_equation": None
        if best_equation_path is None or not best_equation_path.exists()
        else best_equation_path.read_text(encoding="utf-8").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep optional train-time input noise through the adaptive pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/adaptive_encoder_rotated_hill.toml",
        help="Base adaptive TOML config.",
    )
    parser.add_argument(
        "--scales",
        default="0,0.002,0.005,0.01,0.02,0.05",
        help="Comma-separated train_input_noise.scale values.",
    )
    parser.add_argument(
        "--probabilities",
        default="1.0",
        help="Comma-separated train_input_noise.probability values.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds used for training and train-time noise.",
    )
    parser.add_argument(
        "--through-stage",
        choices=["stage1", "stage2", "stage3", "all"],
        default="stage2",
        help="Run stages cumulatively through this stage.",
    )
    parser.add_argument(
        "--pysr-scales",
        default=None,
        help="Optional comma-separated scales allowed to run Stage 3.",
    )
    parser.add_argument(
        "--summary-name",
        default="noise_sweep_summary.json",
        help="Summary JSON name written under the base sweep directory.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_results_subdir = base_config.adaptive.results_subdir
    scales = _parse_float_list(args.scales)
    probabilities = _parse_float_list(args.probabilities)
    seeds = _parse_int_list(args.seeds)
    pysr_scales = None
    if args.pysr_scales is not None:
        pysr_scales = set(_parse_float_list(args.pysr_scales))

    rows: list[dict[str, object]] = []
    run_stage2 = args.through_stage in {"stage2", "stage3", "all"}
    run_stage3 = args.through_stage in {"stage3", "all"}

    for scale in scales:
        for probability in probabilities:
            for seed in seeds:
                config = _variant_config(
                    base_config,
                    base_results_subdir=base_results_subdir,
                    scale=scale,
                    probability=probability,
                    seed=seed,
                )
                print(
                    "[INFO] Sweep case: "
                    f"scale={scale:g}, probability={probability:g}, seed={seed}"
                )

                sweep = run_adaptive_sweep(config)
                selected_n = sweep.selected_n
                selected_checkpoint = sweep.selected_checkpoint
                stage2_summary = None
                symbolic_metrics = None
                best_equation_path = None

                if run_stage2:
                    if selected_n is None or selected_checkpoint is None:
                        print("[WARN] Stage 1 did not select n; skipping Stage 2/3.")
                    else:
                        stage_config = config_for_adaptive_n(config, selected_n)
                        sparse = sparsify_encoder_from_checkpoint(
                            stage_config,
                            checkpoint_path=selected_checkpoint,
                        )
                        stage2_summary = sparse.summary_path
                        selected_checkpoint = sparse.checkpoint_path

                        should_run_pysr = run_stage3 and (
                            pysr_scales is None or float(scale) in pysr_scales
                        )
                        if should_run_pysr:
                            symbolic_config = config_for_adaptive_n(config, selected_n)
                            symbolic_config.train.run_id = adaptive_sparsification_run_id(
                                symbolic_config
                            )
                            result = train_encoded_symbolic_from_config(
                                symbolic_config,
                                checkpoint_path=sparse.checkpoint_path,
                                config_path=args.config,
                            )
                            symbolic_metrics = result.metrics_path
                            best_equation_path = result.best_equation_path
                            create_adaptive_analysis_notebook(
                                symbolic_config,
                                config_path=args.config,
                            )

                rows.append(
                    _row_from_outputs(
                        config,
                        scale=scale,
                        probability=probability,
                        seed=seed,
                        selected_n=selected_n,
                        selected_checkpoint=selected_checkpoint,
                        stage1_summary=sweep.summary_path,
                        stage2_summary=stage2_summary,
                        symbolic_metrics=symbolic_metrics,
                        best_equation_path=best_equation_path,
                    )
                )

    summary_dir = base_config.train.results_dir / base_results_subdir / "noise_sweep"
    summary_path = save_json(
        summary_dir / args.summary_name,
        {
            "config": str(Path(args.config)),
            "base_results_subdir": base_results_subdir,
            "through_stage": args.through_stage,
            "scales": scales,
            "probabilities": probabilities,
            "seeds": seeds,
            "rows": rows,
        },
    )
    print(f"[INFO] Noise sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
