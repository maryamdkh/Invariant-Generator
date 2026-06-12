from __future__ import annotations

import argparse
from copy import deepcopy

from invariant_generator.config import Config, load_config
from invariant_generator.train import train_from_config
from invariant_generator.utils import save_json


def _base_recovery_config(config: Config) -> Config:
    variant = deepcopy(config)
    variant.data.dataset_name = "rotatedhill"
    variant.encoder.enabled = True
    variant.encoder.output_dim = 0
    variant.encoder.init = "identity"
    return variant


def _variant(config: Config, *, name: str, psd: bool, normalize: bool) -> Config:
    variant = _base_recovery_config(config)
    variant.train.run_id = f"{config.train.run_id}_{name}"
    variant.constraints.A_psd.enabled = psd
    variant.constraints.A_psd.mode = "hard" if psd else "check"
    variant.normalization.enabled = normalize
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run rotated-Hill sanity variants: unconstrained, PSD-only, "
            "and PSD plus invariant standardization."
        )
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id-prefix", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_id_prefix is not None:
        config.train.run_id = args.run_id_prefix
    if args.epochs is not None:
        config.train.epochs = args.epochs

    variants = [
        _variant(config, name="bench_unconstrained", psd=False, normalize=False),
        _variant(config, name="bench_psd", psd=True, normalize=False),
        _variant(config, name="bench_psd_standardized", psd=True, normalize=True),
    ]

    summaries = []
    for variant in variants:
        print(f"[INFO] Running {variant.train.run_id}")
        result = train_from_config(variant)
        summaries.append(
            {
                "run_id": variant.train.run_id,
                "experiment_dir": str(result.experiment_dir),
                "best_epoch": result.best_epoch,
                "best_test_mse": result.best_test_mse,
                "A_psd_enabled": variant.constraints.A_psd.enabled,
                "normalization_enabled": variant.normalization.enabled,
            }
        )

    summary_path = save_json(
        config.train.results_dir / f"{config.train.run_id}_benchmark_summary.json",
        {"variants": summaries},
    )
    print(f"[INFO] Benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
