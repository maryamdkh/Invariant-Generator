from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from invariant_generator.config import load_config
from invariant_generator.data import (
    canonicalize_stress_features,
    load_hdf_dataset,
    split_surface_data,
)
from invariant_generator.stress_viz import (
    sample_stress_rows,
    save_stress_space_plots,
    simulate_train_input_noise,
)


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return [float(item) for item in values]


def _load_clean_surface(config_path: str, *, split: str) -> np.ndarray:
    config = load_config(config_path)
    X_raw = load_hdf_dataset(
        config.data.data_dir,
        config.data.dataset_name,
        dataset_key=config.data.dataset_key,
    )
    X, _ = canonicalize_stress_features(
        X_raw,
        stress_format=config.data.stress_format,
    )
    X_train_raw, X_test, _ = split_surface_data(
        X,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        shuffle=config.data.shuffle,
        split_path=config.split_path,
        load_if_exists=config.train.use_saved_split,
        save_if_missing=config.train.save_split_if_missing,
        surface_target=config.augmentation.surface_target,
    )
    if split == "train":
        return X_train_raw
    if split == "test":
        return X_test
    if split == "all":
        return np.vstack([X_train_raw, X_test])
    raise ValueError("split must be 'train', 'test', or 'all'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clean and simulated noisy 6D stress-surface samples."
    )
    parser.add_argument(
        "--config",
        default="configs/adaptive_encoder_rotated_hill.toml",
        help="Config used to load the clean dataset and split.",
    )
    parser.add_argument(
        "--scales",
        default="0.01,0.02,0.05",
        help="Comma-separated noise scales to visualize.",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=1.0,
        help="Probability that a sample is perturbed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Noise and subsampling seed.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help="Clean split to visualize before homogeneous augmentation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of points plotted per scale; 0 disables subsampling.",
    )
    parser.add_argument(
        "--absolute-noise",
        action="store_true",
        help="Use absolute component noise instead of feature-std-relative noise.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs. Defaults to results/stress_surface_plots.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.train.results_dir / "stress_surface_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = _load_clean_surface(args.config, split=args.split)
    clean = sample_stress_rows(clean, max_samples=args.max_samples, seed=args.seed)
    scales = _parse_float_list(args.scales)
    relative_to_feature_std = not args.absolute_noise

    for scale in scales:
        noisy = simulate_train_input_noise(
            clean,
            scale=scale,
            probability=args.probability,
            seed=args.seed,
            relative_to_feature_std=relative_to_feature_std,
        )
        tag = f"scale_{_float_tag(scale)}_prob_{_float_tag(args.probability)}"
        title = f"{args.split} stress surface, noise scale={scale:g}, p={args.probability:g}"
        save_stress_space_plots(clean, noisy, output_dir, tag=tag, title=title)

    print(f"[INFO] Wrote stress-space plots to {output_dir}")


if __name__ == "__main__":
    main()
