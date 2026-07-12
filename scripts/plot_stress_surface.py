from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from invariant_generator.config import load_config
from invariant_generator.data import (
    add_gaussian_input_noise,
    canonicalize_stress_features,
    load_hdf_dataset,
    split_surface_data,
)


def _float_tag(value: float) -> str:
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return [float(item) for item in values]


def _stress_vector_to_tensor_np(stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=np.float64)
    sigma = np.zeros(stress.shape[:-1] + (3, 3), dtype=np.float64)
    sigma[..., 0, 0] = stress[..., 0]
    sigma[..., 1, 1] = stress[..., 1]
    sigma[..., 2, 2] = stress[..., 2]
    sigma[..., 1, 2] = stress[..., 3]
    sigma[..., 2, 1] = stress[..., 3]
    sigma[..., 0, 2] = stress[..., 4]
    sigma[..., 2, 0] = stress[..., 4]
    sigma[..., 0, 1] = stress[..., 5]
    sigma[..., 1, 0] = stress[..., 5]
    return sigma


def _pca_fit_transform(clean: np.ndarray, other: np.ndarray, n_components: int = 3):
    mean = clean.mean(axis=0, keepdims=True)
    centered = clean - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T, (other - mean) @ components.T


def _principal_stresses(stress: np.ndarray) -> np.ndarray:
    sigma = _stress_vector_to_tensor_np(stress)
    values = np.linalg.eigvalsh(sigma)
    return values[:, ::-1]


def _deviatoric_plane(stress: np.ndarray) -> np.ndarray:
    principal = _principal_stresses(stress)
    dev = principal - principal.mean(axis=1, keepdims=True)
    x = (dev[:, 0] - dev[:, 1]) / np.sqrt(2.0)
    y = (dev[:, 0] + dev[:, 1] - 2.0 * dev[:, 2]) / np.sqrt(6.0)
    return np.column_stack([x, y])


def _sample_rows(X: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[np.sort(indices)]


def _apply_noise_with_probability(
    X: np.ndarray,
    *,
    scale: float,
    probability: float,
    seed: int,
    relative_to_feature_std: bool,
) -> np.ndarray:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    noisy = add_gaussian_input_noise(
        X,
        noise_scale=scale,
        random_state=seed,
        relative_to_feature_std=relative_to_feature_std,
    )
    if probability >= 1.0:
        return noisy
    rng = np.random.default_rng(seed + 1)
    mask = rng.random((X.shape[0], 1)) < probability
    return np.where(mask, noisy, X)


def _plot_pca(clean: np.ndarray, noisy: np.ndarray, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    clean_pca, noisy_pca = _pca_fit_transform(clean, noisy, n_components=2)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(clean_pca[:, 0], clean_pca[:, 1], s=7, alpha=0.45, label="clean")
    ax.scatter(noisy_pca[:, 0], noisy_pca[:, 1], s=7, alpha=0.45, label="noisy")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_principal(clean: np.ndarray, noisy: np.ndarray, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    clean_principal = _principal_stresses(clean)
    noisy_principal = _principal_stresses(noisy)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        clean_principal[:, 0],
        clean_principal[:, 1],
        s=7,
        alpha=0.45,
        label="clean",
    )
    ax.scatter(
        noisy_principal[:, 0],
        noisy_principal[:, 1],
        s=7,
        alpha=0.45,
        label="noisy",
    )
    ax.set_xlabel("principal stress 1")
    ax.set_ylabel("principal stress 2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_deviatoric(clean: np.ndarray, noisy: np.ndarray, path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    clean_dev = _deviatoric_plane(clean)
    noisy_dev = _deviatoric_plane(noisy)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(clean_dev[:, 0], clean_dev[:, 1], s=7, alpha=0.45, label="clean")
    ax.scatter(noisy_dev[:, 0], noisy_dev[:, 1], s=7, alpha=0.45, label="noisy")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("deviatoric coordinate 1")
    ax.set_ylabel("deviatoric coordinate 2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_noise_magnitude(
    clean: np.ndarray,
    noisy: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    denominator = np.linalg.norm(clean, axis=1)
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    relative = np.linalg.norm(noisy - clean, axis=1) / denominator
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(relative, bins=40)
    ax.set_xlabel("||noise|| / ||clean stress||")
    ax.set_ylabel("sample count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    clean = _sample_rows(clean, max_samples=args.max_samples, seed=args.seed)
    scales = _parse_float_list(args.scales)
    relative_to_feature_std = not args.absolute_noise

    for scale in scales:
        noisy = _apply_noise_with_probability(
            clean,
            scale=scale,
            probability=args.probability,
            seed=args.seed,
            relative_to_feature_std=relative_to_feature_std,
        )
        tag = f"scale_{_float_tag(scale)}_prob_{_float_tag(args.probability)}"
        title = f"{args.split} stress surface, noise scale={scale:g}, p={args.probability:g}"
        _plot_pca(clean, noisy, output_dir / f"{tag}_pca.png", title=title)
        _plot_principal(
            clean,
            noisy,
            output_dir / f"{tag}_principal_stress.png",
            title=title,
        )
        _plot_deviatoric(
            clean,
            noisy,
            output_dir / f"{tag}_deviatoric_plane.png",
            title=title,
        )
        _plot_noise_magnitude(
            clean,
            noisy,
            output_dir / f"{tag}_noise_magnitude.png",
            title=title,
        )

    print(f"[INFO] Wrote stress-space plots to {output_dir}")


if __name__ == "__main__":
    main()
