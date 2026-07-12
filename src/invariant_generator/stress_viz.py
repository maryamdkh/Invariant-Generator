from __future__ import annotations

from pathlib import Path

import numpy as np

from invariant_generator.data import add_gaussian_input_noise


def stress_vector_to_tensor_np(stress: np.ndarray) -> np.ndarray:
    """Convert canonical 6-component stress vectors to symmetric 3x3 tensors."""
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape[-1] != 6:
        raise ValueError(f"Expected 6 stress components, got shape {stress.shape}.")
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


def pca_project_clean_and_other(
    clean: np.ndarray,
    other: np.ndarray,
    *,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA axes on clean stresses and project clean/other into that basis."""
    clean = np.asarray(clean, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if clean.ndim != 2 or other.ndim != 2:
        raise ValueError("clean and other stress arrays must be 2D.")
    if clean.shape[1] != other.shape[1]:
        raise ValueError("clean and other stress arrays must have the same width.")
    if not 1 <= n_components <= clean.shape[1]:
        raise ValueError("n_components must be between 1 and the stress dimension.")

    mean = clean.mean(axis=0, keepdims=True)
    centered = clean - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T, (other - mean) @ components.T


def principal_stresses(stress: np.ndarray) -> np.ndarray:
    """Return principal stresses sorted from largest to smallest."""
    sigma = stress_vector_to_tensor_np(stress)
    values = np.linalg.eigvalsh(sigma)
    return values[:, ::-1]


def deviatoric_plane_coordinates(stress: np.ndarray) -> np.ndarray:
    """Project principal deviatoric stresses into a 2D pi-plane coordinate system."""
    principal = principal_stresses(stress)
    dev = principal - principal.mean(axis=1, keepdims=True)
    x = (dev[:, 0] - dev[:, 1]) / np.sqrt(2.0)
    y = (dev[:, 0] + dev[:, 1] - 2.0 * dev[:, 2]) / np.sqrt(6.0)
    return np.column_stack([x, y])


def sample_stress_rows(X: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[np.sort(indices)]


def simulate_train_input_noise(
    X: np.ndarray,
    *,
    scale: float,
    probability: float = 1.0,
    seed: int = 42,
    relative_to_feature_std: bool = True,
) -> np.ndarray:
    """Simulate the same per-sample noise policy used at the training batch boundary."""
    X = np.asarray(X, dtype=np.float64)
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
    if probability == 0.0 or scale == 0.0:
        return X.copy()
    rng = np.random.default_rng(seed + 1)
    mask = rng.random((X.shape[0], 1)) < probability
    return np.where(mask, noisy, X)


def relative_noise_magnitude(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    denominator = np.linalg.norm(clean, axis=1)
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    return np.linalg.norm(noisy - clean, axis=1) / denominator


def plot_stress_space_summary(
    clean: np.ndarray,
    *,
    noisy: np.ndarray | None = None,
    title: str = "Stress-space sample distribution",
    max_samples: int = 5000,
    seed: int = 42,
) -> object:
    """Create PCA, principal-stress, deviatoric-plane, and noise-magnitude plots."""
    import matplotlib.pyplot as plt

    clean_sample = sample_stress_rows(clean, max_samples=max_samples, seed=seed)
    noisy_sample = None
    if noisy is not None:
        noisy_sample = sample_stress_rows(noisy, max_samples=max_samples, seed=seed)
        n = min(clean_sample.shape[0], noisy_sample.shape[0])
        clean_sample = clean_sample[:n]
        noisy_sample = noisy_sample[:n]

    comparison = clean_sample if noisy_sample is None else noisy_sample
    clean_pca, other_pca = pca_project_clean_and_other(
        clean_sample,
        comparison,
        n_components=2,
    )
    clean_principal = principal_stresses(clean_sample)
    clean_dev = deviatoric_plane_coordinates(clean_sample)
    other_principal = principal_stresses(comparison)
    other_dev = deviatoric_plane_coordinates(comparison)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    label_other = "noisy" if noisy_sample is not None else "clean"

    axes[0].scatter(clean_pca[:, 0], clean_pca[:, 1], s=7, alpha=0.45, label="clean")
    if noisy_sample is not None:
        axes[0].scatter(other_pca[:, 0], other_pca[:, 1], s=7, alpha=0.45, label=label_other)
    axes[0].set_title("PCA projection of 6D stress")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    axes[1].scatter(
        clean_principal[:, 0],
        clean_principal[:, 1],
        s=7,
        alpha=0.45,
        label="clean",
    )
    if noisy_sample is not None:
        axes[1].scatter(
            other_principal[:, 0],
            other_principal[:, 1],
            s=7,
            alpha=0.45,
            label=label_other,
        )
    axes[1].set_title("Principal-stress projection")
    axes[1].set_xlabel("principal stress 1")
    axes[1].set_ylabel("principal stress 2")
    axes[1].legend()

    axes[2].scatter(clean_dev[:, 0], clean_dev[:, 1], s=7, alpha=0.45, label="clean")
    if noisy_sample is not None:
        axes[2].scatter(other_dev[:, 0], other_dev[:, 1], s=7, alpha=0.45, label=label_other)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("Deviatoric pi-plane projection")
    axes[2].set_xlabel("deviatoric coordinate 1")
    axes[2].set_ylabel("deviatoric coordinate 2")
    axes[2].legend()

    if noisy_sample is None:
        stress_norm = np.linalg.norm(clean_sample, axis=1)
        axes[3].hist(stress_norm, bins=40)
        axes[3].set_title("Clean stress norm")
        axes[3].set_xlabel("||stress||")
    else:
        axes[3].hist(relative_noise_magnitude(clean_sample, noisy_sample), bins=40)
        axes[3].set_title("Relative perturbation magnitude")
        axes[3].set_xlabel("||noise|| / ||clean stress||")
    axes[3].set_ylabel("sample count")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_stress_space_plots(
    clean: np.ndarray,
    noisy: np.ndarray,
    output_dir: str | Path,
    *,
    tag: str,
    title: str,
) -> list[Path]:
    """Save the four stress-space diagnostic plots as separate PNG files."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_pca, noisy_pca = pca_project_clean_and_other(clean, noisy, n_components=2)
    clean_principal = principal_stresses(clean)
    noisy_principal = principal_stresses(noisy)
    clean_dev = deviatoric_plane_coordinates(clean)
    noisy_dev = deviatoric_plane_coordinates(noisy)
    paths: list[Path] = []

    specs = [
        (
            "pca",
            clean_pca,
            noisy_pca,
            "PC1",
            "PC2",
            "PCA projection of 6D stress",
            False,
        ),
        (
            "principal_stress",
            clean_principal[:, :2],
            noisy_principal[:, :2],
            "principal stress 1",
            "principal stress 2",
            "Principal-stress projection",
            False,
        ),
        (
            "deviatoric_plane",
            clean_dev,
            noisy_dev,
            "deviatoric coordinate 1",
            "deviatoric coordinate 2",
            "Deviatoric pi-plane projection",
            True,
        ),
    ]
    for suffix, clean_xy, noisy_xy, xlabel, ylabel, subtitle, equal_aspect in specs:
        path = output_dir / f"{tag}_{suffix}.png"
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(clean_xy[:, 0], clean_xy[:, 1], s=7, alpha=0.45, label="clean")
        ax.scatter(noisy_xy[:, 0], noisy_xy[:, 1], s=7, alpha=0.45, label="noisy")
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{subtitle}: {title}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    path = output_dir / f"{tag}_noise_magnitude.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(relative_noise_magnitude(clean, noisy), bins=40)
    ax.set_xlabel("||noise|| / ||clean stress||")
    ax.set_ylabel("sample count")
    ax.set_title(f"Relative perturbation magnitude: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths
