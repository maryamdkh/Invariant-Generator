from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import json

import h5py
import numpy as np

from invariant_generator.config import Config, STRESS_NAMES


@dataclass(slots=True)
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    split_path: Path


def _resolve_hdf_path(data_dir: Path, dataset_name: str) -> Path:
    candidate = Path(dataset_name)
    if candidate.suffix in {".hdf", ".h5", ".hdf5"}:
        return candidate if candidate.is_absolute() else data_dir / candidate

    for suffix in (".hdf", ".h5", ".hdf5"):
        path = data_dir / f"{dataset_name}{suffix}"
        if path.exists():
            return path

    return data_dir / f"{dataset_name}.hdf"


def load_hdf_dataset(
    data_dir: Path,
    dataset_name: str,
    *,
    dataset_key: str | None = None,
    candidate_keys: tuple[str, ...] = ("stress", "X", "x", "data"),
) -> np.ndarray:
    """Load a stress dataset from an HDF file."""
    file_path = _resolve_hdf_path(data_dir, dataset_name)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        if dataset_key is not None:
            if dataset_key not in f:
                raise KeyError(
                    f"Dataset key '{dataset_key}' not found in {file_path}. "
                    f"Available keys: {list(f.keys())}"
                )
            X = np.asarray(f[dataset_key], dtype=np.float64)
        else:
            found_key = next((key for key in candidate_keys if key in f), None)
            if found_key is None:
                raise KeyError(
                    f"No dataset key found in {file_path}. Tried {candidate_keys}. "
                    f"Available keys: {list(f.keys())}"
                )
            X = np.asarray(f[found_key], dtype=np.float64)

    return X


def canonicalize_stress_features(
    X: np.ndarray,
    *,
    stress_format: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Return stresses in canonical 6-component Voigt order.

    The invariant layer expects:
        [s11, s22, s33, s23, s13, s12]

    Old 2D plane-stress datasets can be embedded by setting
    stress_format="plane_stress_2d"; they must be ordered as [s11, s22, s12].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Stress array must be 2D, got shape {X.shape}.")

    if stress_format == "voigt_3d":
        if X.shape[1] != 6:
            raise ValueError(
                "stress_format='voigt_3d' requires 6 columns ordered as "
                "[s11, s22, s33, s23, s13, s12]. "
                f"Got shape {X.shape}."
            )
        return X.copy(), STRESS_NAMES.copy()

    if stress_format == "plane_stress_2d":
        if X.shape[1] != 3:
            raise ValueError(
                "stress_format='plane_stress_2d' requires 3 columns ordered as "
                "[s11, s22, s12]. "
                f"Got shape {X.shape}."
            )
        X6 = np.zeros((X.shape[0], 6), dtype=np.float64)
        X6[:, 0] = X[:, 0]  # s11
        X6[:, 1] = X[:, 1]  # s22
        X6[:, 5] = X[:, 2]  # s12
        return X6, STRESS_NAMES.copy()

    raise ValueError(
        "stress_format must be either 'voigt_3d' or 'plane_stress_2d', "
        f"got {stress_format!r}."
    )


def add_gaussian_input_noise(
    X: np.ndarray,
    *,
    noise_scale: float,
    random_state: int | None = None,
    relative_to_feature_std: bool = True,
) -> np.ndarray:
    """Add Gaussian perturbation to stress components."""
    X = np.asarray(X, dtype=np.float64)
    if noise_scale < 0:
        raise ValueError("noise_scale must be >= 0.")
    if noise_scale == 0:
        return X.copy()

    rng = np.random.default_rng(random_state)
    if relative_to_feature_std:
        feature_std = X.std(axis=0, ddof=0)
        feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
        noise = rng.normal(loc=0.0, scale=noise_scale, size=X.shape) * feature_std
    else:
        noise = rng.normal(loc=0.0, scale=noise_scale, size=X.shape)

    return X + noise


def save_surface_split(
    split_path: str | Path,
    X_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    test_size: float,
    random_state: Optional[int],
    shuffle: bool,
) -> None:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "test_size": float(test_size),
        "random_state": random_state,
        "shuffle": bool(shuffle),
        "n_train": int(X_train_raw.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train_raw.shape[1]),
    }

    np.savez_compressed(
        split_path,
        X_train_raw=np.asarray(X_train_raw, dtype=np.float64),
        X_test=np.asarray(X_test, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=np.float64),
        metadata=json.dumps(metadata),
    )


def load_surface_split(
    split_path: str | Path,
    *,
    expected_n_features: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data = np.load(split_path, allow_pickle=False)
    required = {"X_train_raw", "X_test", "y_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Split file is missing arrays: {sorted(missing)}")

    X_train_raw = np.asarray(data["X_train_raw"], dtype=np.float64)
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)

    if X_train_raw.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Loaded stress arrays must be 2D.")
    if X_train_raw.shape[1] != expected_n_features or X_test.shape[1] != expected_n_features:
        raise ValueError(
            f"Expected {expected_n_features} features, got "
            f"{X_train_raw.shape[1]} and {X_test.shape[1]}."
        )
    if y_test.ndim != 1 or y_test.shape[0] != X_test.shape[0]:
        raise ValueError("Loaded y_test has an invalid shape.")

    return X_train_raw, X_test, y_test


def split_surface_data(
    X: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    split_path: str | Path | None = None,
    load_if_exists: bool = True,
    save_if_missing: bool = False,
    surface_target: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create or reuse a train/test split for surface points."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples.")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    if split_path is not None:
        split_path = Path(split_path)
        if load_if_exists and split_path.exists():
            return load_surface_split(split_path, expected_n_features=X.shape[1])

    n_samples = X.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("Empty train or test split.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train_raw = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_test = np.full(X_test.shape[0], surface_target, dtype=np.float64)

    if split_path is not None and save_if_missing:
        save_surface_split(
            split_path,
            X_train_raw,
            X_test,
            y_test,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    return X_train_raw, X_test, y_test


def augment_homogeneous_surface_data(
    X_train_raw: np.ndarray,
    *,
    k_values: Optional[Sequence[float]] = None,
    n_aug_per_sample: int = 4,
    k_range: tuple[float, float] = (0.85, 1.15),
    include_original: bool = True,
    surface_target: float = 1.0,
    homogeneity_degree: float = 1.0,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment surface samples using f(k*sigma)=|k|^p f(sigma).

    If original samples satisfy f(sigma)=surface_target, scaled samples satisfy:
        f(k*sigma)=|k|^p * surface_target.
    """
    X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
    if X_train_raw.ndim != 2:
        raise ValueError(f"X_train_raw must be 2D, got shape {X_train_raw.shape}.")
    if X_train_raw.shape[0] == 0:
        raise ValueError("X_train_raw is empty.")
    if n_aug_per_sample < 0:
        raise ValueError("n_aug_per_sample must be >= 0.")

    rng = np.random.default_rng(random_state)
    n_train, n_features = X_train_raw.shape

    if k_values is not None:
        ks = np.asarray(k_values, dtype=np.float64).reshape(-1)
        if ks.size == 0:
            raise ValueError("k_values must contain at least one value.")
        if np.any(np.isclose(ks, 0.0)):
            raise ValueError("k_values must not contain 0.")
        if include_original:
            ks = ks[~np.isclose(ks, 1.0)]

        if ks.size:
            X_scaled = (X_train_raw[:, None, :] * ks[None, :, None]).reshape(
                -1, n_features
            )
            y_scaled = (np.abs(ks) ** homogeneity_degree * surface_target)
            y_scaled = np.broadcast_to(y_scaled[None, :], (n_train, ks.size)).reshape(-1)
        else:
            X_scaled = np.empty((0, n_features), dtype=np.float64)
            y_scaled = np.empty((0,), dtype=np.float64)
    else:
        if n_aug_per_sample == 0:
            X_scaled = np.empty((0, n_features), dtype=np.float64)
            y_scaled = np.empty((0,), dtype=np.float64)
        else:
            k_low, k_high = map(float, k_range)
            if not k_low < k_high:
                raise ValueError(f"k_range must satisfy low < high, got {k_range}.")
            if k_low <= 0.0 <= k_high:
                raise ValueError("k_range must not cross 0.")

            ks = rng.uniform(k_low, k_high, size=(n_train, n_aug_per_sample))
            X_scaled = (X_train_raw[:, None, :] * ks[:, :, None]).reshape(
                -1, n_features
            )
            y_scaled = (np.abs(ks) ** homogeneity_degree * surface_target).reshape(-1)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    if include_original:
        X_parts.append(X_train_raw)
        y_parts.append(np.full(n_train, surface_target, dtype=np.float64))
    if X_scaled.shape[0] > 0:
        X_parts.append(X_scaled)
        y_parts.append(y_scaled)
    if not X_parts:
        raise ValueError("No training data produced by augmentation settings.")

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    if shuffle:
        perm = rng.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]

    return X_train, y_train


def prepare_training_data(config: Config) -> PreparedData:
    """Load, split, perturb, and augment data according to the config."""
    X_raw = load_hdf_dataset(
        config.data.data_dir,
        config.data.dataset_name,
        dataset_key=config.data.dataset_key,
    )
    X, feature_names = canonicalize_stress_features(
        X_raw,
        stress_format=config.data.stress_format,
    )

    X_train_raw, X_test, y_test = split_surface_data(
        X,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        shuffle=config.data.shuffle,
        split_path=config.split_path,
        load_if_exists=config.train.use_saved_split,
        save_if_missing=config.train.save_split_if_missing,
        surface_target=config.augmentation.surface_target,
    )

    if config.noise.enabled:
        X_train_raw = add_gaussian_input_noise(
            X_train_raw,
            noise_scale=config.noise.scale,
            random_state=config.noise.random_state,
            relative_to_feature_std=config.noise.relative_to_feature_std,
        )

    if config.augmentation.enabled:
        augmentation_options = {
            "k_values": config.augmentation.k_values,
            "n_aug_per_sample": config.augmentation.n_aug_per_sample,
            "k_range": config.augmentation.k_range,
            "include_original": config.augmentation.include_original,
            "surface_target": config.augmentation.surface_target,
            "homogeneity_degree": config.augmentation.homogeneity_degree,
            "random_state": config.augmentation.random_state,
            "shuffle": config.augmentation.shuffle,
        }
        X_train, y_train = augment_homogeneous_surface_data(
            X_train_raw,
            **augmentation_options,
        )
        if config.augmentation.augment_test:
            X_test, y_test = augment_homogeneous_surface_data(
                X_test,
                **augmentation_options,
            )
    else:
        X_train = X_train_raw
        y_train = np.full(X_train.shape[0], config.augmentation.surface_target, dtype=np.float64)

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        split_path=config.split_path,
    )
