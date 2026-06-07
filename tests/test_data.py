import h5py
import numpy as np

from invariant_generator.config import Config
from invariant_generator.data import (
    augment_homogeneous_surface_data,
    canonicalize_stress_features,
    prepare_training_data,
)


def test_plane_stress_data_embeds_to_six_components():
    X2 = np.array([[1.0, 2.0, 3.0]])
    X6, names = canonicalize_stress_features(X2, stress_format="plane_stress_2d")

    assert names == ["s11", "s22", "s33", "s23", "s13", "s12"]
    np.testing.assert_allclose(X6, [[1.0, 2.0, 0.0, 0.0, 0.0, 3.0]])


def test_homogeneous_augmentation_targets_follow_scaling_law():
    X = np.ones((2, 6), dtype=np.float64)
    X_aug, y_aug = augment_homogeneous_surface_data(
        X,
        k_values=[0.5, 2.0],
        include_original=True,
        surface_target=1.0,
        homogeneity_degree=1.0,
        shuffle=False,
    )

    assert X_aug.shape == (6, 6)
    np.testing.assert_allclose(y_aug, [1.0, 1.0, 0.5, 2.0, 0.5, 2.0])
    np.testing.assert_allclose(X_aug[0], np.ones(6))
    np.testing.assert_allclose(X_aug[2], 0.5 * np.ones(6))
    np.testing.assert_allclose(X_aug[3], 2.0 * np.ones(6))


def test_prepare_training_data_can_augment_test_targets(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    with h5py.File(data_dir / "toy.h5", "w") as f:
        f.create_dataset("stress", data=X)

    config = Config()
    config.data.data_dir = data_dir
    config.data.dataset_name = "toy"
    config.data.dataset_key = "stress"
    config.data.test_size = 0.5
    config.data.shuffle = False
    config.train.split_dir = tmp_path / "splits"
    config.train.use_saved_split = False
    config.train.save_split_if_missing = False
    config.augmentation.augment_test = True
    config.augmentation.k_values = [0.5, 2.0]
    config.augmentation.include_original = True
    config.augmentation.shuffle = False

    data = prepare_training_data(config)

    assert data.X_test.shape == (6, 6)
    np.testing.assert_allclose(data.y_test, [1.0, 1.0, 0.5, 2.0, 0.5, 2.0])
    np.testing.assert_allclose(data.X_test[0], X[0])
    np.testing.assert_allclose(data.X_test[1], X[1])
    np.testing.assert_allclose(data.X_test[2], 0.5 * X[0])
    np.testing.assert_allclose(data.X_test[3], 2.0 * X[0])
