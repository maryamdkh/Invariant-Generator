import numpy as np

from invariant_generator.data import (
    augment_homogeneous_surface_data,
    canonicalize_stress_features,
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
