import numpy as np

from invariant_generator.stress_viz import (
    deviatoric_plane_coordinates,
    pca_project_clean_and_other,
    principal_stresses,
    relative_noise_magnitude,
    simulate_train_input_noise,
)


def test_pca_projection_shapes_and_noise_determinism():
    X = np.arange(60, dtype=np.float64).reshape(10, 6)
    noisy_a = simulate_train_input_noise(
        X,
        scale=0.05,
        probability=1.0,
        seed=123,
        relative_to_feature_std=True,
    )
    noisy_b = simulate_train_input_noise(
        X,
        scale=0.05,
        probability=1.0,
        seed=123,
        relative_to_feature_std=True,
    )

    clean_pca, noisy_pca = pca_project_clean_and_other(X, noisy_a, n_components=2)

    np.testing.assert_allclose(noisy_a, noisy_b)
    assert clean_pca.shape == (10, 2)
    assert noisy_pca.shape == (10, 2)
    assert np.isfinite(clean_pca).all()
    assert np.isfinite(noisy_pca).all()
    assert np.all(relative_noise_magnitude(X, noisy_a) > 0.0)


def test_physical_projection_shapes():
    X = np.array(
        [
            [3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.2, 0.1, 0.3],
        ],
        dtype=np.float64,
    )

    principal = principal_stresses(X)
    deviatoric = deviatoric_plane_coordinates(X)

    assert principal.shape == (2, 3)
    assert deviatoric.shape == (2, 2)
    assert np.all(np.diff(principal, axis=1) <= 0.0)
