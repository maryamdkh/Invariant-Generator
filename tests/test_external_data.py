from __future__ import annotations

import json
from zipfile import ZipFile

import h5py
import numpy as np

from invariant_generator.external_data import (
    equivalent_plastic_strain,
    extract_gold_yield_points,
    load_dx56d_yield_points,
    write_external_stress_hdf,
)


def test_load_dx56d_yield_points_selects_canonical_stress_columns(tmp_path):
    archive_path = tmp_path / "data.zip"
    member = (
        "data/02_crystal_plasticity_simulations/"
        "sampling_full_stress_state_miller/miller_yield_points.txt"
    )
    table = (
        "job work equivalent sigma_11 sigma_22 sigma_33 sigma_23 sigma_13 sigma_12\n"
        "1 24.58 280.0 11 22 33 23 13 12\n"
        "2 24.58 290.0 -11 -22 -33 -23 -13 -12\n"
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member, table)

    stress = load_dx56d_yield_points(archive_path, sampling="miller")

    np.testing.assert_allclose(
        stress,
        [[11, 22, 33, 23, 13, 12], [-11, -22, -33, -23, -13, -12]],
    )


def test_equivalent_plastic_strain_uses_tensor_shear_convention():
    values = np.asarray([[0.0, 0.0, 0.0, 0.003, 0.0, 0.0]])
    expected = np.sqrt((2.0 / 3.0) * 2.0 * 0.003**2)
    np.testing.assert_allclose(equivalent_plastic_strain(values), [expected])


def test_extract_gold_yield_points_interpolates_and_skips_non_yielding_path(tmp_path):
    def record(stress_scale: float, plastic_values: list[float]) -> dict[str, object]:
        zeros = [0.0] * len(plastic_values)
        return {
            "stress": {
                "S11": [0.0, stress_scale, 2.0 * stress_scale],
                "S22": zeros,
                "S33": zeros,
                "S23": zeros,
                "S13": zeros,
                "S12": zeros,
            },
            "total_strain": {
                "Ep11": plastic_values,
                "Ep22": [-value / 2.0 for value in plastic_values],
                "Ep33": [-value / 2.0 for value in plastic_values],
                "Ep23": zeros,
                "Ep13": zeros,
                "Ep12": zeros,
            },
        }

    source = tmp_path / "gold.json"
    source.write_text(
        json.dumps(
            {
                "yielding": record(100.0, [0.0, 0.001, 0.003]),
                "hydrostatic": record(50.0, [0.0, 0.0, 0.0]),
            }
        ),
        encoding="utf-8",
    )

    stress, metadata, auxiliary = extract_gold_yield_points(
        source,
        plastic_strain_threshold=0.002,
    )

    assert stress.shape == (1, 6)
    np.testing.assert_allclose(stress[0], [150.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert metadata["skipped_trajectory_ids"] == ["hydrostatic"]
    assert auxiliary["trajectory_id"].tolist() == ["yielding"]
    assert auxiliary["yield_history_index"].tolist() == [2]


def test_write_external_stress_hdf_matches_existing_loader_contract(tmp_path):
    output = tmp_path / "external.hdf"
    stress = np.arange(12, dtype=np.float64).reshape(2, 6)

    write_external_stress_hdf(
        output,
        stress,
        metadata={"source_dataset": "test"},
        auxiliary={"trajectory_id": np.asarray(["a", "b"], dtype=object)},
    )

    with h5py.File(output, "r") as handle:
        np.testing.assert_allclose(handle["stress"][:], stress)
        assert json.loads(handle.attrs["metadata_json"])["source_dataset"] == "test"
        assert handle["stress"].attrs["stress_format"] == "voigt_3d"
