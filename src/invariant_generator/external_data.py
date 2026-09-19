from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np


STRESS_COMPONENTS = ("s11", "s22", "s33", "s23", "s13", "s12")
DX56D_SOURCE_URL = (
    "https://fordatis.fraunhofer.de/bitstream/fordatis/292.2/3/data.zip"
)
GOLD_SOURCE_URL = (
    "https://zenodo.org/records/21981226/files/"
    "single_crystal_Au_MiMeDat_FAIR_data_objects.json?download=1"
)


@dataclass(slots=True)
class ExternalDatasetResult:
    output_path: Path
    n_stress_points: int
    metadata: dict[str, Any]


def _dx56d_member(sampling: str) -> str:
    sampling = sampling.lower()
    if sampling == "miller":
        folder = "sampling_full_stress_state_miller"
        filename = "miller_yield_points.txt"
    elif sampling.startswith("active_learning_"):
        index = sampling.removeprefix("active_learning_")
        if index not in {"1", "2", "3", "4", "5"}:
            raise ValueError("DX56D active-learning sampling index must be 1 through 5.")
        folder = f"sampling_full_stress_state_active_learning_data_set_{index}"
        filename = f"active_learning_data_set_{index}_yield_points.txt"
    elif sampling.startswith("random_"):
        index = sampling.removeprefix("random_")
        if index not in {"1", "2", "3", "4", "5"}:
            raise ValueError("DX56D random sampling index must be 1 through 5.")
        folder = f"sampling_full_stress_state_random_data_set_{index}"
        filename = f"random_data_set_{index}_yield_points.txt"
    else:
        raise ValueError(
            "DX56D sampling must be 'miller', 'active_learning_1' through "
            "'active_learning_5', or 'random_1' through 'random_5'."
        )
    return (
        "data/02_crystal_plasticity_simulations/"
        f"{folder}/{filename}"
    )


def _read_archive_text(source: Path, member: str) -> str:
    if source.is_file() and source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            try:
                return archive.read(member).decode("utf-8-sig")
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{member!r} was not found in DX56D archive {source}."
                ) from exc

    candidate = source / member if source.is_dir() else source
    if not candidate.exists():
        raise FileNotFoundError(f"DX56D source file not found: {candidate}")
    return candidate.read_text(encoding="utf-8-sig")


def load_dx56d_yield_points(
    source: str | Path,
    *,
    sampling: str = "miller",
) -> np.ndarray:
    """Load one published DX56D full-stress yield-point table.

    The source tables already use the project's canonical Voigt ordering:
    [sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12].
    """
    source = Path(source)
    member = _dx56d_member(sampling)
    text = _read_archive_text(source, member)
    stress = np.loadtxt(
        StringIO(text),
        dtype=np.float64,
        skiprows=1,
        usecols=(3, 4, 5, 6, 7, 8),
    )
    stress = np.atleast_2d(stress)
    if stress.shape[1] != 6 or not np.all(np.isfinite(stress)):
        raise ValueError(
            f"Invalid DX56D stress table from {member}: shape={stress.shape}."
        )
    return stress


def equivalent_plastic_strain(plastic_strain_voigt: np.ndarray) -> np.ndarray:
    """J2-equivalent strain from tensor-shear Voigt components.

    Input order is [ep11, ep22, ep33, ep23, ep13, ep12]. The published gold
    JSON stores tensor components, so the off-diagonal terms occur twice in
    the tensor double contraction.
    """
    values = np.asarray(plastic_strain_voigt, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Plastic strain history must have shape [N, 6], got {values.shape}.")
    mean_normal = values[:, :3].mean(axis=1, keepdims=True)
    deviatoric_normal = values[:, :3] - mean_normal
    double_contraction = (
        np.sum(deviatoric_normal**2, axis=1)
        + 2.0 * np.sum(values[:, 3:] ** 2, axis=1)
    )
    return np.sqrt((2.0 / 3.0) * np.maximum(double_contraction, 0.0))


def _history_matrix(payload: dict[str, Any], keys: tuple[str, ...], *, label: str) -> np.ndarray:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Gold trajectory is missing {label} components: {missing}")
    lengths = {len(payload[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"Gold trajectory has inconsistent {label} history lengths: {lengths}")
    values = np.column_stack([payload[key] for key in keys]).astype(np.float64)
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"Gold trajectory has an invalid {label} history.")
    return values


def extract_gold_yield_points(
    source: str | Path,
    *,
    plastic_strain_threshold: float = 0.002,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Extract one interpolated yield point from each gold loading history.

    Yield is operationally defined by a configurable J2-equivalent plastic
    strain. Trajectories that never reach the threshold (notably a purely
    hydrostatic direction for pressure-insensitive crystal slip) are recorded
    and omitted.
    """
    if plastic_strain_threshold <= 0.0:
        raise ValueError("plastic_strain_threshold must be positive.")
    source = Path(source)
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict) or not records:
        raise ValueError("Gold source JSON must be a non-empty object of trajectories.")

    stress_rows: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    history_indices: list[int] = []
    interpolation_fractions: list[float] = []
    skipped_ids: list[str] = []

    stress_keys = ("S11", "S22", "S33", "S23", "S13", "S12")
    plastic_keys = ("Ep11", "Ep22", "Ep33", "Ep23", "Ep13", "Ep12")
    for trajectory_id in sorted(records):
        record = records[trajectory_id]
        stress = _history_matrix(record.get("stress", {}), stress_keys, label="stress")
        plastic = _history_matrix(
            record.get("total_strain", {}),
            plastic_keys,
            label="plastic strain",
        )
        if stress.shape[0] != plastic.shape[0]:
            raise ValueError(
                f"Gold trajectory {trajectory_id!r} has mismatched stress and strain histories."
            )

        equivalent = equivalent_plastic_strain(plastic)
        crossing = np.flatnonzero(equivalent >= plastic_strain_threshold)
        if crossing.size == 0:
            skipped_ids.append(str(trajectory_id))
            continue

        upper = int(crossing[0])
        lower = max(0, upper - 1)
        if upper == lower or equivalent[upper] <= equivalent[lower]:
            fraction = 0.0
            yield_stress = stress[upper]
        else:
            fraction = float(
                (plastic_strain_threshold - equivalent[lower])
                / (equivalent[upper] - equivalent[lower])
            )
            fraction = float(np.clip(fraction, 0.0, 1.0))
            yield_stress = stress[lower] + fraction * (stress[upper] - stress[lower])

        stress_rows.append(yield_stress)
        trajectory_ids.append(str(trajectory_id))
        history_indices.append(upper)
        interpolation_fractions.append(fraction)

    if not stress_rows:
        raise ValueError("No gold trajectories reached the requested plastic-strain threshold.")

    stress_array = np.vstack(stress_rows).astype(np.float64)
    metadata = {
        "source_dataset": "single_crystal_gold_cpfe",
        "source_url": GOLD_SOURCE_URL,
        "stress_units": "MPa",
        "stress_format": "voigt_3d",
        "yield_definition": "J2-equivalent plastic strain threshold",
        "plastic_strain_threshold": float(plastic_strain_threshold),
        "n_source_trajectories": len(records),
        "n_yield_points": len(stress_rows),
        "skipped_trajectory_ids": skipped_ids,
    }
    auxiliary = {
        "trajectory_id": np.asarray(trajectory_ids, dtype=object),
        "yield_history_index": np.asarray(history_indices, dtype=np.int64),
        "yield_interpolation_fraction": np.asarray(
            interpolation_fractions,
            dtype=np.float64,
        ),
    }
    return stress_array, metadata, auxiliary


def write_external_stress_hdf(
    output_path: str | Path,
    stress: np.ndarray,
    *,
    metadata: dict[str, Any],
    auxiliary: dict[str, np.ndarray] | None = None,
) -> Path:
    output_path = Path(output_path)
    values = np.asarray(stress, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Stress data must have shape [N, 6], got {values.shape}.")
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Stress data must contain at least two finite rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset("stress", data=values)
        dataset.attrs["component_order"] = json.dumps(STRESS_COMPONENTS)
        dataset.attrs["stress_format"] = "voigt_3d"
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        for name, auxiliary_values in (auxiliary or {}).items():
            array = np.asarray(auxiliary_values)
            if array.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Auxiliary array {name!r} has {array.shape[0]} rows; "
                    f"expected {values.shape[0]}."
                )
            if array.dtype.kind in {"O", "U"}:
                string_dtype = h5py.string_dtype(encoding="utf-8")
                handle.create_dataset(name, data=array.astype(string_dtype), dtype=string_dtype)
            else:
                handle.create_dataset(name, data=array)
    return output_path


def prepare_dx56d_dataset(
    source: str | Path,
    output_path: str | Path,
    *,
    sampling: str = "miller",
) -> ExternalDatasetResult:
    stress = load_dx56d_yield_points(source, sampling=sampling)
    metadata = {
        "source_dataset": "dx56d_cpfe_initial_yield_surface",
        "source_url": DX56D_SOURCE_URL,
        "sampling": sampling,
        "stress_units": "MPa",
        "stress_format": "voigt_3d",
        "n_yield_points": int(stress.shape[0]),
    }
    path = write_external_stress_hdf(output_path, stress, metadata=metadata)
    return ExternalDatasetResult(path, int(stress.shape[0]), metadata)


def prepare_gold_dataset(
    source: str | Path,
    output_path: str | Path,
    *,
    plastic_strain_threshold: float = 0.002,
) -> ExternalDatasetResult:
    stress, metadata, auxiliary = extract_gold_yield_points(
        source,
        plastic_strain_threshold=plastic_strain_threshold,
    )
    path = write_external_stress_hdf(
        output_path,
        stress,
        metadata=metadata,
        auxiliary=auxiliary,
    )
    return ExternalDatasetResult(path, int(stress.shape[0]), metadata)
