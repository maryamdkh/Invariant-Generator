"""Regenerate the Hill-ladder yield datasets with points in all stress directions.

The original files (data/vonMises25.hdf, vonMises100.hdf, hill.hdf,
rotatedhill.hdf) only contain points with every component >= 0. They are left
untouched. Their quadratic forms sigma^T M sigma = 1 (Mandel) are recovered
exactly by least squares on the 21 quadratic monomials, and new points are
sampled like scripts/prepare_yield_benchmark.py: directions uniform on the unit
sphere, scaled onto the surface.

Writes data/hill_ladder_full/<name>_full.hdf (key "stress") and a JSON with M:
  vonMises25, vonMises100, hill, rotatedhill   6 columns, mandel_3d
  rotatedhill2d                                3 columns [s11, s22, s12],
      plane_stress_2d, the plane-stress section of the rotated-Hill tensor.

    uv run python scripts/prepare_hill_ladder.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import h5py
import numpy as np

SOURCES = ["vonMises25", "vonMises100", "hill", "rotatedhill"]
PAIRS = list(combinations_with_replacement(range(6), 2))
PLANE_STRESS = [0, 1, 5]  # Mandel indices of s11, s22, sqrt(2)*s12


def fit_quadratic_form(stress: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.column_stack([stress[:, i] * stress[:, j] for i, j in PAIRS])
    c, *_ = np.linalg.lstsq(X, np.ones(len(stress)), rcond=None)
    M = np.zeros((6, 6))
    for (i, j), v in zip(PAIRS, c):
        M[i, j] = M[j, i] = v if i == j else v / 2
    return M, float(np.abs(X @ c - 1).max())


def sample_surface(M: np.ndarray, samples: int, seed: int) -> np.ndarray:
    directions = np.random.default_rng(seed).normal(size=(samples, M.shape[0]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.sqrt(np.einsum("ni,ij,nj->n", directions, M, directions))[:, None]


def write(path: Path, stress: np.ndarray, meta: dict) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("stress", data=stress)
        handle["stress"].attrs["stress_format"] = meta["stress_format"]
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{path.name:28s} {stress.shape}  max |f - 1| = {meta['max_abs_surface_error']:.1e}"
          f"  rows with a negative entry = {meta['fraction_rows_with_negative']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hill_ladder_full"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in SOURCES:
        old = np.asarray(h5py.File(args.source_dir / f"{name}.hdf", "r")["stress"][()], float)
        M, fit_error = fit_quadratic_form(old)
        if fit_error > 1e-8:
            raise RuntimeError(f"{name}: not an exact quadratic form (residual {fit_error:.1e}).")
        variants = [(name, M, "mandel_3d")]
        if name == "rotatedhill":
            variants.append(("rotatedhill2d", M[np.ix_(PLANE_STRESS, PLANE_STRESS)], "plane_stress_2d"))
        for out_name, M_out, stress_format in variants:
            stress = sample_surface(M_out, args.samples, args.seed)
            error = float(np.abs(np.einsum("ni,ij,nj->n", stress, M_out, stress) - 1).max())
            if not np.isfinite(stress).all() or error > 1e-10:
                raise RuntimeError(f"{out_name}: invalid surface points (max |f - 1| = {error}).")
            if stress_format == "plane_stress_2d":
                stress = stress.copy()
                stress[:, 2] /= np.sqrt(2.0)  # Mandel sqrt(2)*s12 -> tensor s12
            write(args.output_dir / f"{out_name}_full.hdf", stress, {
                "name": f"{out_name}_full",
                "source_file": str(args.source_dir / f"{name}.hdf"),
                "source_fit_max_residual": fit_error,
                "surface": "sigma_M^T M sigma_M = 1 (sigma_M in Mandel order "
                           "[s11, s22, s33, sqrt2 s23, sqrt2 s13, sqrt2 s12]"
                           + ("; plane-stress block s11, s22, sqrt2 s12)" if stress_format == "plane_stress_2d" else ")"),
                "M_mandel": M_out.tolist(),
                "stress_format": stress_format,
                "columns": ["s11", "s22", "s12"] if stress_format == "plane_stress_2d" else
                           ["s11", "s22", "s33", "sqrt2*s23", "sqrt2*s13", "sqrt2*s12"],
                "sampling": "directions uniform on the unit sphere, scaled onto the surface",
                "samples": args.samples,
                "seed": args.seed,
                "max_abs_surface_error": error,
                "fraction_rows_with_negative": float((stress < 0).any(axis=1).mean()),
            })


if __name__ == "__main__":
    main()
