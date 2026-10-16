"""Generate yield-surface datasets from published anisotropic criteria.

Writes `data/yield_benchmarks/<criterion>.hdf` (key "stress", Mandel order
[s11, s22, s33, sqrt2*s23, sqrt2*s13, sqrt2*s12], as for data/rotatedhill.hdf)
with points on the surface sigma_bar = 1, plus a JSON with the criterion,
coefficients and source. Directions are uniform on the 6-D unit sphere.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from invariant_generator.yield_criteria import CRITERIA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", default=",".join(sorted(CRITERIA)))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data/yield_benchmarks"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.criteria.split(","):
        criterion = CRITERIA[name]
        rng = np.random.default_rng(args.seed)
        directions = rng.normal(size=(args.samples, 6))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        surface = directions / criterion.equivalent_stress(directions)[:, None]
        error = np.abs(criterion.equivalent_stress(surface) - 1.0).max()
        if not np.isfinite(surface).all() or error > 1e-10:
            raise RuntimeError(f"{name}: invalid surface points (max |sigma_bar - 1| = {error}).")
        mandel = surface.copy()
        mandel[:, 3:] *= np.sqrt(2.0)
        path = args.output_dir / f"{name}.hdf"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("stress", data=mandel)
        path.with_suffix(".json").write_text(json.dumps({
            "criterion": name,
            "class": type(criterion).__name__,
            "parameters": {k: v for k, v in asdict(criterion).items() if k != "source"},
            "source": criterion.source,
            "samples": args.samples,
            "seed": args.seed,
            "stress_format": "mandel_3d",
            "max_abs_surface_error": float(error),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"{name:16s} -> {path} (max |sigma_bar - 1| = {error:.1e})")


if __name__ == "__main__":
    main()
