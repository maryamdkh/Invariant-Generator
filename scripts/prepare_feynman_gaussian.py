"""Generate a reproducible Feynman I.6.20a input-expansion pilot.

These are newly sampled noiseless data from the published equation, not the
original numerical table. No result is inferred from the known formula.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data/feynman_gaussian"))
    args = parser.parse_args()
    if args.samples < 30:
        raise ValueError("At least 30 samples are required.")
    rng = np.random.default_rng(args.seed)
    a = rng.uniform(0.25, 1.25, size=args.samples)
    b = rng.uniform(0.25, 1.25, size=args.samples)
    for variant in ("original", "sum", "product"):
        theta = a + b if variant != "product" else a * b
        X = theta[:, None] if variant == "original" else np.column_stack((a, b))
        y = np.exp(-theta**2 / 2.0) / np.sqrt(2.0 * np.pi)
        path = args.output_dir / f"i_6_20a_{variant}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as file:
            file.create_dataset("X", data=X)
            file.create_dataset("y", data=y)
        (path.with_suffix(".json")).write_text(
            json.dumps({
                "equation_id": "I.6.20a",
                "formula": "exp(-theta**2/2)/sqrt(2*pi)",
                "substitution": {
                    "original": "theta",
                    "sum": "theta=a+b",
                    "product": "theta=a*b",
                }[variant],
                "variant": variant,
                "samples": args.samples,
                "seed": args.seed,
                "input_ranges": {"a": [0.25, 1.25], "b": [0.25, 1.25]},
                "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/",
                "generated_data": True,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
