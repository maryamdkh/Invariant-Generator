"""Generate noiseless Feynman benchmark data from PMLB metadata files.

Formulas and ranges come from the Feynman Symbolic Regression Database as
stored in PMLB (`data/feynman_tables/pmlb_metadata/*.yaml`). Samples are drawn
uniformly in the published ranges. `--verify` checks each parsed formula
against PMLB's official data table for that problem.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import urllib.request

import h5py
import numpy as np

from invariant_generator.feynman import (
    log_reducibility,
    parse_pmlb_metadata,
    select_pilot_problems,
)

# Data tables are stored with Git LFS, so raw.githubusercontent.com only serves pointers.
PMLB_DATA = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"


def verify_against_pmlb(problem, cache_dir: Path, rows: int = 2000) -> float:
    """Max relative error of the parsed formula on PMLB's official rows."""
    path = cache_dir / f"{problem.name}.tsv.gz"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PMLB_DATA.format(name=problem.name), path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().split()
        data = np.loadtxt(handle, max_rows=rows)
    columns = [header.index(name) for name in problem.variables]
    target = data[:, header.index("target")]
    predicted = problem.evaluate(data[:, columns])
    return float(np.max(np.abs(predicted - target) / np.maximum(np.abs(target), 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/feynman_tables/pmlb_metadata"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/feynman"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    problems = [parse_pmlb_metadata(path) for path in sorted(args.metadata_dir.glob("*.yaml"))]
    if args.selection == "pilot":
        problems = select_pilot_problems(problems)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for problem in problems:
        X, y = problem.sample(args.samples, seed=args.seed)
        path = args.output_dir / f"{problem.name}.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("X", data=X)
            file.create_dataset("y", data=y)
        entry = {
            "equation_id": problem.name,
            "target": problem.target,
            "formula": str(problem.expression()),
            "source_formula": problem.formula,
            "variables": problem.variables,
            "input_ranges": {name: list(problem.ranges[name]) for name in problem.variables},
            "log_reducibility": log_reducibility(problem),
            "samples": args.samples,
            "seed": args.seed,
            "source": "PMLB metadata of the Feynman Symbolic Regression Database (AI Feynman)",
            "generated_data": True,
        }
        if args.verify:
            entry["pmlb_max_relative_error"] = verify_against_pmlb(
                problem, Path("data/feynman_tables/pmlb_data"),
            )
        path.with_suffix(".json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        manifest.append(entry)
        print(
            f"{problem.name:20s} d={len(problem.variables)} n*={entry['log_reducibility']}"
            + (f" pmlb_err={entry['pmlb_max_relative_error']:.1e}" if args.verify else "")
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
