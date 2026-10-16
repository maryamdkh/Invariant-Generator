"""Merge per-task Feynman benchmark summaries and report recovery across seeds.

Expects <run_dir>/seed<k>/<equation>/benchmark_summary.json (cluster layout)
or <run_dir>/benchmark_summary.json files found recursively. Writes
<run_dir>/merged_summary.json and prints per-arm recovery rates (mean and
standard deviation over seeds) and per-equation recovery counts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

ARMS = ("pipeline", "portfolio", "baseline")


def arm_result(row: dict, arm: str) -> dict:
    result = row.get(arm, {})
    # Without a reduction the pipeline hands the raw problem to PySR.
    if arm == "pipeline" and result.get("reduction") is False:
        return row.get("baseline", {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows_by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.run_dir.rglob("benchmark_summary.json")):
        match = re.search(r"seed(\d+)", str(path.relative_to(args.run_dir)))
        seed = int(match.group(1)) if match else -1
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows_by_seed[seed][row["equation_id"]] = row
    merged = [
        {**row, "seed": seed}
        for seed, rows in sorted(rows_by_seed.items())
        for row in rows.values()
    ]
    (args.run_dir / "merged_summary.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    seeds = sorted(rows_by_seed)
    equations = sorted({row["equation_id"] for row in merged})
    print(f"{len(merged)} rows, seeds {seeds}, {len(equations)} equations\n")
    print("| arm | recovered per seed | mean ± sd |")
    print("|---|---|---|")
    for arm in ARMS:
        counts = []
        for seed in seeds:
            rows = rows_by_seed[seed].values()
            if not any(arm in row for row in rows):
                continue
            counts.append(sum(bool(arm_result(row, arm).get("numerical_recovery")) for row in rows))
        if counts:
            print(f"| {arm} | {counts} | {np.mean(counts):.1f} ± {np.std(counts):.1f} |")
    print("\n| equation | " + " | ".join(ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for equation in equations:
        cells = []
        for arm in ARMS:
            hits = [
                bool(arm_result(rows_by_seed[seed][equation], arm).get("numerical_recovery"))
                for seed in seeds
                if equation in rows_by_seed[seed] and arm in rows_by_seed[seed][equation]
            ]
            cells.append(f"{sum(hits)}/{len(hits)}" if hits else "-")
        print(f"| {equation.removeprefix('feynman_')} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
