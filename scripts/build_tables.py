"""Build result tables from every experiments/runs/<ID>/summary.csv.

    uv run python scripts/build_tables.py            # print markdown
    uv run python scripts/build_tables.py --output experiments/outputs/tables.md

Table 1: per experiment, recovered count per seed and mean +- sd, median
test relative L2 and median structure score (F').
Table 2: dataset x method, mean recovered count (a cell per experiment).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from invariant_generator.experiment_summary import read_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = read_summaries(args.runs_dir)
    if not rows:
        raise SystemExit(f"No summary.csv files under {args.runs_dir}.")

    per_experiment: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    errors: dict[str, list[float]] = defaultdict(list)
    structure: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for row in rows:
        hits = per_experiment[row["experiment_id"]][row["seed"]]
        if row["recovered"]:  # empty for rows without a recovery criterion (e.g. F')
            hits.append(row["recovered"] == "True")
        meta[row["experiment_id"]] = (row["dataset"], row["method"])
        if row["test_rel_l2"]:
            errors[row["experiment_id"]].append(float(row["test_rel_l2"]))
        if row["structure_score"]:
            structure[row["experiment_id"]].append(float(row["structure_score"]))

    lines = ["| experiment | dataset | method | instances | recovered per seed | mean +- sd | median test rel. L2 | median structure |",
             "|---|---|---|---|---|---|---|---|"]
    cell: dict[tuple[str, str], str] = {}
    for experiment_id in sorted(per_experiment):
        seeds = per_experiment[experiment_id]
        counts = [sum(seeds[seed]) for seed in sorted(seeds)]
        instances = max(len(hits) for hits in seeds.values())
        dataset, method = meta[experiment_id]
        summary = f"{np.mean(counts):.1f} +- {np.std(counts):.1f}" if instances else "-"
        if not instances:
            counts = "-"
        error = f"{np.median(errors[experiment_id]):.2e}" if errors[experiment_id] else "-"
        score = f"{np.median(structure[experiment_id]):.10f}" if structure[experiment_id] else "-"
        lines.append(f"| {experiment_id} | {dataset} | {method} | {instances} | {counts} | {summary} | {error} | {score} |")
        cell[(dataset, method)] = f"{np.mean(counts):.1f}/{instances}" if instances else "-"

    datasets = sorted({d for d, _ in cell})
    methods = sorted({m for _, m in cell})
    lines += ["", "| dataset | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for dataset in datasets:
        lines.append(f"| {dataset} | " + " | ".join(cell.get((dataset, m), "-") for m in methods) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
