"""Common per-run result format for the paper experiments (EXPERIMENTS.md).

Every experiment folder experiments/runs/<ID>/ holds a summary.csv with
exactly these columns, one row per (instance, seed), so one script can build
all tables from all experiments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable

COLUMNS = [
    "experiment_id", "dataset", "instance", "method", "seed", "commit", "recovered",
    "test_rel_l2", "test_r2", "equation", "complexity", "n_selected", "n_true",
    "structure_score", "time_encoder_s", "time_sr_s", "notes",
]


def write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows in the fixed column order; unknown keys are an error."""
    rows = list(rows)
    for row in rows:
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Unknown summary columns: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else row[column] for column in COLUMNS})
    return len(rows)


def read_summaries(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(runs_dir.glob("*/summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def benchmark_rows(
    run_dir: Path, *, arm: str, experiment_id: str, dataset: str, method: str, commit: str,
) -> list[dict[str, Any]]:
    """Rows from scripts/run_feynman_benchmark.py output (benchmark_summary.json
    files, seed taken from a seed<k> path component) for one arm.

    As in scripts/merge_feynman_benchmark.py, a pipeline row without a
    reduction reports the PySR-alone result. With target noise, exact numerical
    recovery is impossible, so `recovered` is the symbolic match.
    """
    rows = []
    for path in sorted(run_dir.rglob("benchmark_summary.json")):
        match = re.search(r"seed(\d+)", str(path.relative_to(run_dir)))
        seed = int(match.group(1)) if match else None
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if arm not in entry:
                continue
            result = entry[arm]
            notes = []
            if arm == "pipeline" and result.get("reduction") is False:
                notes.append("no reduction: PySR-alone result"
                             + (f" ({result['skipped']})" if result.get("skipped") else ""))
                encoder_seconds = result.get("stage1_seconds")
                result = {**entry.get("baseline", {}), "selected_n": None}
            else:
                encoder_seconds = None
                if result.get("stage1_seconds") is not None:
                    encoder_seconds = result["stage1_seconds"] + (result.get("stage2_seconds") or 0.0)
            if "error" in result:
                notes.append(f"error: {result['error'][:200]}")
            noisy = float(entry.get("target_noise") or 0.0) > 0
            recovered = result.get("symbolic_recovery" if noisy else "numerical_recovery")
            rows.append({
                "experiment_id": experiment_id,
                "dataset": dataset,
                "instance": entry["equation_id"],
                "method": method,
                "seed": seed,
                "commit": commit,
                "recovered": None if recovered is None else bool(recovered),
                "test_rel_l2": result.get("test_relative_l2"),
                "equation": result.get("expanded_equation"),
                "n_selected": result.get("selected_n"),
                "n_true": entry.get("log_reducibility_analysis"),
                "time_encoder_s": encoder_seconds,
                "time_sr_s": result.get("pysr_seconds"),
                "notes": "; ".join(notes),
            })
    return rows
