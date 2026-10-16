"""Print a Markdown table from a Feynman benchmark_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arm(row: dict, arm: str) -> dict:
    result = row.get(arm, {})
    # Without a reduction the pipeline hands the problem to PySR unchanged.
    if arm == "pipeline" and result.get("reduction") is False:
        return {**row.get("baseline", {}), "fallback": True}
    return result


def _cell(result: dict) -> str:
    if not result:
        return "not run"
    if "error" in result:
        return "error"
    recovered = result.get("numerical_recovery")
    error = result.get("fresh_sample_relative_l2")
    mark = "**yes**" if recovered else "no"
    if error is not None:
        mark += f" ({error:.1e})"
    if result.get("fallback"):
        mark += " [fallback]"
    return mark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    print("| equation | inputs | n* (analysis) | Stage 1 n | groups (exponents) | pipeline recovered | PySR alone recovered |")
    print("|---|---|---|---|---|---|---|")
    totals = {"pipeline": 0, "baseline": 0}
    for row in rows:
        pipeline = row.get("pipeline", {})
        exponents = pipeline.get("exponents")
        groups = "; ".join("(" + ", ".join(r) + ")" for r in exponents) if exponents else (
            "dense" if pipeline.get("reduction") else "-"
        )
        for arm in totals:
            totals[arm] += bool(_arm(row, arm).get("numerical_recovery"))
        print(
            f"| {row['equation_id'].removeprefix('feynman_')} | {row['n_inputs']} "
            f"| {row.get('log_reducibility_analysis')} | {pipeline.get('selected_n', '-')} "
            f"| {groups} | {_cell(_arm(row, 'pipeline'))} | {_cell(_arm(row, 'baseline'))} |"
        )
    print(f"\nRecovered (relative error < 1e-6 on fresh points): pipeline {totals['pipeline']}/{len(rows)}, "
          f"PySR alone {totals['baseline']}/{len(rows)}")


if __name__ == "__main__":
    main()
