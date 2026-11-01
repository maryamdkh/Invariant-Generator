"""Compare SA-B (PySR alone) with SA-A (ours) on the 70 SA problems.

A task killed during scoring has no benchmark_summary.json; its selected
equation is rescored from baseline_pysr/equations.csv if present. Usage:
uv run python experiments/probes/sa_b_compare.py
Writes experiments/outputs/sa_b_score.csv (needs sa_a_score.csv).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

THRESHOLD = 3.2e-4
ROOT = Path("results/feynman_benchmark/sa_B/seed42")
OUTPUT = Path("experiments/outputs/sa_b_score.csv")


def number(value: str | float | None) -> float:
    return float("nan") if value in (None, "") else float(value)


def main() -> None:
    ours = {r["problem"]: r for r in csv.DictReader(open("experiments/outputs/sa_a_score.csv"))}
    rows = []
    for entry in json.loads(Path("data/synthetic/sa/manifest.json").read_text()):
        name = entry["equation_id"]
        summary = ROOT / name / "benchmark_summary.json"
        row = {"problem": name, "d": len(entry["variables"]), "form": entry["outer_form"], "source": "summary",
               "fresh_rel_l2": None, "test_rel_l2": None, "pysr_seconds": None, "equation": None,
               "symbolic_recovery": None}
        if summary.exists():
            b = json.loads(summary.read_text())[0]["baseline"]
            if "error" in b:
                row["source"] = "error: " + b["error"][:80]
            row["fresh_rel_l2"] = b.get("fresh_sample_relative_l2")
            row["test_rel_l2"] = b.get("test_relative_l2")
            row["pysr_seconds"] = b.get("pysr_seconds")
            row["equation"] = b.get("best_equation_raw_names")
            row["symbolic_recovery"] = b.get("symbolic_recovery")
        else:
            row["source"] = "missing summary"
        error = row["fresh_rel_l2"]
        row["recovered"] = bool(error is not None and np.isfinite(error) and error < THRESHOLD)
        rows.append(row)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("not clean:", [(r["problem"], r["source"]) for r in rows if r["source"] != "summary"])
    print("non-finite:", [r["problem"] for r in rows if not np.isfinite(number(r["fresh_rel_l2"]))])
    print("symbolic:", {str(k): sum(str(r["symbolic_recovery"]) == str(k) for r in rows) for k in (True, False, None)})

    def table(label, group):
        a = [ours[r["problem"]] for r in group]
        ea = [number(x["fresh_rel_l2"]) for x in a]
        eb = [number(r["fresh_rel_l2"]) for r in group]
        both = [(x, y) for x, y in zip(ea, eb) if np.isfinite(x) and np.isfinite(y)]
        ratio = np.median([y / x for x, y in both]) if both else float("nan")
        print(f"{label:16s} total {len(group):2d} | A under {sum(x['recovered']=='True' for x in a):2d} B under {sum(r['recovered'] for r in group):2d}"
              f" | median A {np.nanmedian(ea) if np.isfinite(ea).any() else float('nan'):.1e} B {np.nanmedian(eb):.1e}"
              f" | paired {len(both):2d}, A better {sum(x<y for x,y in both):2d}, median B/A {ratio:.1f}")

    table("all", rows)
    for form in sorted({r["form"] for r in rows}):
        table(form, [r for r in rows if r["form"] == form])
    for d in sorted({r["d"] for r in rows}):
        table(f"d={d}", [r for r in rows if r["d"] == d])
    for form in sorted({r["form"] for r in rows}):
        for d in sorted({r["d"] for r in rows}):
            table(f"{form[:6]} d{d}", [r for r in rows if r["form"] == form and r["d"] == d])
    print("A has no n; B there:")
    for r in rows:
        if ours[r["problem"]]["selected_n"] == "":
            print(" ", r["problem"], r["form"], f"{number(r['fresh_rel_l2']):.1e}")
    print("B beats A (n chosen):")
    for r in rows:
        x = number(ours[r["problem"]]["fresh_rel_l2"])
        if np.isfinite(x) and number(r["fresh_rel_l2"]) <= x:
            print(" ", r["problem"], r["form"], f"A {x:.1e} B {number(r['fresh_rel_l2']):.1e}")
    t = [number(r["pysr_seconds"]) / 60 for r in rows]
    print(f"PySR minutes median {np.nanmedian(t):.1f} max {np.nanmax(t):.1f}")


if __name__ == "__main__":
    main()
