"""Compare SA-E (PySR on the fixed term list, no encoder) with SA-A, SA-B and SA-D.

Usage: uv run python experiments/probes/sa_e_compare.py
Writes experiments/outputs/sa_e_score.csv (needs sa_a, sa_b, sa_d score files).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

THRESHOLD = 3.2e-4
ROOT = Path("results/feynman_benchmark/sa_E/seed42")
OUTPUT = Path("experiments/outputs/sa_e_score.csv")
COMMIT = "9030a1b"


def number(value) -> float:
    return float("nan") if value in (None, "") else float(value)


def main() -> None:
    load = lambda name: {r["problem"]: r for r in csv.DictReader(open(f"experiments/outputs/{name}_score.csv"))}
    ours, pysr, sparse = load("sa_a"), load("sa_b"), load("sa_d")
    rows = []
    for entry in json.loads(Path("data/synthetic/sa/manifest.json").read_text()):
        name = entry["equation_id"]
        summary = ROOT / name / "benchmark_summary.json"
        e = json.loads(summary.read_text())[0]["library_pysr"] if summary.exists() else {"error": "missing summary"}
        error = e.get("fresh_sample_relative_l2")
        rows.append({
            "problem": name, "d": len(entry["variables"]), "form": entry["outer_form"],
            "library_size": e.get("library_size"), "fresh_rel_l2": error,
            "test_rel_l2": e.get("test_relative_l2"), "pysr_seconds": e.get("pysr_seconds"),
            "symbolic_recovery": e.get("symbolic_recovery"),
            "recovered": bool(error is not None and np.isfinite(error) and error < THRESHOLD),
            "equation": e.get("best_equation_in_features"), "error": e.get("error"),
        })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run = Path("experiments/runs/SA-E/summary.csv")
    columns = open("experiments/runs/SA-D/summary.csv").readline().strip().split(",")
    with run.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "experiment_id": "SA-E", "dataset": "SA", "instance": r["problem"], "method": "E", "seed": 42,
                "commit": COMMIT, "recovered": r["recovered"], "test_rel_l2": r["test_rel_l2"],
                "equation": r["equation"], "time_sr_s": r["pysr_seconds"],
                "notes": f"fresh {r['fresh_rel_l2']:.2e}; {r['library_size']} terms",
            })

    print("errors:", [(r["problem"], str(r["error"])[:80]) for r in rows if r["error"]])
    print("non-finite:", [r["problem"] for r in rows if not np.isfinite(number(r["fresh_rel_l2"]))])
    print("symbolic true:", sum(r["symbolic_recovery"] is True for r in rows))
    print("library size by d:", sorted({(r["d"], r["library_size"]) for r in rows}))

    def table(label, group):
        ea = [number(ours[r["problem"]]["fresh_rel_l2"]) for r in group]
        eb = [number(pysr[r["problem"]]["fresh_rel_l2"]) for r in group]
        ee = [number(r["fresh_rel_l2"]) for r in group]
        both = [(a, x) for a, x in zip(ea, ee) if np.isfinite(a) and np.isfinite(x)]
        pair_b = [(b, x) for b, x in zip(eb, ee) if np.isfinite(b) and np.isfinite(x)]
        ratio = np.median([x / a for a, x in both]) if both else float("nan")
        print(f"{label:16s} {len(group):2d} | under A {sum(ours[r['problem']]['recovered']=='True' for r in group):2d} "
              f"B {sum(pysr[r['problem']]['recovered']=='True' for r in group):2d} "
              f"D {sum(sparse[r['problem']]['recovered']=='True' for r in group):2d} E {sum(r['recovered'] for r in group):2d} "
              f"| median A {np.nanmedian(ea) if np.isfinite(ea).any() else float('nan'):.1e} B {np.nanmedian(eb):.1e} E {np.nanmedian(ee):.1e} "
              f"| A better {sum(a<x for a,x in both):2d}/{len(both):2d} median E/A {ratio:.1f} "
              f"| E better than B {sum(x<b for b,x in pair_b):2d}/{len(pair_b):2d}")

    table("all", rows)
    for form in sorted({r["form"] for r in rows}):
        table(form, [r for r in rows if r["form"] == form])
    for d in sorted({r["d"] for r in rows}):
        table(f"d={d}", [r for r in rows if r["d"] == d])
    print("E under:", [(r["problem"], r["form"], f"{r['fresh_rel_l2']:.1e}", "A n=" + ours[r["problem"]]["selected_n"]) for r in rows if r["recovered"]])
    print("E beats A:", [(r["problem"], f"A {number(ours[r['problem']]['fresh_rel_l2']):.1e} E {r['fresh_rel_l2']:.1e}") for r in rows if number(r["fresh_rel_l2"]) <= number(ours[r["problem"]]["fresh_rel_l2"])])
    no_n = [r for r in rows if ours[r["problem"]]["selected_n"] == ""]
    print("A no n:", len(no_n), "E under there:", sum(r["recovered"] for r in no_n),
          "B/D/E under there:", sum(r["recovered"] or pysr[r["problem"]]["recovered"] == "True" or sparse[r["problem"]]["recovered"] == "True" for r in no_n))
    print("A or E under:", sum(r["recovered"] or ours[r["problem"]]["recovered"] == "True" for r in rows))
    t = [number(r["pysr_seconds"]) / 60 for r in rows]
    print(f"PySR minutes median {np.nanmedian(t):.1f} max {np.nanmax(t):.1f}")
    for name in ("sa_d04_00", "sa_d12_00", "sa_d12_01", "sa_d12_02"):
        r = next(x for x in rows if x["problem"] == name)
        print(name, r["form"], f"{r['fresh_rel_l2']:.1e}", r["equation"])


if __name__ == "__main__":
    main()
