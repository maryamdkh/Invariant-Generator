"""Score SA-D (sparse linear) and compare with SA-A (ours) and SA-B (PySR alone).

The fresh score stored by the runner is NaN when y^2 = polynomial goes negative at
a fresh point (real sqrt). The test score clips those values to 0 (_inverse_power),
so the fresh score here does the same and reports how many points were affected.
Usage: uv run python experiments/probes/sa_d_compare.py
Writes experiments/outputs/sa_d_score.csv (needs sa_a_score.csv, sa_b_score.csv).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import sympy as sp

THRESHOLD = 3.2e-4
SUMMARY = Path("results/feynman_benchmark/sa_D/seed42/benchmark_summary.json")
OUTPUT = Path("experiments/outputs/sa_d_score.csv")


def number(value) -> float:
    return float("nan") if value in (None, "") else float(value)


def fresh_score(entry: dict, row: dict, random_state: int) -> tuple[float, int]:
    names = entry["variables"]
    symbols = {key: sp.Symbol(key) for key in names}
    rng = np.random.default_rng(random_state)
    X = np.column_stack([rng.uniform(*entry["input_ranges"][key], size=10_000) for key in names])
    columns = [X[:, i] for i in range(X.shape[1])]
    expression = sp.sympify(row["expanded_equation"], locals=symbols)
    truth = sp.sympify(entry["formula"], locals=symbols)
    ordered = list(symbols.values())
    inner = expression.args[0] if row["power"] == 2 else expression
    values = np.broadcast_to(np.asarray(sp.lambdify(ordered, inner, "numpy")(*columns), dtype=float), (len(X),))
    negative = int(np.sum(values < 0)) if row["power"] == 2 else 0
    prediction = np.sqrt(np.clip(values, 0, None)) if row["power"] == 2 else values
    expected = np.asarray(sp.lambdify(ordered, truth, "numpy")(*columns), dtype=float)
    return float(np.linalg.norm(prediction - expected) / np.linalg.norm(expected)), negative


def main() -> None:
    entries = {e["equation_id"]: e for e in json.loads(Path("data/synthetic/sa/manifest.json").read_text())}
    ours = {r["problem"]: r for r in csv.DictReader(open("experiments/outputs/sa_a_score.csv"))}
    pysr = {r["problem"]: r for r in csv.DictReader(open("experiments/outputs/sa_b_score.csv"))}
    random_state = 42
    rows = []
    for item in json.loads(SUMMARY.read_text()):
        name = item["equation_id"]
        d = item["sparse_linear"]
        error, negative = d.get("fresh_sample_relative_l2"), 0
        rescored = error is None or not np.isfinite(error)
        if rescored and d.get("power") in (1, 2):
            error, negative = fresh_score(entries[name], d, random_state)
        if d.get("power") == 3 and rescored:
            print("power 3 with no finite score:", name)
        rows.append({
            "problem": name, "d": len(entries[name]["variables"]), "form": entries[name]["outer_form"],
            "power": d.get("power"), "active_terms": d.get("active_terms"), "library_size": d.get("library_size"),
            "fresh_rel_l2": error, "stored_fresh": d.get("fresh_sample_relative_l2"), "negative_points": negative,
            "test_rel_l2": d.get("test_relative_l2"), "fit_seconds": d.get("fit_seconds"),
            "symbolic_recovery": d.get("symbolic_recovery"),
            "recovered": bool(error is not None and np.isfinite(error) and error < THRESHOLD),
            "equation": d.get("expanded_equation"), "error": d.get("error"),
        })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("errors:", [r["problem"] for r in rows if r["error"]])
    print("rescored:", [(r["problem"], r["negative_points"], f"{r['fresh_rel_l2']:.1e}") for r in rows if not np.isfinite(number(r["stored_fresh"]))])
    print("symbolic true:", sum(r["symbolic_recovery"] is True for r in rows))
    print("powers:", {p: sum(r["power"] == p for r in rows) for p in (1, 2, 3)})

    def table(label, group):
        ea = [number(ours[r["problem"]]["fresh_rel_l2"]) for r in group]
        eb = [number(pysr[r["problem"]]["fresh_rel_l2"]) for r in group]
        ed = [number(r["fresh_rel_l2"]) for r in group]
        both = [(a, x) for a, x in zip(ea, ed) if np.isfinite(a) and np.isfinite(x)]
        ratio = np.median([x / a for a, x in both]) if both else float("nan")
        print(f"{label:16s} {len(group):2d} | under A {sum(ours[r['problem']]['recovered']=='True' for r in group):2d} "
              f"B {sum(pysr[r['problem']]['recovered']=='True' for r in group):2d} D {sum(r['recovered'] for r in group):2d} "
              f"| median A {np.nanmedian(ea) if np.isfinite(ea).any() else float('nan'):.1e} B {np.nanmedian(eb):.1e} D {np.nanmedian(ed):.1e} "
              f"| paired {len(both):2d} A better {sum(a<x for a,x in both):2d} median D/A {ratio:.1f}")

    table("all", rows)
    for form in sorted({r["form"] for r in rows}):
        table(form, [r for r in rows if r["form"] == form])
    for d in sorted({r["d"] for r in rows}):
        table(f"d={d}", [r for r in rows if r["d"] == d])
    for form in sorted({r["form"] for r in rows}):
        for d in sorted({r["d"] for r in rows}):
            table(f"{form[:7]} d{d}", [r for r in rows if r["form"] == form and r["d"] == d])
    print("D beats A on non-sqrt(q1):", [(r["problem"], f"A {number(ours[r['problem']]['fresh_rel_l2']):.1e} D {r['fresh_rel_l2']:.1e}") for r in rows if r["form"] != "sqrt(q1)" and number(r["fresh_rel_l2"]) <= number(ours[r["problem"]]["fresh_rel_l2"])])
    best = lambda r: min(x for x in (number(ours[r["problem"]]["fresh_rel_l2"]), number(pysr[r["problem"]]["fresh_rel_l2"])) if np.isfinite(x))
    print("A no n, D under:", sum(1 for r in rows if ours[r["problem"]]["selected_n"] == "" and r["recovered"]), "A no n, D or B under:", sum(1 for r in rows if ours[r["problem"]]["selected_n"] == "" and (r["recovered"] or pysr[r["problem"]]["recovered"] == "True")))
    print("D fit seconds median", np.median([r["fit_seconds"] for r in rows]), "max", max(r["fit_seconds"] for r in rows))


if __name__ == "__main__":
    main()
