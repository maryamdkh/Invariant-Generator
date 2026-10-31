"""Score the SA-A run: per-problem table, including tasks that lost their summary.

A task killed after PySR finished has no benchmark_summary.json, but its
equation_in_original_inputs.txt is on disk. Those are scored here with the
numerical part of recovery_report (same fresh points); nothing is rerun.

Usage: uv run python experiments/probes/sa_a_score.py [results/feynman_benchmark/sa_A/seed42]
Writes experiments/outputs/sa_a_score.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

THRESHOLD = 3.2e-4
DATA = Path("data/synthetic/sa")
OUTPUT = Path("experiments/outputs/sa_a_score.csv")


def fresh_error(root: Path, entry: dict) -> tuple[float, float]:
    """Fresh and test relative L2 from the saved equation (load_known_formula's points)."""
    name = entry["equation_id"]
    stage3 = root / name / name / "stage3_pysr"
    random_state = json.loads((stage3 / "config.json").read_text())["symbolic"]["random_state"]
    names = entry["variables"]
    symbols = {key: sp.Symbol(key) for key in names}
    rng = np.random.default_rng(random_state)
    X = np.column_stack([rng.uniform(*entry["input_ranges"][key], size=10_000) for key in names])
    columns = [X[:, i] for i in range(X.shape[1])]
    ordered = list(symbols.values())
    candidate = sp.sympify((stage3 / "equation_in_original_inputs.txt").read_text(), locals=symbols)
    truth = sp.sympify(entry["formula"], locals=symbols)
    with np.errstate(all="ignore"):
        predicted = np.asarray(sp.lambdify(ordered, candidate, "numpy")(*columns), dtype=np.complex128)
        expected = np.asarray(sp.lambdify(ordered, truth, "numpy")(*columns), dtype=np.float64)
    held = np.load(stage3 / "heldout_predictions.npz")
    test = np.linalg.norm(held["prediction_test"] - held["y_test"]) / np.linalg.norm(held["y_test"])
    return float(np.linalg.norm(predicted - expected) / np.linalg.norm(expected)), float(test)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/feynman_benchmark/sa_A/seed42")
    entries = json.loads((DATA / "manifest.json").read_text())
    rows = []
    for entry in entries:
        name = entry["equation_id"]
        summary = root / name / "benchmark_summary.json"
        stage1 = json.loads((root / name / name / "stage1" / "adaptive_stage1_summary.json").read_text())
        row = {
            "problem": name,
            "d": len(entry["variables"]),
            "form": entry["outer_form"],
            "true_n": entry["n_true_groups"],
            "selected_n": stage1["selected_n"],
            "source": "summary",
            "fresh_rel_l2": None,
            "test_rel_l2": None,
            "pysr_seconds": None,
            "equation": None,
        }
        if summary.exists():
            pipeline = json.loads(summary.read_text())[0]["pipeline"]
            row["fresh_rel_l2"] = pipeline.get("fresh_sample_relative_l2")
            row["test_rel_l2"] = pipeline.get("test_relative_l2")
            row["pysr_seconds"] = pipeline.get("pysr_seconds")
            row["equation"] = pipeline.get("best_equation_in_groups")
            if "error" in pipeline:
                row["source"] = "error: " + pipeline["error"][:80]
        else:
            row["source"] = "rescored (task killed)"
            row["fresh_rel_l2"], row["test_rel_l2"] = fresh_error(root, entry)
            row["equation"] = (root / name / name / "stage3_pysr" / "best_equation.txt").read_text().strip()
        error = row["fresh_rel_l2"]
        row["recovered"] = bool(error is not None and np.isfinite(error) and error < THRESHOLD)
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def report(label: str, group: list[dict]) -> None:
        reduced = [r for r in group if r["selected_n"] is not None]
        errors = [r["fresh_rel_l2"] for r in reduced if r["fresh_rel_l2"] is not None]
        print(
            f"{label:18s} total {len(group):2d}  n chosen {len(reduced):2d}  "
            f"n correct {sum(r['selected_n'] == r['true_n'] for r in reduced):2d}  "
            f"recovered {sum(r['recovered'] for r in group):2d}  "
            f"median error {np.median(errors) if errors else float('nan'):.2e}"
        )

    # The rescoring path must reproduce a stored score.
    check = next(e for e in entries if e["equation_id"] == "sa_d12_03")
    stored = next(r for r in rows if r["problem"] == "sa_d12_03")
    print(f"rescoring check sa_d12_03: stored {stored['fresh_rel_l2']:.6e}, rescored {fresh_error(root, check)[0]:.6e}")
    report("all", rows)
    for form in sorted({r["form"] for r in rows}):
        report(form, [r for r in rows if r["form"] == form])
    for d in sorted({r["d"] for r in rows}):
        report(f"d={d}", [r for r in rows if r["d"] == d])
    for r in rows:
        if r["source"] != "summary":
            print(r["problem"], r["form"], r["source"], r["fresh_rel_l2"])
    print("no n chosen:", [r["problem"] for r in rows if r["selected_n"] is None])
    print("wrong n:", [(r["problem"], r["selected_n"]) for r in rows if r["selected_n"] not in (None, r["true_n"])])
    print("n chosen, not recovered:", [(r["problem"], f"{r['fresh_rel_l2']:.1e}") for r in rows if r["selected_n"] is not None and not r["recovered"]])


if __name__ == "__main__":
    main()
