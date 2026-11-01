"""Compare SA-C (distillation) with SA-A, SA-B, SA-D and SA-E.

Usage: uv run python experiments/probes/sa_c_compare.py
Writes experiments/outputs/sa_c_score.csv and experiments/runs/SA-C/summary.csv
(needs the sa_a, sa_b, sa_d, sa_e score files).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

THRESHOLD = 3.2e-4
ROOT = Path("results/feynman_benchmark/sa_C/seed42")
OUTPUT = Path("experiments/outputs/sa_c_score.csv")
COMMIT = "1e8ef7c"


def number(value) -> float:
    return float("nan") if value in (None, "") else float(value)


def main() -> None:
    load = lambda name: {r["problem"]: r for r in csv.DictReader(open(f"experiments/outputs/{name}_score.csv"))}
    ours, pysr, sparse, fixed = load("sa_a"), load("sa_b"), load("sa_d"), load("sa_e")
    rows = []
    for entry in json.loads(Path("data/synthetic/sa/manifest.json").read_text()):
        name = entry["equation_id"]
        summary = ROOT / name / "benchmark_summary.json"
        c = json.loads(summary.read_text())[0]["distillation"] if summary.exists() else {"error": "missing summary"}
        error = c.get("fresh_sample_relative_l2")
        rows.append({
            "problem": name, "d": len(entry["variables"]), "form": entry["outer_form"],
            "true_n": entry["n_true_groups"], "kept": len(c.get("kept", [])) if "kept" in c else None,
            "network_nrmse": c.get("train_nrmse"), "fresh_rel_l2": error,
            "test_rel_l2": c.get("test_relative_l2"), "network_seconds": c.get("network_seconds"),
            "pysr_seconds": c.get("pysr_seconds"), "symbolic_recovery": c.get("symbolic_recovery"),
            "recovered": bool(error is not None and np.isfinite(error) and error < THRESHOLD),
            "outer": c.get("outer_equation"), "latents": json.dumps(c.get("latent_equations")), "error": c.get("error"),
        })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = open("experiments/runs/SA-D/summary.csv").readline().strip().split(",")
    with Path("experiments/runs/SA-C/summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "experiment_id": "SA-C", "dataset": "SA", "instance": r["problem"], "method": "C", "seed": 42,
                "commit": COMMIT, "recovered": r["recovered"], "test_rel_l2": r["test_rel_l2"],
                "equation": r["outer"], "n_selected": r["kept"], "n_true": r["true_n"],
                "time_encoder_s": r["network_seconds"], "time_sr_s": r["pysr_seconds"],
                "notes": f"fresh {number(r['fresh_rel_l2']):.2e}; network nrmse {number(r['network_nrmse']):.2e}",
            })

    print("errors:", [(r["problem"], str(r["error"])[:100]) for r in rows if r["error"]])
    print("non-finite:", [r["problem"] for r in rows if not np.isfinite(number(r["fresh_rel_l2"]))])
    print("symbolic true:", sum(r["symbolic_recovery"] is True for r in rows))

    def table(label, group):
        ea = [number(ours[r["problem"]]["fresh_rel_l2"]) for r in group]
        eb = [number(pysr[r["problem"]]["fresh_rel_l2"]) for r in group]
        ec = [number(r["fresh_rel_l2"]) for r in group]
        both = [(a, x) for a, x in zip(ea, ec) if np.isfinite(a) and np.isfinite(x)]
        pair_b = [(b, x) for b, x in zip(eb, ec) if np.isfinite(b) and np.isfinite(x)]
        ratio = np.median([x / a for a, x in both]) if both else float("nan")
        count = lambda scores: sum(scores[r["problem"]]["recovered"] == "True" for r in group)
        print(f"{label:16s} {len(group):2d} | under A {count(ours):2d} B {count(pysr):2d} C {sum(r['recovered'] for r in group):2d} D {count(sparse):2d} E {count(fixed):2d} "
              f"| median A {np.nanmedian(ea) if np.isfinite(ea).any() else float('nan'):.1e} C {np.nanmedian(ec):.1e} net {np.nanmedian([number(r['network_nrmse']) for r in group]):.1e} "
              f"| A better {sum(a<x for a,x in both):2d}/{len(both):2d} median C/A {ratio:.1f} "
              f"| C better than B {sum(x<b for b,x in pair_b):2d}/{len(pair_b):2d}")

    table("all", rows)
    for form in sorted({r["form"] for r in rows}):
        table(form, [r for r in rows if r["form"] == form])
    for d in sorted({r["d"] for r in rows}):
        table(f"d={d}", [r for r in rows if r["d"] == d])
    print("C under:", [(r["problem"], r["form"], f"{r['fresh_rel_l2']:.1e}", "A n=" + ours[r["problem"]]["selected_n"]) for r in rows if r["recovered"]])
    print("C beats A:", [(r["problem"], f"A {number(ours[r['problem']]['fresh_rel_l2']):.1e} C {r['fresh_rel_l2']:.1e}") for r in rows if number(r["fresh_rel_l2"]) <= number(ours[r["problem"]]["fresh_rel_l2"])])
    for true_n in sorted({r["true_n"] for r in rows}):
        group = [r for r in rows if r["true_n"] == true_n]
        print(f"true n {true_n}: kept counts", {k: sum(r["kept"] == k for r in group) for k in sorted({r["kept"] for r in group if r["kept"] is not None})},
              "correct", sum(r["kept"] == true_n for r in group), "/", len(group))
    print("kept correct by d:", {d: sum(r["kept"] == r["true_n"] for r in rows if r["d"] == d) for d in sorted({r["d"] for r in rows})})
    composed = [number(r["fresh_rel_l2"]) / number(r["network_nrmse"]) for r in rows]
    print(f"composed error / network nrmse: median {np.nanmedian(composed):.1f}; network better on {sum(x > 1 for x in composed)}/70")
    print("A or C under:", sum(r["recovered"] or ours[r["problem"]]["recovered"] == "True" for r in rows))
    print(f"network s median {np.nanmedian([number(r['network_seconds']) for r in rows]):.0f}; PySR min median {np.nanmedian([number(r['pysr_seconds']) for r in rows]) / 60:.1f}")
    for name in ("sa_d03_00", "sa_d06_00", "sa_d06_09", "sa_d12_09"):
        r = next(x for x in rows if x["problem"] == name)
        print(name, r["form"], f"{r['fresh_rel_l2']:.1e}", r["outer"], r["latents"][:220])


if __name__ == "__main__":
    main()
