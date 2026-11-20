"""Score a redundant-Feynman encoder run against PySR alone (same inputs) and FEY-B.

For every equation it reports recovery (numerical: fresh relative L2 < 1e-6),
the selected bottleneck n vs the number of original variables d, and how the
snapped encoder rows relate to the true sums stored in the dataset JSON:

    pure   every row is proportional to one true sum (e.g. u3 + u8)
    span   every row lies in the span of the true sums (constant weight on
           the parts of each original variable, zero on distractors), but
           some row mixes variables
    other  at least one row is not in that span (split parts weighted
           differently, or distractors used), or the rows were not snapped
    cover  the rows span every original variable y actually needs is not
           checked here; n vs d is reported instead

    uv run python experiments/probes/fey_redundant_score.py \
        --run fey_l1_a --baseline fey_l1_b --level L1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

RESULTS = Path("results/feynman_benchmark")
DATA = Path("data/feynman_redundant")


def load(run: str) -> dict[str, dict]:
    out = {}
    for path in sorted((RESULTS / run / "seed42").glob("*/benchmark_summary.json")):
        entry = json.loads(path.read_text())[0]
        out[entry["equation_id"]] = entry
    return out


def arm(entry: dict) -> dict:
    return entry.get("pipeline") or entry.get("baseline") or {}


def row_class(rows: list[list[str]], meta: dict) -> str:
    names = meta["variables"]
    groups = [[names.index(u.strip()) for u in s.split("+")] for s in meta["mapping"].values()]
    true = np.zeros((len(groups), len(names)))
    for g, cols in enumerate(groups):
        true[g, cols] = 1.0
    R = np.array([[float(Fraction(c)) for c in row] for row in rows])
    in_span, pure = True, True
    for r in R:
        coef, *_ = np.linalg.lstsq(true.T, r, rcond=None)
        if np.linalg.norm(true.T @ coef - r) > 1e-9 * max(1.0, np.linalg.norm(r)):
            in_span = False
        if np.count_nonzero(np.abs(coef) > 1e-9) != 1:
            pure = False
    return "pure" if (in_span and pure) else "span" if in_span else "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="fey_l1_a")
    p.add_argument("--baseline", default="fey_l1_b")
    p.add_argument("--original", default="feyb_paper")
    p.add_argument("--level", default="L1")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    ours, base, orig = load(a.run), load(a.baseline), load(a.original)
    rows = []
    for eq in sorted(set(ours) | set(base)):
        meta = json.loads((DATA / a.level / f"{eq}.json").read_text())
        o = arm(ours.get(eq, {}))
        d = len(meta["original_variables"])
        cls = row_class(o["exponents"], meta) if o.get("snapped") and o.get("exponents") else "other"
        rows.append({
            "eq": eq, "d": d, "inputs": len(meta["variables"]),
            "n": o.get("selected_n"), "rows": cls if o else "missing",
            "ours": bool(o.get("numerical_recovery")),
            "ours_l2": o.get("fresh_sample_relative_l2"),
            "ours_pysr_s": o.get("pysr_seconds"),
            "ours_stage12_s": (o.get("stage1_seconds") or 0) + (o.get("stage2_seconds") or 0),
            "base": bool(arm(base.get(eq, {})).get("numerical_recovery")),
            "orig": bool(arm(orig.get(eq, {})).get("numerical_recovery")),
        })

    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    print(f"equations {n} (ours present {sum(r['rows'] != 'missing' for r in rows)})")
    print(f"recovered: ours {tot('ours')}  PySR-alone same inputs {tot('base')}  PySR-alone original {tot('orig')}")
    gained = [r["eq"] for r in rows if r["ours"] and not r["base"]]
    lost = [r["eq"] for r in rows if r["base"] and not r["ours"]]
    print(f"vs same-input baseline: gained {len(gained)}, lost {len(lost)}")
    print(f"vs original: ours-not-orig {sum(r['ours'] and not r['orig'] for r in rows)}, "
          f"orig-not-ours {sum(r['orig'] and not r['ours'] for r in rows)}")
    print("lost:", lost)
    print("row classes:", dict(Counter(r["rows"] for r in rows)))
    print("n vs d:", dict(Counter("n<d" if r["n"] < r["d"] else "n=d" if r["n"] == r["d"] else "n>d"
                                 for r in rows if r["n"] is not None)))
    for cls in ("pure", "span", "other"):
        sub = [r for r in rows if r["rows"] == cls]
        print(f"  {cls:5s} {len(sub):3d} eqs, recovered ours {sum(r['ours'] for r in sub)}, orig {sum(r['orig'] for r in sub)}")
    print("by d:   d  count  orig  base  ours")
    for d in sorted({min(r["d"], 6) for r in rows}):
        sub = [r for r in rows if min(r["d"], 6) == d]
        lab = f"{d}" if d < 6 else ">=6"
        print(f"      {lab:>3s}  {len(sub):5d} {sum(r['orig'] for r in sub):5d} {sum(r['base'] for r in sub):5d} {sum(r['ours'] for r in sub):5d}")
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    print(f"median stage1+2 s {med([r['ours_stage12_s'] for r in rows]):.0f}; "
          f"median PySR s {med([r['ours_pysr_s'] for r in rows if r['ours_pysr_s']]):.0f}; "
          f"PySR at 1 h cap {sum((r['ours_pysr_s'] or 0) > 3500 for r in rows)}")
    near = [r for r in rows if not r["ours"] and r["ours_l2"] is not None and r["ours_l2"] < 1e-3]
    print(f"near misses (<1e-3) {len(near)}: {[r['eq'] for r in near]}")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    main()
