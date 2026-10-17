"""Re-pick each arm's PySR equation with the first hall-of-fame entry scored.

PySR's model_selection="best" takes the highest score (-dlog(loss)/dcomplexity)
among equations with loss <= 1.5 x the minimum loss, and the first (simplest)
entry always has score 0. When Stage 2 hands PySR the exact group, the truth
is often that first entry (`J1`), so under noise a slightly better wiggle
(`J1 - 0.11 sin(19.6 J1)`) wins. Here the first entry is scored against the
constant model (loss = var(y)), the rest as in PySR, and the same rule is
applied to the pipeline and PySR-alone arms. Offline: uses the saved
equations.csv files, no new PySR runs.

Usage: python experiments/probes/reselect_simplest.py results/feynman_benchmark/v3_noise1pct [--data-dir data/feynman_all]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pick(rows: list[dict], reference_loss: float, *, score_first: bool) -> dict:
    rows = sorted(rows, key=lambda r: int(r["complexity"]))
    losses = np.array([float(r["loss"]) for r in rows])
    complexities = np.array([int(r["complexity"]) for r in rows])
    scores = np.zeros(len(rows))
    scores[1:] = -np.diff(np.log(losses)) / np.diff(complexities)
    if score_first:
        scores[0] = -np.log(losses[0] / reference_loss) / complexities[0]
    eligible = losses <= 1.5 * losses.min()
    return rows[int(np.argmax(np.where(eligible, scores, -np.inf)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/feynman_all"))
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args()

    import sympy as sp

    from invariant_generator.adaptive_symbolic import (
        _rationalize_floats, _with_time_limit, load_known_formula, symbolic_ratio_check,
    )
    from invariant_generator.config import Config

    merged = json.loads((args.run_dir / "merged_summary.json").read_text(encoding="utf-8"))
    out = []
    for row in merged:
        eq = row["equation_id"]
        meta = json.loads((args.data_dir / f"{eq}.json").read_text(encoding="utf-8"))
        names = list(meta["variables"])
        config = Config()
        config.data.data_dir = args.data_dir
        config.data.dataset_name = eq
        config.data.feature_names = names
        truth, info = load_known_formula(config, fallback_X=np.empty((0, len(names))))
        symbols = info["symbols"]
        ordered = [symbols[n] for n in names]
        X = info["X_fresh"]
        cols = [X[:, i] for i in range(X.shape[1])]
        y_true = np.broadcast_to(np.asarray(sp.lambdify(ordered, truth, "numpy")(*cols), dtype=float), (len(X),))
        sst = float(np.sum((y_true - y_true.mean()) ** 2))

        def score(expr) -> dict:
            with np.errstate(all="ignore"):
                y = np.broadcast_to(np.asarray(sp.lambdify(ordered, expr, "numpy")(*cols), dtype=complex), (len(X),))
                r2 = 1.0 - float(np.sum(np.abs(y - y_true) ** 2)) / sst
            r2 = r2 if np.isfinite(r2) else None
            symbolic = False
            if r2 is not None and r2 > 0.99:
                cleaned = _rationalize_floats(expr, digits=17, zero_tolerance=1e-6)
                checked = _with_time_limit(30, lambda: symbolic_ratio_check(cleaned, truth, digits=args.digits))
                symbolic = None if checked is None else bool(checked[0])
            return {"r2": r2, "accurate": bool(r2 is not None and r2 > 0.999), "symbolic": symbolic, "equation": str(expr)}

        base_root = args.run_dir / f"seed{row.get('seed', 42)}" / eq / eq
        entry = {"equation_id": eq}
        # PySR alone: x0..x{d-1} -> variable names.
        hof = list(csv.DictReader(open(base_root / "baseline_pysr" / "equations.csv")))
        y_ref = np.load(next((base_root).rglob("heldout_predictions.npz")))["y_test"] if any(base_root.rglob("heldout_predictions.npz")) else None
        reference = float(np.var(y_true)) if y_ref is None else float(np.var(y_ref))
        xmap = {sp.Symbol(f"x{i}"): symbols[n] for i, n in enumerate(names)}
        for mode in ("pysr_best", "score_first"):
            chosen = pick(hof, reference, score_first=mode == "score_first")
            entry[f"baseline_{mode}"] = score(sp.sympify(chosen["sympy_format"]).subs(xmap, simultaneous=True))
        # Pipeline: J_k = prod x_j^e_kj (exact snapped exponents, no intercept).
        pipe = row["pipeline"]
        stage3 = base_root / "stage3_pysr" / "equations.csv"
        if pipe.get("reduction") and pipe.get("snapped") and stage3.exists():
            jmap = {
                sp.Symbol(f"J{k + 1}"): sp.Mul(*[symbols[n] ** sp.Rational(e) for n, e in zip(names, exps)])
                for k, exps in enumerate(pipe["exponents"])
            }
            hof = list(csv.DictReader(open(stage3)))
            for mode in ("pysr_best", "score_first"):
                chosen = pick(hof, reference, score_first=mode == "score_first")
                entry[f"pipeline_{mode}"] = score(sp.sympify(chosen["sympy_format"]).subs(jmap, simultaneous=True))
        else:
            for mode in ("pysr_best", "score_first"):
                entry[f"pipeline_{mode}"] = entry[f"baseline_{mode}"]
        out.append(entry)
        print(eq, {k: (v["accurate"], v["symbolic"]) for k, v in entry.items() if isinstance(v, dict)}, flush=True)
    (args.run_dir / "reselect_simplest.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n| arm / selection | R2 > 0.999 | symbolic |")
    print("|---|---|---|")
    for key in ("pipeline_pysr_best", "pipeline_score_first", "baseline_pysr_best", "baseline_score_first"):
        print(f"| {key} | {sum(e[key]['accurate'] for e in out)} | {sum(e[key]['symbolic'] is True for e in out)} |")


if __name__ == "__main__":
    main()
