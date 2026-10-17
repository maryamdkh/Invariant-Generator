"""Noise-tolerant scoring of a merged benchmark run (SRBench-style).

With target noise the fitted constants are never exact, so the 1e-6
relative-L2 criterion mostly tests constant precision. For every arm's final
candidate (`expanded_equation`) this reports, on noiseless fresh points:
- r2: coefficient of determination against the true formula;
- accurate: r2 > 0.999 (SRBench accuracy solution);
- symbolic: candidate/truth simplifies to a nonzero constant after rounding
  floats to `--digits` significant digits (SRBench symbolic solution uses
  a similar rounding; advisory, see symbolic_ratio_check).

Usage: python scripts/score_noisy_benchmark.py results/feynman_benchmark/v3_noise1pct
Run scripts/merge_feynman_benchmark.py first. Writes <run_dir>/noisy_scores.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("pipeline", "portfolio", "baseline")


def arm_result(row: dict, arm: str) -> dict:
    result = row.get(arm, {})
    if arm == "pipeline" and result.get("reduction") is False:
        return row.get("baseline", {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/feynman"))
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--simplify-seconds", type=int, default=30)
    args = parser.parse_args()

    import sympy as sp

    from invariant_generator.adaptive_symbolic import (
        _rationalize_floats,
        _with_time_limit,
        load_known_formula,
        symbolic_ratio_check,
    )
    from invariant_generator.config import Config

    rows = json.loads((args.run_dir / "merged_summary.json").read_text(encoding="utf-8"))
    scores = []
    for row in rows:
        config = Config()
        config.data.data_dir = args.data_dir
        config.data.dataset_name = row["equation_id"]
        metadata = json.loads((args.data_dir / f"{row['equation_id']}.json").read_text(encoding="utf-8"))
        config.data.feature_names = list(metadata.get("variables") or metadata["input_ranges"])
        known = load_known_formula(config, fallback_X=np.empty((0, len(config.data.feature_names))))
        truth, info = known
        ordered = list(info["symbols"].values())
        X = info["X_fresh"]
        columns = [X[:, i] for i in range(X.shape[1])]
        y_true = np.broadcast_to(np.asarray(sp.lambdify(ordered, truth, "numpy")(*columns), dtype=np.float64), (len(X),))
        entry = {"equation_id": row["equation_id"], "seed": row.get("seed")}
        for arm in ARMS:
            result = arm_result(row, arm)
            expression = result.get("expanded_equation")
            if expression is None:
                entry[arm] = {"r2": None, "accurate": False, "symbolic": False}
                continue
            candidate = sp.sympify(expression, locals=info["symbols"])
            with np.errstate(all="ignore"):
                y_hat = np.broadcast_to(
                    np.asarray(sp.lambdify(ordered, candidate, "numpy")(*columns), dtype=np.complex128),
                    (len(X),),
                )
                residual = float(np.sum(np.abs(y_hat - y_true) ** 2))
            r2 = 1.0 - residual / float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = r2 if np.isfinite(r2) else None
            symbolic = False
            if r2 is not None and r2 > 0.99:
                cleaned = _rationalize_floats(candidate, digits=17, zero_tolerance=1e-6)
                checked = _with_time_limit(
                    args.simplify_seconds,
                    lambda: symbolic_ratio_check(cleaned, truth, digits=args.digits),
                )
                symbolic = None if checked is None else bool(checked[0])
            entry[arm] = {
                "r2": r2,
                "accurate": bool(r2 is not None and r2 > 0.999),
                "symbolic": symbolic,
                "exact": bool(result.get("numerical_recovery")),
            }
        scores.append(entry)
        print(entry["equation_id"], {arm: (entry[arm]["accurate"], entry[arm]["symbolic"]) for arm in ARMS}, flush=True)
    (args.run_dir / "noisy_scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    print("\n| arm | exact (1e-6) | R2 > 0.999 | symbolic (%d digits) | median R2 |" % args.digits)
    print("|---|---|---|---|---|")
    for arm in ARMS:
        r2s = [s[arm]["r2"] for s in scores if s[arm]["r2"] is not None]
        print(
            f"| {arm} | {sum(bool(s[arm].get('exact')) for s in scores)} "
            f"| {sum(s[arm]['accurate'] for s in scores)} "
            f"| {sum(s[arm]['symbolic'] is True for s in scores)} "
            f"| {np.median(r2s):.6f} |"
        )


if __name__ == "__main__":
    main()
