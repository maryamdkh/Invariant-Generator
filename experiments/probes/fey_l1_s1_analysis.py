"""FEY-L1-S1 analysis: choose the Stage 1 n-selection rule offline (RESEARCH_LOG 89, 90).

Reads the full Stage 1 sweeps (`stage1_sweep.py`) and, per equation:

  n*   true minimal linear bottleneck = rank of the formula's input gradients
       (sympy Jacobian at 400 training rows; singular values > 1e-6 of the
       largest, rows normalised). n* <= d; e.g. I_8_14 (distance) has n* = 2.
  old  the current rule: first n with train and val nrmse <= max(2 x ref, 1e-3)
  A(r) knee: first n that passes the old threshold and whose val nrmse is
       <= r x val(n + 1) (the last swept n needs no n + 1)
  B    target: log y or y, whichever reference has the lower val nrmse
       (y only when log is impossible)

r is chosen on the 21 pilot (dev) equations only, from {1.1, 1.25, 1.5, 2}:
most dev equations with n = n*; ties -> the larger r (closest to the old
rule). It is then frozen and scored on the 98 held-out equations.

    uv run python experiments/probes/fey_l1_s1_analysis.py \
        --out experiments/outputs/fey_l1_s1_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import sympy as sp

SWEEP = Path("results/feynman_benchmark/fey_l1_s1/seed42")
DATA = Path("data/feynman_redundant/L1")
DEV = {f"feynman_{e}" for e in (
    "III_9_52 II_36_38 II_6_15a I_11_19 I_32_17 I_40_1 I_9_18 test_1 test_11 test_12 "
    "test_16 test_17 test_18 test_19 test_3 test_4 test_5 test_6 test_7 test_8 test_9").split()}
RS = (1.1, 1.25, 1.5, 2.0)


def true_rank(meta: dict, rows: int = 400, tol: float = 1e-6) -> tuple[int, list[float]]:
    names = meta["variables"]
    syms = sp.symbols(names)
    expr = sp.sympify(meta["formula"], locals=dict(zip(names, syms)))
    grad = sp.lambdify(syms, [sp.diff(expr, s) for s in syms], "numpy")
    X = h5py.File(DATA / f"{meta['equation_id']}.h5")["X"][:rows]
    with np.errstate(all="ignore"):
        G = np.array([np.broadcast_to(np.asarray(g, dtype=float), (len(X),))
                      for g in grad(*X.T)]).T
    G = G[np.all(np.isfinite(G), axis=1)]
    G = G[np.linalg.norm(G, axis=1) > 0]
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    s = np.linalg.svd(G, compute_uv=False)
    s = s / s[0]
    return int(np.sum(s > tol)), [float(v) for v in s]


def passes(run: dict, thr: float) -> bool:
    return run["train"] <= thr and run["val"] <= thr


def select(t: dict, r: float | None) -> int | None:
    runs, thr = t["runs"], t["threshold_old_rule"]
    for i, run in enumerate(runs):
        if not passes(run, thr):
            continue
        if r is None or i == len(runs) - 1 or run["val"] <= r * runs[i + 1]["val"]:
            return run["n"]
    return None


def best_target(targets: dict) -> str:
    ok = {k: v for k, v in targets.items() if "skipped" not in v}
    return min(ok, key=lambda k: ok[k]["reference"]["val"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    rows = []
    for path in sorted(SWEEP.glob("*/stage1_sweep.json")):
        s = json.loads(path.read_text())
        eq = s["equation_id"]
        meta = json.loads((DATA / f"{eq}.json").read_text())
        n_star, spec = true_rank(meta)
        T = {k: v for k, v in s["targets"].items() if "skipped" not in v}
        default = "log" if "log" in T else "identity"   # what the pipeline uses now
        bt = best_target(T)
        row = {"eq": eq, "dev": eq in DEV, "d": s["d"], "inputs": s["inputs"], "n_star": n_star,
               "grad_spectrum": spec[: n_star + 2], "default_target": default, "best_target": bt,
               "ref_val": {k: v["reference"]["val"] for k, v in T.items()},
               "old_recorded": T[default]["selected_n_old_rule"]}
        for tname, tgt in (("def", default), ("B", bt)):
            row[f"old_{tname}"] = select(T[tgt], None)
            for r in RS:
                row[f"A{r}_{tname}"] = select(T[tgt], r)
        # step ratios val(n)/val(n+1) on the default target, split by role
        runs = T[default]["runs"]
        row["ratio_missing"] = [runs[i]["val"] / runs[i + 1]["val"]
                                for i in range(len(runs) - 1) if runs[i]["n"] < n_star]
        row["ratio_unneeded"] = [runs[i]["val"] / runs[i + 1]["val"]
                                 for i in range(len(runs) - 1) if runs[i]["n"] >= n_star]
        at = [r for r in runs if r["n"] == n_star]
        rv = sorted(at[0]["restarts_val"]) if at else []
        row["restart_spread_at_nstar"] = rv[1] / rv[0] if len(rv) > 1 else None
        rows.append(row)

    dev = [r for r in rows if r["dev"]]
    held = [r for r in rows if not r["dev"]]
    assert len(dev) == 21, len(dev)
    print(f"equations {len(rows)} (dev {len(dev)}, held out {len(held)})")
    print(f"old rule recomputed == recorded: {sum(r['old_def'] == r['old_recorded'] for r in rows)}/{len(rows)}")
    print(f"n* < d: {sum(r['n_star'] < r['d'] for r in rows)}: "
          f"{[(r['eq'][8:], r['d'], r['n_star']) for r in rows if r['n_star'] < r['d']]}")
    print(f"best target: log {sum(r['best_target'] == 'log' for r in rows)}, "
          f"y {sum(r['best_target'] == 'identity' for r in rows)} "
          f"(log impossible {sum('log' not in r['ref_val'] for r in rows)})")

    def table(sub: list[dict], label: str) -> None:
        print(f"\n{label} ({len(sub)} eqs): rule  n=n*  n<n*  n>n*  none  n=d")
        keys = ["old_def", "old_B"] + [f"A{r}_{t}" for t in ("def", "B") for r in RS]
        for k in keys:
            got = [(r[k], r["n_star"], r["d"]) for r in sub]
            print(f"  {k:9s} {sum(n == s for n, s, _ in got):5d} {sum(n is not None and n < s for n, s, _ in got):5d} "
                  f"{sum(n is not None and n > s for n, s, _ in got):5d} {sum(n is None for n, s, _ in got):5d} "
                  f"{sum(n == d for n, _, d in got):5d}")

    table(dev, "DEV")
    q = lambda xs, p: float(np.quantile(xs, p)) if xs else float("nan")
    mis = [x for r in dev for x in r["ratio_missing"]]
    unn = [x for r in dev for x in r["ratio_unneeded"]]
    print(f"dev step ratios val(n)/val(n+1): missing-variable steps n<n* "
          f"(n={len(mis)}) min {min(mis):.2f} 5% {q(mis, .05):.2f} median {q(mis, .5):.2f}; "
          f"unneeded steps n>=n* (n={len(unn)}) median {q(unn, .5):.2f} 95% {q(unn, .95):.2f} max {max(unn):.2f}")
    sp_ = [r["restart_spread_at_nstar"] for r in dev if r["restart_spread_at_nstar"]]
    print(f"dev restart spread at n* (2nd best / best val): median {q(sp_, .5):.2f}, 90% {q(sp_, .9):.2f}")

    choose = {}
    for tname in ("def", "B"):
        score = {r_: sum(x[f"A{r_}_{tname}"] == x["n_star"] for x in dev) for r_ in RS}
        best = max(score.values())
        choose[tname] = max(r_ for r_ in RS if score[r_] == best)
        print(f"dev choice ({tname} target): scores {score} -> r = {choose[tname]}")

    print("\nFROZEN: r =", choose)
    table(held, "HELD OUT")
    for tname in ("def", "B"):
        k = f"A{choose[tname]}_{tname}"
        print(f"  frozen {k}: held-out n=n* {sum(x[k] == x['n_star'] for x in held)}/{len(held)} "
              f"vs old {sum(x['old_def'] == x['n_star'] for x in held)}/{len(held)}")
    mis = [x for r in held for x in r["ratio_missing"]]
    unn = [x for r in held for x in r["ratio_unneeded"]]
    print(f"held-out step ratios: missing 5% {q(mis, .05):.2f} min {min(mis):.2f}; "
          f"unneeded 95% {q(unn, .95):.2f} max {max(unn):.2f}")

    print("\nheld-out misses under frozen A+B:")
    k = f"A{choose['B']}_B"
    for r in held:
        if r[k] != r["n_star"]:
            print(f"  {r['eq'][8:]:12s} d={r['d']} n*={r['n_star']} old={r['old_def']} A+B={r[k]} "
                  f"target={r['best_target']} ref_val={ {t: round(v, 4) for t, v in r['ref_val'].items()} }")
    if a.out:
        a.out.write_text(json.dumps({"frozen_r": choose, "rows": rows}, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    main()
