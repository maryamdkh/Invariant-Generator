"""SA structure: do the encoder rows span {q1, q2}, and are they mixed?

    uv run python experiments/probes/sa_row_mixing.py results/synthetic/sa_dev3 \
        --output experiments/outputs/sa_row_mixing.json

For each problem with n = 2 (and n = 1), compares the learned rows (Stage 2)
with the true quadratic forms: principal-angle cosines between the row space
and span{q1, q2}, the largest cosine of each row with q1 and with q2 (near 1
= one row per form, otherwise mixed), and the row share on the linear terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp


def true_forms(formula: str, names: list[str], outer: str) -> list[sp.Expr]:
    expression = sp.sympify(formula, locals={name: sp.Symbol(name) for name in names})
    if outer == "sqrt(q1)":
        return [expression.args[0]]
    if outer == "q1/(1 + q2)":
        numerator, denominator = sp.fraction(sp.together(expression))
        return [sp.expand(numerator), sp.expand(denominator)]
    roots = list(expression.atoms(sp.Pow) & {a for a in expression.atoms(sp.Pow) if a.exp == sp.Rational(1, 2)})
    root = roots[0]
    return [root.args[0], sp.expand(expression - root)]


def coefficients(form: sp.Expr, names: list[str]) -> np.ndarray:
    symbols = [sp.Symbol(name) for name in names]
    poly = sp.Poly(form, *symbols)
    d = len(names)
    out = []
    for i in range(d):
        for j in range(i, d):
            powers = [0] * d
            powers[i] += 1
            powers[j] += 1
            out.append(float(poly.coeff_monomial(sp.prod(s**p for s, p in zip(symbols, powers)))))
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic/sa/manifest.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = {e["equation_id"]: e for e in json.loads(args.manifest.read_text(encoding="utf-8"))}
    rows_out = []
    for checkpoint in sorted(args.results.glob("*/stage2_graded/checkpoint_best.pt")):
        name = checkpoint.parts[-3]
        entry = manifest[name]
        names, d = entry["variables"], len(entry["variables"])
        summary = json.loads((checkpoint.parent / "adaptive_stage2_sparsify.json").read_text(encoding="utf-8"))
        B = np.asarray(summary["candidate_results"][0]["rows"], dtype=float)
        linear, quadratic = B[:, :d], B[:, d:]
        forms = true_forms(entry["formula"], names, entry["outer_form"])
        Q = np.vstack([coefficients(form, names) for form in forms])
        row_basis = np.linalg.qr(quadratic.T)[0]
        form_basis = np.linalg.qr(Q.T)[0]
        cosines = np.linalg.svd(row_basis.T @ form_basis, compute_uv=False)
        unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
        rows_out.append({
            "problem": name, "outer": entry["outer_form"], "n_rows": len(B),
            "principal_cosines": cosines.tolist(),
            "row_vs_form_cosines": np.abs(unit(quadratic) @ unit(Q).T).tolist(),
            "linear_share": (np.linalg.norm(linear, axis=1) / np.linalg.norm(B, axis=1)).tolist(),
        })
        print(json.dumps(rows_out[-1]))
    if args.output is not None:
        args.output.write_text(json.dumps(rows_out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
