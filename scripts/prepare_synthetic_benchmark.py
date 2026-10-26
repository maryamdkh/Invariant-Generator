"""Synthetic scaling benchmarks (EXPERIMENTS.md, datasets SA, SB, SBD).

Same layout as scripts/prepare_feynman_benchmark.py (<id>.h5 with X, y;
<id>.json; manifest.json), so scripts/run_feynman_benchmark.py runs them.
"log_reducibility" holds the true number of groups n* (as for Feynman).

SA  rotated quadratic forms, d = 2, 3, 4, 6, 8, 10, 12, inputs U[-1, 1]:
    q_i = x^T A_i x with A_i = Q diag(lambda) Q^T, Q random orthogonal,
    lambda ~ U[0.5, 2]; coefficients rounded to 2 decimals.
      sqrt(q1)          (n* = 1; y^2 is a quadratic polynomial, so a sparse
                         linear fit on y^2 solves it: the SINDy control)
      q1 / (1 + q2)     (n* = 2)
      sqrt(q1) + q2     (n* = 2)
SB  power-law groups, d = 4, 6, 8, 10, 12, inputs U[0.5, 2]: inputs split at
    random into k = 1, 2, 3 groups pi_j = prod x_i^e_i, e_i in
    {-2, -1, -1/2, 1/2, 1, 2}; outer functions
      k = 1: pi1,  pi1 / (1 + pi1)
      k = 2: pi1 + pi2,  pi1 / (1 + pi2),  pi1 * sqrt(1 + pi2)
      k = 3: pi1 + pi2 / (1 + pi3),  pi1 * pi2 + pi3
SBD SB plus 3 distractor inputs U[0.5, 2] that do not enter y.

    uv run python scripts/prepare_synthetic_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import sympy as sp

SA_DIMENSIONS = (2, 3, 4, 6, 8, 10, 12)
SB_DIMENSIONS = (4, 6, 8, 10, 12)
SA_FORMS = ("sqrt(q1)",) * 3 + ("q1/(1 + q2)",) * 4 + ("sqrt(q1) + q2",) * 3
SB_FORMS = {
    1: ("pi1", "pi1/(1 + pi1)"),
    2: ("pi1 + pi2", "pi1/(1 + pi2)", "pi1*sqrt(1 + pi2)"),
    3: ("pi1 + pi2/(1 + pi3)", "pi1*pi2 + pi3"),
}
EXPONENTS = (-2, -1, sp.Rational(-1, 2), sp.Rational(1, 2), 1, 2)


def random_quadratic_form(d: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = Q @ np.diag(rng.uniform(0.5, 2.0, size=d)) @ Q.T
    A = np.round(0.5 * (A + A.T), 2)
    if np.linalg.eigvalsh(A).min() <= 0.1:
        return random_quadratic_form(d, rng)
    return A


def write_problem(output_dir: Path, entry: dict, X: np.ndarray, y: np.ndarray) -> None:
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"{entry['equation_id']}: non-finite targets")
    with h5py.File(output_dir / f"{entry['equation_id']}.h5", "w") as handle:
        handle.create_dataset("X", data=X)
        handle.create_dataset("y", data=y)
    (output_dir / f"{entry['equation_id']}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def make_sa(output_dir: Path, samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    manifest = []
    for d in SA_DIMENSIONS:
        x = sp.symbols(f"x1:{d + 1}")
        for index, form in enumerate(SA_FORMS):
            forms = {}
            for name in ("q1", "q2"):
                A = random_quadratic_form(d, rng)
                forms[name] = sum(sp.nsimplify(A[i, j]) * x[i] * x[j] for i in range(d) for j in range(d))
            expression = sp.sympify(form, locals={k: v for k, v in forms.items()})
            n_true = 1 if "q2" not in form else 2
            X = rng.uniform(-1.0, 1.0, size=(samples, d))
            y = sp.lambdify(x, expression, "numpy")(*X.T)
            entry = {
                "equation_id": f"sa_d{d:02d}_{index:02d}",
                "target": "y",
                "formula": str(expression),
                "outer_form": form,
                "variables": [str(s) for s in x],
                "input_ranges": {str(s): [-1.0, 1.0] for s in x},
                "log_reducibility": n_true,
                "n_true_groups": n_true,
                "samples": samples,
                "seed": seed,
                "source": "synthetic SA: rotated quadratic forms (scripts/prepare_synthetic_benchmark.py)",
                "generated_data": True,
            }
            write_problem(output_dir, entry, X, np.asarray(y, dtype=np.float64))
            manifest.append(entry)
    return manifest


def make_sb(output_dir: Path, samples: int, seed: int, distractors: int, prefix: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    manifest = []
    for d in SB_DIMENSIONS:
        x = sp.symbols(f"x1:{d + distractors + 1}", positive=True)
        for index in range(10):
            k = (index % 3) + 1
            forms = SB_FORMS[k]
            form = forms[(index // 3) % len(forms)]
            assignment = np.concatenate([np.arange(k), rng.integers(0, k, size=d - k)])
            rng.shuffle(assignment)
            groups = {}
            for j in range(k):
                members = np.flatnonzero(assignment == j)
                groups[f"pi{j + 1}"] = sp.Mul(*[x[i] ** EXPONENTS[rng.integers(len(EXPONENTS))] for i in members])
            expression = sp.sympify(form, locals=groups)
            X = rng.uniform(0.5, 2.0, size=(samples, d + distractors))
            y = sp.lambdify(x, expression, "numpy")(*X.T)
            entry = {
                "equation_id": f"{prefix}_d{d:02d}_{index:02d}",
                "target": "y",
                "formula": str(expression),
                "outer_form": form,
                "groups": {name: str(group) for name, group in groups.items()},
                "variables": [str(s) for s in x],
                "distractors": [str(s) for s in x[d:]],
                "input_ranges": {str(s): [0.5, 2.0] for s in x},
                "log_reducibility": k,
                "n_true_groups": k,
                "samples": samples,
                "seed": seed,
                "source": f"synthetic {prefix.upper()}: power-law groups (scripts/prepare_synthetic_benchmark.py)",
                "generated_data": True,
            }
            write_problem(output_dir, entry, X, np.asarray(y, dtype=np.float64))
            manifest.append(entry)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for name, build in {
        "sa": lambda out: make_sa(out, args.samples, args.seed),
        "sb": lambda out: make_sb(out, args.samples, args.seed + 1, 0, "sb"),
        "sbd": lambda out: make_sb(out, args.samples, args.seed + 1, 3, "sbd"),
    }.items():
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build(output_dir)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(manifest)} problems -> {output_dir}")


if __name__ == "__main__":
    main()
