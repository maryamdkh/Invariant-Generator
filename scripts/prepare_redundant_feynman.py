"""Redundant-input Feynman datasets (x = sum of parts, plus distractors).

Each original input x (range [lo, hi]) is replaced by parts whose sum is x:

    L1  x = w + z            z ~ U[lo, hi], w = x - z
    L2  x = u1 + u2 + u3     u2, u3 ~ U[lo, hi], u1 = x - u2 - u3
    L3  L1 plus d distractors that do not enter y (d = number of original
        inputs); distractor k has the distribution of the w (k even) or z
        (k odd) part of original input k, so scale does not give it away.

The original rows (X, y) of every equation are reused unchanged, so y, the
train/test split and the formula's difficulty are those of the FEY-B
baseline; only the representation of x changes. Columns are shuffled and
named u1..uD; the true mapping is stored in the JSON ("mapping",
"distractors"). The JSON formula is the original formula with each x
replaced by its sum, so the existing scoring works unchanged. Fresh scoring
points ("X_eval", 10,000 rows) are drawn the same way: x uniform on its
original range, then split, so the sums stay inside the original domain.

    uv run python scripts/prepare_redundant_feynman.py \
        --source data/feynman --output data/feynman_redundant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import sympy as sp

LEVELS = {"L1": (2, False), "L2": (3, False), "L3": (2, True)}
N_EVAL = 10_000


def split_columns(x: np.ndarray, lo: float, hi: float, parts: int, rng) -> list[np.ndarray]:
    """Return `parts` columns that sum to x; all but the first are U[lo, hi]."""
    free = [rng.uniform(lo, hi, size=x.shape[0]) for _ in range(parts - 1)]
    return [x - np.sum(free, axis=0)] + free


def build(entry: dict, X: np.ndarray, level: str, seed: int):
    """Expanded inputs for the data rows and fresh rows, plus the metadata."""
    parts, distractors = LEVELS[level]
    names = list(entry["variables"])
    ranges = entry["input_ranges"]
    rng = np.random.default_rng(seed)
    X_eval = np.column_stack([rng.uniform(*ranges[n], size=N_EVAL) for n in names])

    columns, eval_columns, labels = [], [], []  # labels: (kind, original, part)
    for i, name in enumerate(names):
        lo, hi = ranges[name]
        columns += split_columns(X[:, i], lo, hi, parts, rng)
        eval_columns += split_columns(X_eval[:, i], lo, hi, parts, rng)
        labels += [("part", name, p) for p in range(parts)]
    if distractors:
        for k, name in enumerate(names):
            lo, hi = ranges[name]
            for rows, out in ((X.shape[0], columns), (N_EVAL, eval_columns)):
                fake_x = rng.uniform(lo, hi, size=rows)
                w, z = split_columns(fake_x, lo, hi, 2, rng)
                out.append(w if k % 2 == 0 else z)
            labels.append(("distractor", name, k % 2))

    order = rng.permutation(len(columns))
    new_names = [f"u{j + 1}" for j in range(len(columns))]
    U = np.column_stack([columns[j] for j in order])
    U_eval = np.column_stack([eval_columns[j] for j in order])
    labels = [labels[j] for j in order]

    sums = {name: [] for name in names}
    distractor_names = []
    for new, (kind, name, _) in zip(new_names, labels):
        (sums[name] if kind == "part" else distractor_names).append(new)
    mapping = {name: " + ".join(sums[name]) for name in names}

    originals = {n: sp.Symbol(n) for n in names}
    new_symbols = {n: sp.Symbol(n) for n in new_names}
    formula = sp.sympify(entry["formula"], locals=originals)
    expanded = formula.subs(
        {originals[n]: sp.sympify(mapping[n], locals=new_symbols) for n in names},
        simultaneous=True,
    )
    meta = {
        "equation_id": entry["equation_id"],
        "target": entry.get("target"),
        "formula": str(expanded),
        "variables": new_names,
        "input_ranges": {n: [float(U_eval[:, j].min()), float(U_eval[:, j].max())]
                         for j, n in enumerate(new_names)},
        "inputs_positive": False,
        "eval_key": "X_eval",
        "redundancy_level": level,
        "parts_per_input": parts,
        "original_formula": entry["formula"],
        "original_variables": names,
        "original_input_ranges": ranges,
        "mapping": mapping,
        "distractors": distractor_names,
        "samples": int(X.shape[0]),
        "seed": seed,
        "source": f"{entry.get('source', 'Feynman')}; redundant inputs "
                  "(scripts/prepare_redundant_feynman.py)",
    }
    return U, U_eval, meta


def check(meta: dict, U: np.ndarray, y: np.ndarray) -> float:
    """Relative L2 of the expanded formula on the stored rows vs the stored y."""
    symbols = [sp.Symbol(n) for n in meta["variables"]]
    f = sp.lambdify(symbols, sp.sympify(meta["formula"], locals=dict(zip(meta["variables"], symbols))), "numpy")
    with np.errstate(all="ignore"):
        pred = np.broadcast_to(np.asarray(f(*U.T), dtype=np.float64), y.shape)
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("data/feynman"))
    parser.add_argument("--output", type=Path, default=Path("data/feynman_redundant"))
    parser.add_argument("--levels", default="L1,L2,L3")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    entries = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    for level in args.levels.split(","):
        out_dir = args.output / level
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest, worst = [], 0.0
        for idx, entry in enumerate(entries):
            with h5py.File(args.source / f"{entry['equation_id']}.h5", "r") as handle:
                X = np.asarray(handle["X"], dtype=np.float64)
                y = np.asarray(handle["y"], dtype=np.float64)
            U, U_eval, meta = build(entry, X, level, args.seed + idx)
            error = check(meta, U, y)
            if not error < 1e-10:
                raise RuntimeError(f"{level} {entry['equation_id']}: y not reproduced ({error:.2e})")
            worst = max(worst, error)
            with h5py.File(out_dir / f"{entry['equation_id']}.h5", "w") as handle:
                handle["X"], handle["y"], handle["X_eval"] = U, y, U_eval
            (out_dir / f"{entry['equation_id']}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest.append({k: meta[k] for k in (
                "equation_id", "target", "formula", "variables", "input_ranges",
                "original_formula", "original_variables", "mapping", "distractors",
                "redundancy_level")})
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        sizes = [len(m["variables"]) for m in manifest]
        print(f"{level}: {len(manifest)} equations, inputs {min(sizes)}-{max(sizes)}, "
              f"worst y reproduction error {worst:.1e} -> {out_dir}")


if __name__ == "__main__":
    main()
