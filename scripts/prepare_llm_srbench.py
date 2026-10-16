"""Convert LLM-SRBench LSR-Transform problems to the supervised benchmark format.

Input: the Hugging Face dataset nnheui/llm-srbench downloaded to
data/llm_srbench (gated; accept its terms first). Data are stored as float32
with columns [output, inputs...] in the order of `symbols`.

Output per problem, data/llm_srbench_converted/lsrt_<name>.h5:
  X, y      a seeded subsample of the official train split (pipeline data;
            our train/validation/test split is made from these rows)
  X_eval    a seeded subsample of the official test split, used only to score
            candidates against the ground-truth formula evaluated in float64
and a JSON with the formula, variables and provenance, plus manifest.json.
LSR-Synth problems are not converted: their ground truths contain named
constants without values, so they can only be scored by numeric error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sympy as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/llm_srbench"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/llm_srbench_converted"))
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    table = pd.read_parquet(args.source / "data" / "lsr_transform-00000-of-00001.parquet")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    with h5py.File(args.source / "lsr_bench_data.hdf5", "r") as source:
        for _, row in table.iterrows():
            name = str(row["name"])
            symbols = [str(s) for s in row["symbols"]]
            target, variables = symbols[0], symbols[1:]
            train = np.asarray(source[f"lsr_transform/{name}/train"], dtype=np.float64)
            test = np.asarray(source[f"lsr_transform/{name}/test"], dtype=np.float64)
            rng = np.random.default_rng(args.seed)
            train = train[rng.choice(len(train), min(args.train_samples, len(train)), replace=False)]
            test = test[rng.choice(len(test), min(args.eval_samples, len(test)), replace=False)]

            local = {s: sp.Symbol(s) for s in symbols}
            formula = sp.sympify(str(row["expression"]), locals=local)
            function = sp.lambdify([local[v] for v in variables], formula, "numpy")
            with np.errstate(all="ignore"):
                check = np.broadcast_to(np.asarray(function(*train[:, 1:].T), float), (len(train),))
            median_error = float(np.nanmedian(np.abs(check - train[:, 0]) / np.maximum(np.abs(train[:, 0]), 1e-12)))

            equation_id = f"lsrt_{name}"
            path = args.output_dir / f"{equation_id}.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("X", data=train[:, 1:])
                handle.create_dataset("y", data=train[:, 0])
                handle.create_dataset("X_eval", data=test[:, 1:])
            entry = {
                "equation_id": equation_id,
                "benchmark": "LLM-SRBench LSR-Transform",
                "source_name": name,
                "target": target,
                "formula": str(formula),
                "variables": variables,
                "eval_key": "X_eval",
                "inputs_positive": bool((train[:, 1:] > 0).all() and (test[:, 1:] > 0).all()),
                "target_positive": bool((train[:, 0] > 0).all()),
                "formula_vs_data_median_relative_error": median_error,
                "train_samples": int(len(train)),
                "eval_samples": int(len(test)),
                "seed": args.seed,
                "source": "huggingface.co/datasets/nnheui/llm-srbench (lsr_bench_data.hdf5, float32)",
            }
            path.with_suffix(".json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
            manifest.append(entry)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    positive = sum(e["inputs_positive"] for e in manifest)
    worst = max(e["formula_vs_data_median_relative_error"] for e in manifest)
    print(f"{len(manifest)} problems -> {args.output_dir} ({positive} with positive inputs; "
          f"worst median formula-vs-data error {worst:.1e})")


if __name__ == "__main__":
    main()
