"""Export one arm of a benchmark run to experiments/runs/<ID>/summary.csv.

    uv run python scripts/export_run_summary.py results/feynman_benchmark/llmsr_v3 \
        --arm pipeline --experiment-id LLM-A --dataset LLM --method A --commit 37b71cb

--commit is the commit the cluster ran (write it in the run README too).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from invariant_generator.experiment_summary import benchmark_rows, write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--arm", required=True, help="pipeline | baseline | portfolio")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: experiments/runs/<experiment-id>/summary.csv")
    args = parser.parse_args()
    rows = benchmark_rows(
        args.run_dir, arm=args.arm, experiment_id=args.experiment_id,
        dataset=args.dataset, method=args.method, commit=args.commit,
    )
    if not rows:
        raise SystemExit(f"No '{args.arm}' results under {args.run_dir}.")
    output = args.output or Path("experiments/runs") / args.experiment_id / "summary.csv"
    write_summary(output, rows)
    seeds = sorted({row["seed"] for row in rows if row["seed"] is not None})
    recovered = sum(bool(row["recovered"]) for row in rows)
    print(f"{output}: {len(rows)} rows, seeds {seeds}, recovered {recovered}")


if __name__ == "__main__":
    main()
