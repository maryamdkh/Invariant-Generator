"""FEY-L1-S1: full Stage 1 sweep (every n, both targets), no Stage 2/3, no PySR.

Purpose: data for choosing a better n-selection rule offline (RESEARCH_LOG
89). For each equation and each Stage 1 target (log y, y) it trains the
bottleneck-free reference and every n = 1 .. min(inputs - 1, d + 2) with the
usual restarts, without stopping at the first accepted n. Uses the existing
pipeline code unchanged (only `adaptive.stop_on_first_success = false` and
`data.target_transform` are set per run), so the old rule's choice is still
recorded as `selected_n_old_rule`.

    uv run python experiments/probes/stage1_sweep.py \
        --config configs/feynman_redundant_linear.toml \
        --manifest data/feynman_redundant/L1/manifest.json \
        --data-dir data/feynman_redundant/L1 \
        --results-dir results/feynman_benchmark/fey_l1_s1/seed42/<eq> \
        --equations <eq> --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def compact(summary: dict, metric: str) -> dict:
    ref = summary.get("reference") or {}
    return {
        "metric": metric,
        "threshold_old_rule": summary["threshold"],
        "selected_n_old_rule": summary["selected_n"],
        "reference": {
            "train": ref.get("train_metrics", {}).get(metric),
            "val": ref.get("test_metrics", {}).get(metric),
            "gradient_spectrum": ref.get("gradient_spectrum"),
        },
        "runs": [
            {
                "n": run["n"],
                "train": run["train_metrics"][metric],
                "val": run["test_metrics"][metric],
                "restarts_val": [r["test_metrics"][metric] for r in run.get("restarts", [])],
                "restarts_train": [r["train_metrics"][metric] for r in run.get("restarts", [])],
            }
            for run in summary["runs"]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/feynman_redundant_linear.toml"))
    p.add_argument("--manifest", type=Path, default=Path("data/feynman_redundant/L1/manifest.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/feynman_redundant/L1"))
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--equations", required=True, help="Comma-separated equation ids.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--targets", default="log,identity")
    p.add_argument("--n-max", type=int, default=0, help="Cap n (smoke tests only); 0 = inputs - 1.")
    p.add_argument("--device", default="", help="Override [train].device (cpu, cuda); default: config.")
    p.add_argument("--above-d", type=int, default=2,
                   help="Diagnostic cap n <= d + above_d (d = original variables); -1 = no cap. "
                        "No candidate rule needs n beyond d + 2; the rules never see d.")
    args = p.parse_args()

    from invariant_generator.adaptive import run_adaptive_sweep
    from invariant_generator.config import PROJECT_ROOT, load_config
    from invariant_generator.feynman_benchmark import problem_config

    base = load_config(args.config)
    base.data.data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    base.train.seed = args.seed
    if args.device:
        base.train.device = args.device
    import torch
    print(f"device {base.train.device}, cuda available {torch.cuda.is_available()}", flush=True)
    base.symbolic.random_state = args.seed
    entries = {e["equation_id"]: e for e in json.loads(args.manifest.read_text(encoding="utf-8"))}

    for eq in args.equations.split(","):
        entry = entries[eq]
        out = {"equation_id": eq, "inputs": len(entry["variables"]), "seed": args.seed, "targets": {}}
        if "original_variables" in entry:
            out["d"] = len(entry["original_variables"])
        for target in args.targets.split(","):
            base.data.target_transform = target
            base.symbolic.target_transform = "exp" if target == "log" else "identity"
            config = problem_config(base, entry, results_dir=args.results_dir / target)
            if config.data.target_transform != target:
                out["targets"][target] = {"skipped": "y <= 0 somewhere; log not possible"}
                continue
            config.adaptive.stop_on_first_success = False
            if args.n_max:
                config.adaptive.n_max = min(config.adaptive.n_max, args.n_max)
            if args.above_d >= 0 and "d" in out:
                config.adaptive.n_max = min(config.adaptive.n_max, out["d"] + args.above_d)
            started = time.perf_counter()
            result = run_adaptive_sweep(config)
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            out["targets"][target] = {
                **compact(summary, summary["metric"]),
                "n_max": config.adaptive.n_max,
                "seconds": time.perf_counter() - started,
            }
            print(f"{eq} {target}: old rule n={summary['selected_n']}, "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / "stage1_sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
