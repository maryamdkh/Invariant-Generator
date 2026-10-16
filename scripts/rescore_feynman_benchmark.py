"""Recompute recovery fields in a Feynman benchmark summary from the saved
expanded equations, using the current recovery_report. The original summary
is kept next to it as benchmark_summary.before_rescore.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from invariant_generator.adaptive_symbolic import load_known_formula, recovery_report
from invariant_generator.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--config", default="configs/feynman_benchmark.toml")
    parser.add_argument("--manifest", type=Path, default=Path("data/feynman/manifest.json"))
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    backup = args.summary.with_name("benchmark_summary.before_rescore.json")
    if not backup.exists():
        shutil.copyfile(args.summary, backup)
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    entries = {e["equation_id"]: e for e in json.loads(args.manifest.read_text(encoding="utf-8"))}
    base = load_config(args.config)
    if args.data_dir is not None:
        base.data.data_dir = args.data_dir.resolve()
    for row in rows:
        entry = entries[row["equation_id"]]
        base.data.dataset_name = entry["equation_id"]
        base.data.feature_names = list(entry["variables"])
        known = load_known_formula(base, fallback_X=None)
        if known is None:
            continue
        truth, info = known
        for arm in ("pipeline", "baseline"):
            result = row.get(arm, {})
            if "expanded_equation" not in result:
                continue
            for stale in ("recovered_within_tolerance", "snapped_candidate", "tolerance_criterion"):
                result.pop(stale, None)
            result.update(recovery_report(
                result["expanded_equation"], truth, info["symbols"], X_fresh=info["X_fresh"],
            ))
            print(f"{row['equation_id']:20s} {arm:8s} numerical={result['numerical_recovery']} "
                  f"symbolic={result['symbolic_recovery']} err={result['fresh_sample_relative_l2']:.1e}")
    args.summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
