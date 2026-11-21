"""Check the implemented Stage 1 rule (knee_ratio + target_choice) against
the offline FEY-L1-S1 analysis (RESEARCH_LOG 90, 91). Stage 1 only, no PySR.

    uv run python experiments/probes/stage1_rule_check.py --equations feynman_III_15_12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/feynman_redundant_linear_knee.toml"))
    p.add_argument("--level", default="L1")
    p.add_argument("--equations", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("results/feynman_benchmark/fey_l1_a2_check/seed42"))
    p.add_argument("--analysis", type=Path, default=Path("experiments/outputs/fey_l1_s1_analysis.json"))
    a = p.parse_args()

    from invariant_generator.adaptive import run_adaptive_sweep
    from invariant_generator.config import PROJECT_ROOT, load_config
    from invariant_generator.feynman_benchmark import choose_stage1_target, problem_config

    base = load_config(a.config)
    base.data.data_dir = PROJECT_ROOT / "data/feynman_redundant" / a.level
    entries = {e["equation_id"]: e for e in json.loads((base.data.data_dir / "manifest.json").read_text())}
    expected = {r["eq"]: r for r in json.loads(a.analysis.read_text())["rows"]}
    for eq in a.equations.split(","):
        started = time.perf_counter()
        config = problem_config(base, entries[eq], results_dir=a.results_dir / eq)
        config, reference, record = choose_stage1_target(config)
        sweep = run_adaptive_sweep(config, reference=reference)
        e = expected[eq]
        target = record.get("stage1_target", config.data.target_transform)
        ok = sweep.selected_n == e["A1.25_B"] and target == e["best_target"]
        print(f"{eq}: target {target} (expected {e['best_target']}), n {sweep.selected_n} "
              f"(expected {e['A1.25_B']}, n* {e['n_star']}, old {e['old_def']}), trained n "
              f"{[r.n for r in sweep.runs]}, {time.perf_counter() - started:.0f} s -> "
              f"{'MATCH' if ok else 'DIFFERENT'}", flush=True)


if __name__ == "__main__":
    main()
