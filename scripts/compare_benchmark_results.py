from __future__ import annotations

import argparse

from invariant_generator.benchmark import collect_benchmark_results
from invariant_generator.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect aggregate results for generated benchmark cases."
    )
    parser.add_argument("--config", default="configs/self_eval_benchmark.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    result = collect_benchmark_results(config)
    print(f"[INFO] Benchmark root:    {result['root']}")
    print(f"[INFO] Completed cases:   {result['n_completed']}/{result['n_cases']}")
    print(f"[INFO] Classifications:   {result['classification_counts']}")
    print(f"[INFO] Comparison JSON:   {result['summary_path']}")


if __name__ == "__main__":
    main()
