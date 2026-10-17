"""Write one SLURM array task per (equation, seed): `<equation_id> <seed>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/feynman/manifest.json"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--equations", default="", help="Comma-separated subset; default all.")
    parser.add_argument("--positive-only", action="store_true",
                        help="Skip equations whose inputs are not strictly positive.")
    parser.add_argument("--output", type=Path, default=Path("cluster/tasks.txt"))
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.equations:
        wanted = set(args.equations.split(","))
        entries = [e for e in entries if e["equation_id"] in wanted]
    if args.positive_only:
        entries = [
            e for e in entries
            if e.get("inputs_positive", None) is True
            or ("input_ranges" in e and all(low > 0 for low, _ in e["input_ranges"].values()))
        ]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    lines = [f"{entry['equation_id']} {seed}" for seed in seeds for entry in entries]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} tasks -> {args.output} (submit with --array=1-{len(lines)})")


if __name__ == "__main__":
    main()
