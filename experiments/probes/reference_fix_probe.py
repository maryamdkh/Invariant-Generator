"""Reference-model fixes for Stage 1 (RESEARCH_LOG 93). Trains only the
bottleneck-free reference (no encoder sweep, no PySR) under variants:

  base     A2 setting: [64, 64], 3000 epochs, early-stopping patience 20
  neglog   y < 0 everywhere: fit log(-y) (a copy of the data with y -> -y)
  big      [128, 128, 128], 20000 epochs, early-stopping patience 200,
           lr-plateau patience 20

    uv run python experiments/probes/reference_fix_probe.py --equation feynman_test_20 --variant big
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--equation", required=True)
    p.add_argument("--variant", default="base", help="base | neglog | big | big+neglog")
    p.add_argument("--target", default="", help="log or identity; default: as A2 chooses (both tried)")
    p.add_argument("--config", type=Path, default=Path("configs/feynman_redundant_linear_knee.toml"))
    p.add_argument("--out", type=Path, default=Path("results/feynman_benchmark/ref_probe"))
    a = p.parse_args()

    import h5py
    import numpy as np
    from invariant_generator.adaptive import _train_reference
    from invariant_generator.config import PROJECT_ROOT, load_config
    from invariant_generator.feynman_benchmark import problem_config

    src = PROJECT_ROOT / "data/feynman_redundant/L1"
    base = load_config(a.config)
    base.data.data_dir = src
    parts = a.variant.split("+")
    if "neglog" in parts:
        # Copy with y -> -y so the existing log target applies; X and split unchanged.
        dst = a.out / "data_neglog"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"{a.equation}.json", dst / f"{a.equation}.json")
        with h5py.File(src / f"{a.equation}.h5") as fin, h5py.File(dst / f"{a.equation}.h5", "w") as fout:
            for k in fin:
                fout[k] = -np.asarray(fin[k]) if k == base.data.target_key else np.asarray(fin[k])
        base.data.data_dir = dst
    if "big" in parts:
        base.model.hidden_dims = [128, 128, 128]
        base.train.epochs = 20000
        base.train.early_stopping_patience = 200
        base.train.lr_plateau_patience = 20
    entry = next(e for e in json.loads((src / "manifest.json").read_text()) if e["equation_id"] == a.equation)
    targets = [a.target] if a.target else ["log", "identity"]
    for target in targets:
        base.data.target_transform = target
        config = problem_config(base, entry, results_dir=a.out / a.variant / target / a.equation)
        if config.data.target_transform != target:
            print(f"{a.equation} {a.variant} {target}: log impossible", flush=True)
            continue
        started = time.perf_counter()
        ref = _train_reference(config, "nrmse")
        print(f"{a.equation} {a.variant} {target}: reference train {ref['train_metrics']['nrmse']:.4f} "
              f"val {ref['test_metrics']['nrmse']:.4f}, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
