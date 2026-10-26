# VM25-A: von Mises 25 (all directions), our pipeline

- Goal: control case. PySR alone is expected to solve von Mises; the
  pipeline should find n = 1, the single invariant sigma:A:sigma with A the
  deviatoric projector, and y proportional to it.
- Data: `data/hill_ladder_full/vonMises25_full.hdf` (1000 points, all
  directions), config `config.toml` (= `configs/hill_ladder/vm25.toml`).
- Run (Arrhenius, interactive GPU node, CPU only, 2026-10-26, commit
  `32a258a`, assumed: latest at the time of the run):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm25.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm25 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR (Stage 3) stopped by hand after ~1 min: loss 2.1e-10 at complexity 4
  (y = c * J1); longer equations only lowered the loss to 2.11e-10.
- Raw results (not in git): `results/yield_benchmark/hill_ladder/seed42/vm25/`.

## Result

| | |
|---|---|
| Stage 1 | n = 1 (val rmse 1.3e-3), 1159 s |
| Stage 2 | gated, one active term: J1 = -0.162562 * I11 (val rmse 1.5e-4), 689 s |
| Stage 3 | y = -0.42559 * J1 (complexity 4), 62 s |
| Learned A vs true tensor | cosine 0.9999999995; eigenvalues / max: 5e-5 (true 0), 0.99994-1.0 (true 1) |
| Fresh-point relative L2 | 1.31e-5 (test 1.34e-5) |
| Recovered (< 1e-6) | no |

## Conclusion

Structure fully right: one encoded variable, one invariant (I11), learned
tensor equal to the von Mises tensor up to cosine 5e-10, and PySR picks the
exact form y = c * sqrt(sigma:A:sigma). The 1.3e-5 error is the learned
tensor's own error (hydrostatic eigenvalue 5e-5 instead of 0, deviatoric
eigenvalues within 6e-5 of each other): sqrt(sigma:A_learned:sigma) differs
from the truth by 1.30e-5 for any constant, so stopping PySR earlier or later
does not matter. The 1e-6 threshold (taken from the Feynman benchmark, where
exponents are snapped to exact rationals) cannot be met by a tensor learned
in float32 without a final refit; see RESEARCH_LOG 42.
