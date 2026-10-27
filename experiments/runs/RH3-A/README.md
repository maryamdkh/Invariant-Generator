# RH3-A: rotated Hill 3D (6 inputs, all directions), our pipeline

- Goal: the core case. PySR alone struggles on the 6 raw components; the
  pipeline should find n = 1, the invariant sigma:A:sigma with A the rotated
  Hill tensor (21 non-zero Mandel entries), and y proportional to its sqrt.
- Data: `data/hill_ladder_full/rotatedhill_full.hdf` (1000 points, all
  directions), config `config.toml` (= `configs/hill_ladder/rh3.toml`).
- Run (Arrhenius interactive node, 2026-10-27, commit `de42785`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 68 s at loss 1.4e-10 (complexity 4 selected).
- Raw results (not in git): `results/yield_benchmark/hill_ladder/seed42/rh3/`.

## Result

| | |
|---|---|
| Stage 1 | n = 1 (val rmse 8.5e-4), 975 s |
| Stage 2 | gated, one active term: J1 = -1.57933 * I11 (val rmse 1.7e-4), 694 s |
| Stage 3 | y = -0.43385 * J1 (complexity 4, loss 1.4e-10), 68 s |
| Learned A vs true rotated Hill tensor | cosine 0.9999999997; eigenvalues / max 2.5e-5 (true 0), 0.537890, 0.597656, 0.657430, 0.792950, 1 (true 0.537890, 0.597655, 0.657421, 0.792966, 1) |
| Fresh-point relative L2 | 1.05e-5 (test 1.11e-5) |
| Recovered (< 1e-4) | yes |
| Time to recovery | 1689 s (Stages 1-2 1669 s + PySR 20 s) |

## Conclusion

The core case works as on aligned Hill: the rotation is absorbed into the
learned tensor A (same eigenvalues as HILL-A, different eigenvectors), and
PySR sees one variable and needs a 4-node formula. Error floor of the learned
tensor 8.9e-6 (`experiments/outputs/structure_recovery_rh3.json`).
