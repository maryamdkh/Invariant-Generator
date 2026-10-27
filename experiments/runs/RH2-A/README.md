# RH2-A: rotated Hill, plane stress (3 inputs, all directions), our pipeline

- Goal: n = 1, the single invariant sigma:A:sigma with A the rotated Hill
  tensor (plane-stress block), y proportional to sqrt of it.
- Data: `data/hill_ladder_full/rotatedhill2d_full.hdf` (1000 points, inputs
  s11, s22, s12), config `config.toml` (= `configs/hill_ladder/rh2.toml`).
- Run (Arrhenius interactive node, 2026-10-27, commit `22e0d10`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh2.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh2 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 54 s at loss 1.2e-12 (complexity 4).
- Raw results (not in git): `results/yield_benchmark/hill_ladder/seed42/rh2/`.

## Result

| | |
|---|---|
| Stage 1 | n = 1 (val rmse 1.4e-3), 591 s |
| Stage 2 | gated, one active term: J1 = -1.19932 * I11 (val rmse 2.2e-4), 694 s |
| Stage 3 | y = J1 / -1.74720 (complexity 4, loss 1.2e-12), 54 s |
| Learned plane-stress block vs truth | cosine 0.99999999999; eigenvalues / max 0.259761, 0.694907, 1 (true 0.259762, 0.694911, 1) |
| Fresh-point relative L2 | 1.18e-6 (test 1.07e-6) |
| Recovered (< 1e-4) | yes |
| Time to recovery | 1306 s (Stages 1-2 1285 s + PySR 20 s) |

## Conclusion

Exact structure, and ten times more precise than the 3D runs (1.2e-6 vs
1.1-1.3e-5): with 3 inputs only a 3x3 block of A has to be learned. First
Hill-ladder run with a recorded time to recovery; nearly all of it is the
encoder (Stages 1-2), PySR needs 20 s.
