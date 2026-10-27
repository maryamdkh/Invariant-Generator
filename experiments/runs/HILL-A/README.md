# HILL-A: Hill with aligned axes (all directions), our pipeline

- Goal: the pipeline should find n = 1, the single invariant sigma:A:sigma
  with A the Hill tensor (6 eigenvalues: one zero, five distinct), and y
  proportional to sqrt of it.
- Data: `data/hill_ladder_full/hill_full.hdf` (1000 points, all directions),
  config `config.toml` (= `configs/hill_ladder/hill.toml`, patience 50).
- Run (Arrhenius interactive node, 2026-10-27, commit `1b22c97`, assumed:
  latest at the time; first attempt died with a lost connection, this is the
  rerun):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/hill.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/hill \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 243 s at loss 1.6e-10 (complexity 4).
- Raw results (not in git): `results/yield_benchmark/hill_ladder/seed42/hill/`.

## Result

| | |
|---|---|
| Stage 1 | n = 1 (val rmse 1.0e-3), 964 s |
| Stage 2 | gated, one active term: J1 = -1.58264 * I11 (val rmse 5.5e-4), 694 s |
| Stage 3 | y = -0.43579 * J1 (complexity 4, loss 1.6e-10), 243 s |
| Learned A vs true Hill tensor | cosine 0.9999999996; eigenvalues / max 2.6e-5 (true 0), 0.53790, 0.59765, 0.65742, 0.79295, 1 (true 0.53789, 0.59766, 0.65742, 0.79297, 1) |
| Fresh-point relative L2 | 1.14e-5 (test 1.30e-5) |
| Recovered (< 1e-4) | yes |

## Conclusion

Same picture as VM25-A, now with an anisotropic tensor: one encoded
variable, the right invariant and all five non-zero Hill eigenvalues to 4-5
digits. The error floor of c * sqrt(sigma:A_learned:sigma) is 9.5e-6 on
random directions (`experiments/outputs/structure_recovery_hill.json`), so
the 1.1e-5 is the learned tensor's float32 precision, not PySR. Longer
equations lower the loss only in the 4th digit (1.62e-10).
