# RH2-C: rotated Hill 2D (all directions), symbolic distillation

- Method as VM25-C (scale-invariant latent L1, RESEARCH_LOG 58).
- Run (Arrhenius interactive node, 2026-10-27, commit `9274664`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh2.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh2 \
          --arms distillation --seed 42 --numprocs 16 --device cpu

- Network 99 s (train nrmse 3.7e-3); one component kept (z0; next 13%).
  Two PySR searches stopped by hand, 2331 s in total.

## Result

Inner (z0 from s11, s22, s12; loss 1.2e-3 on the standardised latent):

    z0 = (s11^2 + s22 (s22 - s11 + 0.54168 s12) + 3.68853 (s12 - 0.05760 s11)^2) / 4.89992 - 1.72792

Outer: y = sqrt(z0 + 2.12346) - 0.31521.

| | |
|---|---|
| Fresh-point relative L2 | 1.15e-2 (test 1.10e-2) |
| Recovered (< 1e-4) | no |

Inner quadratic vs truth (normalised to s22^2 = 1):

| term | latent | truth |
|---|---|---|
| s11^2 | 1.012 | 1.001 |
| s11 s22 | -1 | -0.981 |
| s11 s12 | -0.425 | -0.428 |
| s22 s12 | 0.542 | 0.557 |
| s12^2 | 3.689 | 3.661 |

## Conclusion

The latent is the rotated Hill quadratic (affine in y^2) with coefficients
0.6-2.7% off (s11 s22 fixed at -1, i.e. one constant short again), and the
outer map has the right sqrt shape but its constants do not match the
latent's offset exactly (2.123 vs 1.728, plus -0.315). Both steps add error:
1.1e-2 composed, vs 1.1e-6 for our pipeline (RH2-A) and 1.5e-4 for PySR
alone (RH2-B) on the same 3 inputs.
