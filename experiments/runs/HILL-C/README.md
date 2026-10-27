# HILL-C: Hill with aligned axes (all directions), symbolic distillation

- Method as VM25-C (scale-invariant latent L1, RESEARCH_LOG 58).
- Run (Arrhenius interactive node, 2026-10-27, commit `3ae90d6`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/hill.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/hill \
          --arms distillation --seed 42 --numprocs 16 --device cpu

- Network 99 s (train nrmse 2.0e-3); one component kept (z7; next 12%).
  Two PySR searches stopped by hand, 1076 s in total.

## Result

Inner (z7 from stresses; selected loss ~3e-3 on the standardised latent,
complexity at the maxsize 40 cap):

    z7 = 3.31061 - sqrt((s11 - s22)^2 + (s11 - s33)^2 + (0.84487 (s33 - s22))^2
                        + 3.41622 (s13^2 + s23^2) + 4 s12^2)

Outer: y = 1.10202 - 0.33785 z7 (linear).

| | |
|---|---|
| Fresh-point relative L2 | 1.38e-2 (test 1.37e-2) |
| Recovered (< 1e-4) | no |

Inner coefficients vs the true Hill form, normalised to (s11 - s22)^2 = 1:

| term | latent | truth |
|---|---|---|
| (s11 - s22)^2 | 1 | 1 |
| (s33 - s11)^2 | 1 | 0.833 |
| (s22 - s33)^2 | 0.714 | 0.667 |
| s23^2 | 3.42 | 3.00 |
| s13^2 | 3.42 | 3.33 |
| s12^2 | 4 | 3.67 |

## Conclusion

The latent is the Hill form (c - sqrt(quadratic)), and the outer map is
linear, so here the chain itself is not the problem: the error is the inner
PySR search on 6 raw components, which, like PySR alone (HILL-B), runs out
of complexity (40) before it can give every term its own constant (two
coefficients shared, others rounded, 5-20% off). Composed 1.4e-2, better
than VM25-C because the latent happened to be affine in y.
