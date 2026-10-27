# RH3-C: rotated Hill 3D (all directions), symbolic distillation

- Method as VM25-C (scale-invariant latent L1, RESEARCH_LOG 58).
- Run (Arrhenius interactive node, 2026-10-27, commit `1079f13`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3 \
          --arms distillation --seed 42 --numprocs 16 --device cpu

- Network 99 s (train nrmse 2.2e-3); one component kept (z7; next 7%).
  Two PySR searches stopped by hand, 1730 s in total.

## Result

Inner (z7 from the 6 components; loss ~0.025-0.03 on the standardised
latent, complexity at the maxsize 40 cap):

    z7 = 3.18572 - 1.14597 sqrt(3.71010 s12^2 + 2.71010 (s13^2 + s23^2)
                                 + (s33 - s11)^2 - (s11 - s22)(s22 - s33))

Outer: y = sqrt((0.24763 z7 - 1.48609)^2 - 0.94486).

| | |
|---|---|
| Fresh-point relative L2 | 5.70e-2 (test 5.39e-2) |
| Recovered (< 1e-4) | no |

## Conclusion

The latent search on 6 raw inputs fails the same way PySR alone does
(RH3-B): it returns a near-isotropic Hill-like form (two shears share a
constant, normal terms nearly equal) without the rotation's cross terms, so
the composed formula has the same ~5.5% error as B and E. The network's
latent itself is fine (it predicts y to 2.2e-3 nrmse); the symbolic step
on 6 inputs is what fails. Our pipeline replaces that step with a learned
tensor (RH3-A, 1.1e-5).
