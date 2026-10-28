# CB-C: CB2001 (2090-T3), symbolic distillation

- Method as VM25-C (scale-invariant latent L1, RESEARCH_LOG 58); same
  config as the other CB arms.
- Run (Arrhenius interactive node, 2026-10-28, commit `86541cb`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/cb/cb2001_2090T3.toml \
          --results-dir results/yield_benchmark/cb/seed42/cb2001_2090T3 \
          --arms distillation --seed 42 --numprocs 16 --device cpu

- Network 195 s (train nrmse 2.8e-3); one latent kept (z7; the next is 2%
  of it). Two PySR searches stopped by hand, 859 s in total.

## Result

Inner (z7 from the 6 components; loss ~0.015-0.02 on the standardised
latent, complexity at the maxsize 40 cap): a quadratic form
(-5.18 shear^2 - (s22 - s33)^2 - (s11 - 0.51 (s22 + s33))^2 + 1.375)
divided by (shear^2 sum)^(1/9), a correction PySR adds to bend it.
Outer: y = cbrt((0.12672 z7 - 1.08923)^6 - 0.32526).

| | |
|---|---|
| Fresh / test relative L2 | 4.1e-2 / 3.9e-2 |
| Recovered (< 3.2e-4) | no |

## Conclusion

The network's latent carries the right information (train nrmse 2.8e-3),
but the inner symbolic search on 6 raw inputs finds only the quadratic
(Hill-like) part, as PySR alone does (CB-B), plus an ad hoc shear factor;
the cubic J3^0 is missing. The outer search cannot fix what the inner one
lost. Composed 3.9e-2, worse than PySR alone (2.4e-2). Our encoder gives
J2^0 and J3^0 directly (CB-A, exact form, 2.9e-4).
