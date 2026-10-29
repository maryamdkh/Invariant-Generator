# YLD-C: Yld2004-18p (2090-T3, a = 8), symbolic distillation

- Cranmer-style distillation (scale-invariant latent L1, keep_fraction 0.2),
  same PySR config as YLD-A/B for both searches (latent from the 6 raw
  inputs, then y from the kept latents).
- Run on the Mac (Arrhenius GPU queue full), 2026-10-29, commit `010d585`,
  8 processes, in a detached `screen` session:

      uv run python scripts/run_yield_benchmark.py --config configs/yld/yld2004_2090T3.toml \
          --results-dir results/yield_benchmark/yld/seed42/yld2004_2090T3 \
          --arms distillation --seed 42 --numprocs 8 --device cpu

- Both PySR searches stopped by hand (latent search at loss 2.7e-2 on the
  standardised latent, plateau); 3907 s of PySR in total.

## Result

| | |
|---|---|
| Network | train nrmse 2.5e-3, 44 s |
| Latents kept | 1 of 8 (z7; importance 3.6e-2, next 7.4e-4) |
| Latent equation | z7 = (1.406 - (s11 - s22)^2 - (s33 - 0.592(s11+s22))^2 - 4.662(s12^2+s13^2+s23^2)) / (s11^2+s13^2+s23^2)^(1/9) |
| Outer equation | y = 1.10408 - 0.33250 z7 - 0.00324 z7^2 |
| Fresh / test relative L2 | 4.8e-2 / 4.4e-2 |
| Recovered (< 3.2e-4) | no |

## Conclusion

The network fits Yld2004 well (2.5e-3, an order of magnitude below the
best quadratic, 2.9e-2), but the symbolic step loses it: the latent
equation is again a Hill-type quadratic with an ad hoc (..)^(1/9) divisor,
as in CB-C, and the composed formula (4.8e-2) is worse than PySR alone
(YLD-B, 3.4e-2) and our pipeline (YLD-A, 2.9e-2).
