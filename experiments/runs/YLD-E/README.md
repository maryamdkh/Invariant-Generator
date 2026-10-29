# YLD-E: Yld2004-18p (2090-T3, a = 8), PySR on the fixed library (no encoder)

- PySR on the same 29 orthotropic monomials of degree 2 and 3 (F1-F29) that
  feed our encoder in YLD-A, without the encoder; same PySR config.
- Run on the Mac (Arrhenius GPU queue full), 2026-10-29, commit `d9a9954`,
  8 processes, `screen` session `ylde`:

      uv run python scripts/run_yield_benchmark.py --config configs/yld/yld2004_2090T3.toml \
          --results-dir results/yield_benchmark/yld/seed42/yld2004_2090T3 \
          --arms library_pysr --seed 42 --numprocs 8 --device cpu

- PySR stopped by hand after 1305 s at loss 8.1e-4 (plateau).

## Result

Selected (complexity 24, loss 9.9e-4), F1-F9 the quadratic monomials:

    y = sqrt((F1 + F4 + F6 - F3 - F5)/0.87414 - F2 + (F7 + F8 + F9)/0.24415)

| | |
|---|---|
| Fresh / test relative L2 | 3.4e-2 / 3.2e-2 |
| Best loss in hall of fame | 8.1e-4 (~2.9e-2 relative) |
| Recovered (< 3.2e-4) | no |

## Conclusion

Uses only the quadratic monomials; the cubic ones never help. Same plateau
as YLD-A/B (best quadratic approximation, 2.91e-2): the library, with or
without the encoder, does not contain the degree-8 answer.
