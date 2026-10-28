# CB-B: CB2001 (2090-T3), PySR alone

- PySR on the 6 raw components (s11, s22, s33, s23, s13, s12), same config
  as CB-A (author's settings + cube/cbrt).
- Run (Arrhenius interactive node, 2026-10-28, commit `35a378d`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/cb/cb2001_2090T3.toml \
          --results-dir results/yield_benchmark/cb/seed42/cb2001_2090T3 \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 697 s at loss 3.6e-4 (complexity 39).

## Result

Selected (complexity 27, loss 4.8e-4):

    y = sqrt((s11 - (s22 + s33)/2)^2 + 4.67463 (s13^2 + s12^2 + s23^2) + (s22 - s33)^2)

| | |
|---|---|
| Fresh / test relative L2 | 2.4e-2 / 2.4e-2 |
| Recovered (< 3.2e-4) | no |

## Conclusion

PySR alone returns a sqrt-of-quadratic (Hill/von Mises-like) surface; the
cubic invariant J3^0 that makes CB2001 asymmetric (tension vs compression)
never appears, apart from a small cbrt(s12) s23 s13 correction in the
longest equations. Loss plateau ~3.6e-4 (2.4% error), four orders of
magnitude above the exact form our pipeline finds at complexity 13 in two
variables (CB-A: 2.9e-4 relative error, loss 8e-8).
