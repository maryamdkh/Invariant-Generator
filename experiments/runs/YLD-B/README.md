# YLD-B: Yld2004-18p (2090-T3, a = 8), PySR alone

- PySR on the 6 raw components (s11, s22, s33, s23, s13, s12), same config
  as YLD-A (author's settings + cube/cbrt).
- Run on the Mac (Arrhenius GPU queue full, project FairShare 0.34;
  the arm uses only the CPU), 2026-10-29, commit `6aee578`, 8 processes
  instead of 16 (search settings unchanged; timing not comparable):

      uv run python scripts/run_yield_benchmark.py --config configs/yld/yld2004_2090T3.toml \
          --results-dir results/yield_benchmark/yld/seed42/yld2004_2090T3 \
          --arms baseline --seed 42 --numprocs 8 --device cpu

  A first attempt was killed after 7 min when its terminal closed; it was
  deleted and rerun from scratch in a detached `screen` session.
- PySR stopped by hand after 1450 s at loss 8.3e-4 (complexity 38, plateau).

## Result

Selected (complexity 27, loss 9.8e-4):

    y = sqrt((s11 - 0.5049 (s22 + s33))^2 + (s22 - s33)^2 + 4.15346 (s12^2 + s23^2 + s13^2))

| | |
|---|---|
| Fresh / test relative L2 | 3.4e-2 / 3.4e-2 |
| Best loss in hall of fame | 8.3e-4 (~2.9e-2 relative), complexity 38 |
| Recovered (< 3.2e-4) | no |

## Conclusion

PySR alone finds a sqrt-of-quadratic (von Mises/Hill-type) surface, and its
best loss (8.3e-4, ~2.9% error) equals the best quadratic approximation of
Yld2004 (2.91e-2, `experiments/outputs/yld_polynomial_floor.json`). The
longest equations add only small s12^4 corrections. Same plateau as YLD-A
(2.94e-2): on Yld2004 neither PySR alone nor our pipeline gets past the
quadratic; only sparse linear on a degree-8 library (YLD-D) is exact.
