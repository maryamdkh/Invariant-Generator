# RH2-B: rotated Hill 2D (all directions), PySR alone

- Goal: PySR on the 3 raw plane-stress components (s11, s22, s12), same
  PySR settings as RH2-A. Control case: the author has seen PySR alone
  solve this before.
- Run (Arrhenius interactive node, 2026-10-27, commit `def3529`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh2.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh2 \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 468 s at loss 2.1e-8 (complexity 32).

## Result

    y = sqrt(((s11 - 0.21356 s12)^2 + (1.90071 s12)^2
              - 0.98065 s22 (s11 - s12 / 1.76030) + s22^2) / 6.34718)

| | |
|---|---|
| Fresh-point relative L2 | 1.39e-4 (test 1.52e-4) |
| Recovered (< 1e-4) | no, just above |

Quadratic form of the equation vs the truth, entry by entry: all off-diagonal
and s12^2 entries within 0.15%; s11^2 and s22^2 are forced equal (both
coefficient 1), while the truth differs by 0.12% (0.157646 vs 0.157452).
That missing constant is the error, as in HILL-B.

## Conclusion

PySR alone finds the right shape, one constant short of exact; a borderline
case: the loss (2.1e-8) is below the 1e-7 "found" rule of thumb but the
relative error is just above our 1e-4 threshold. Our pipeline reached
1.1e-6 at complexity 4 (RH2-A). Several seeds needed: another seed may
well cross the threshold.
