# RH3-B: rotated Hill 3D (6 inputs, all directions), PySR alone

- Goal: the comparison the paper rests on: PySR on the 6 raw components
  (s11, s22, s33, s23, s13, s12), same PySR settings as RH3-A.
- Run (Arrhenius interactive node, 2026-10-27, commit `7c0822f`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3 \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 1056 s at loss 2.2e-3 (complexity 39).

## Result

Selected equation (complexity 34):

    y = sqrt(sqrt((s33 - s22)^2 + (s11 - s22)^2 + (s11 - s33)^2
                  + (s23^2 + s13^2 + (1.15449 s12)^2) / 0.18211)) - 0.86757

| | |
|---|---|
| Fresh-point relative L2 | 6.0e-2 (test 5.8e-2) |
| Recovered (< 1e-4) | no |

## Conclusion

PySR alone does not find rotated Hill in 6D. The best equations are a
von Mises-like form (equal-weight squared stress differences plus squared
shears, one shear rescaled) with a fourth root and a constant offset to
bend it toward the data; the rotation's cross terms (s11 s23, s13 s12, ...)
never appear. Loss plateau ~2.2e-3 (6% error). Compare RH3-A: 1.1e-5 at
complexity 4, PySR at loss 1e-8 after 20 s.

Contrast within the ladder (seed 42): PySR alone recovers von Mises, is one
constant short on aligned Hill (7e-3) and on rotated Hill 2D (1.5e-4), and
fails on rotated Hill 3D (6e-2). One seed only.
