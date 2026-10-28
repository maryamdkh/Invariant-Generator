# CB-D: CB2001 (2090-T3), sparse linear (SINDy-style)

- STLSQ for y^m on a polynomial library of the 6 stress components;
  m = 1..6 with library degree max(3, m) (`--sparse-max-power 6`); (m,
  threshold) chosen on a 10% holdout.
- Run locally on the Mac (no PySR, ~1 s), 2026-10-28:

      uv run python scripts/run_yield_benchmark.py --config configs/cb/cb2001_2090T3.toml \
          --results-dir results/yield_benchmark/cb/seed42/cb2001_2090T3 \
          --arms sparse_linear --seed 42 --device cpu --sparse-max-power 6

## Result

m = 6, degree-6 library (924 terms), 138 active terms; fresh relative L2
8.1e-9 (test 1.6e-9). Recovered.

CB2001 is y^6 proportional to J2^0^3 - c J3^0^2, a homogeneous degree-6
polynomial in sigma, so a linear fit on degree-6 monomials is exact. With
the previous default (m <= 3) D gave 2.1e-2; restricting it would be unfair.

Bug found on the way: `_inverse_power` took a square root for every even
m, so m = 4, 6 fits were inverted wrongly. m <= 3 (all Hill-ladder D runs,
the Feynman D arm) was unaffected.
