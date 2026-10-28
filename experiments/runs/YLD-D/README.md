# YLD-D: Yld2004-18p (2090-T3, a = 8), sparse linear (SINDy-style)

- STLSQ for y^m, m = 1..8, library degree max(3, m) (`--sparse-max-power 8`).
- Run locally on the Mac (no PySR, ~26 s fit), 2026-10-28:

      uv run python scripts/run_yield_benchmark.py --config configs/yld/yld2004_2090T3.toml \
          --results-dir results/yield_benchmark/yld/seed42/yld2004_2090T3 \
          --arms sparse_linear --seed 42 --device cpu --sparse-max-power 8

## Result

m = 8, degree-8 library (3003 terms), 363 active terms; fresh relative L2
9.0e-7 (test 2.6e-7). Recovered.

Why: with the integer exponent a = 8, sum_ij |S'_i - S''_j|^8 =
sum_ij (S'_i - S''_j)^8 = sum_k C(8,k) (-1)^k tr(X'^(8-k)) tr(X''^k), and
traces of powers of the linearly transformed deviators X', X'' are
polynomials in sigma. So Yld2004 at a = 8 is a degree-8 polynomial and a
linear fit on degree-8 monomials is exact (up to conditioning).
