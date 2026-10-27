# RH3-D: rotated Hill 3D (all directions), sparse linear (SINDy-style)

- STLSQ on a degree-3 polynomial library (84 terms) for y^2.
- Run locally on the Mac (no PySR, 0.03 s), commit `de42785`:

      uv run python scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3 \
          --arms sparse_linear --seed 42 --device cpu

## Result

y^2 = the full 21-term quadratic in the 6 components; fresh relative L2
3.4e-11 (test 4.1e-11). Recovered.

Important for the paper: rotated Hill is still a quadratic form, so once y
is squared it is linear in 21 known monomials and a linear fit is exact,
even where PySR alone struggles. The case for the encoder over sparse linear
regression rests on surfaces that are not polynomial in sigma (CB2001,
Yld2004, CPFEM) and on the interpretable output (one tensor A, one
invariant) rather than on the Hill ladder (RESEARCH_LOG 40).
