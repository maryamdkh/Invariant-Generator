# HILL-D: Hill with aligned axes (all directions), sparse linear (SINDy-style)

- STLSQ on a degree-3 polynomial library (84 terms) for y^p, p chosen from
  the homogeneity degree (p = 2).
- Run locally on the Mac (no PySR, 0.03 s), commit `dcb66db`:

      uv run python scripts/run_yield_benchmark.py --config configs/hill_ladder/hill.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/hill \
          --arms sparse_linear --seed 42 --device cpu

## Result

y^2 = the 9-term Hill quadratic, coefficients exact to ~1e-10
(0.22, -0.24, -0.20, 0.20, -0.16, 0.18, 0.36, 0.40, 0.44); fresh relative L2
3.8e-11 (test 4.7e-11). Recovered.

As for VM25-D: every Hill-ladder surface is a quadratic form in sigma, so
y^2 is linear in the quadratic monomials and a linear fit is exact. This arm
does not support "needs a nonlinear search" on the Hill ladder; that argument
rests on CB2001, the synthetic sets and CPFEM data (RESEARCH_LOG 40).
