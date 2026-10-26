# VM25-D: von Mises 25 (all directions), sparse linear regression (SINDy-style)

- Goal: baseline D; expected to solve every quadratic yield surface.
- Run locally on the Mac (no PySR, 0.03 s), commit `bcd8e18`:

      uv run python scripts/run_yield_benchmark.py --config configs/hill_ladder/vm25.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm25 \
          --arms sparse_linear --seed 42 --device cpu

## Result

y^2 = 0.0016 (s11^2 + s22^2 + s33^2 - s11 s22 - s11 s33 - s22 s33)
      + 0.0048 (s23^2 + s13^2 + s12^2), i.e. 9 of 84 degree-<=3 terms, power
m = 2 chosen on the holdout. Fresh relative L2 2.9e-11: **recovered**.

## Conclusion

As expected: for a quadratic yield function, regressing y^2 linearly on
quadratic monomials is exact.
