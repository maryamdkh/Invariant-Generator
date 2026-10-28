# CB-E: CB2001 (2090-T3), PySR on the feature library, no encoder

- PySR on the 29 orthotropic invariant monomials the encoder reads (F1-F9
  quadratic, F10-F29 cubic), without the encoder's compression to n = 2.
  Same config as the other CB arms.
- Run (Arrhenius interactive node, 2026-10-28, commit `122e479`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/cb/cb2001_2090T3.toml \
          --results-dir results/yield_benchmark/cb/seed42/cb2001_2090T3 \
          --arms library_pysr --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 1962 s at loss 3.9e-4 (complexity 39).

## Result

Selected (complexity 24, loss 5.3e-4):

    y = sqrt(1.22449 (F6 + F4 - F3 - F5) + (F7 + F8 + F9) / 0.21310 + F1 - F2)

i.e. the square root of a quadratic form in the stresses. Fresh / test
relative L2 2.4e-2 / 2.6e-2. Not recovered (< 3.2e-4).

## Conclusion

Given all 29 invariant monomials, PySR uses only the 9 quadratic ones and
never builds the cubic invariant J3^0 (a 20-term linear combination of
F10-F29); it ends at the same 2.4% plateau as PySR alone on raw components
(CB-B). The library alone is not enough: the encoder's step of combining
the monomials into J2^0 and J3^0 (n = 2) is what makes the problem easy
(CB-A: exact form at complexity 13).
