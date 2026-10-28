# YLD-A: Yld2004-18p (2090-T3, a = 8), our pipeline (stress test)

- Same feature set as CB-A: the 29 orthotropic monomials of degree 2 and 3.
  Yld2004 with a = 8 is a degree-8 polynomial in sigma (RESEARCH_LOG 72),
  so the features cannot express it exactly; this run tests what the
  pipeline does when its library does not contain the answer.
- Run (Arrhenius interactive node, 2026-10-28, commit `2bd3f48`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/yld/yld2004_2090T3.toml \
          --results-dir results/yield_benchmark/yld/seed42/yld2004_2090T3 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 317 s at loss 6.4e-4.

## Result

| | |
|---|---|
| Stage 1 reference (no bottleneck, 29 features) | test nrmse 5.8e-2 |
| Stage 1 | n = 1 accepted (nrmse 9.7e-2 < 2 x reference), 325 s |
| Stage 2 | graded: all 29 terms kept (quadratic part dominant, cubic ~1-4%) |
| Stage 3 | y = sqrt(J1) / 1.64292 (complexity 5); longer formulas only to loss 6.4e-4 |
| Fresh / test relative L2 | 2.94e-2 / 2.88e-2 |
| Recovered (< 3.2e-4) | no |

Best polynomial approximation of y^m by degree
(`experiments/probes/yld_polynomial_floor.py`,
`experiments/outputs/yld_polynomial_floor.json`):

| degree | terms | fresh relative L2 |
|---|---|---|
| 2 | 28 | 2.91e-2 |
| 3 | 84 | 1.56e-1 |
| 4 | 210 | 1.58e-2 |
| 6 | 924 | 8.3e-3 |
| 8 | 3003 | 4.1e-11 |

## Conclusion

The pipeline returns sqrt(quadratic form), a Hill-type surface, at 2.94e-2:
exactly the best quadratic approximation of Yld2004 (2.91e-2). It does what
its library allows and no more. The answer needs degree-8 terms (or the
principal values of two linear transforms of the deviator), which the
degree-2/3 orthotropic features do not contain; even degree 6 only reaches
8e-3. The Stage 1 reference model on these features was itself weak
(5.8e-2), so the relative acceptance rule accepted n = 1. Expected
outcome for a stress test; it marks the limit of the current library, not
of the encoder idea.
