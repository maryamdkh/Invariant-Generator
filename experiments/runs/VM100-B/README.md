# VM100-B: von Mises 100 (all directions), PySR alone

- Goal: scale check for PySR alone (VM25-B recovered von Mises 25).
- Run (Arrhenius interactive node, 2026-10-26, commit `9248eab`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm100.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm100 \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 222 s at loss 3.9e-12.

## Result

    y = sqrt( [ ((s33 - s11)/1.7320542)^2 + s23^2 + s13^2 + s12^2
              + 0.3333320 (s22 - s11)(s22 - s33) ] / 3333.37018 ) + 5.5e-6

This is the exact von Mises form: with 1.7320542 -> sqrt(3), 0.3333320 -> 1/3
and 3333.37 -> 10000/3, y^2 = [(s11 - s33)^2 + (s22 - s11)(s22 - s33)
+ 3 tau^2] / 100^2 = sigma_eq^2 / 100^2. The fitted constants are off by
~1e-5 relative, so fresh relative L2 is 2.2e-6 (test 2.3e-6): **not
recovered** under the 1e-6 threshold, although the structure is exact.

Time to recovery: none (final loss 3.9e-12 > 1e-12; the stop-rule marker
fired on a minibatch loss and was ignored).

## Conclusion

PySR alone finds the right formula at scale 100 too, but in 222 s its
constants were not yet precise enough for the strict threshold (at scale 25
it reached 7e-8 in 136 s). Like VM25-A/VM100-A, "right structure, error just
above 1e-6": the strict numerical threshold alone does not separate these
cases well (RESEARCH_LOG 48).
