# HILL-B: Hill with aligned axes (all directions), PySR alone

- Goal: PySR on the 6 raw stress components (s11, s22, s33, s23, s13, s12),
  same PySR settings as HILL-A.
- Data / config: as HILL-A (`configs/hill_ladder/hill.toml`).
- Run (Arrhenius interactive node, 2026-10-27, commit `dcb66db`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/hill.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/hill \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 1116 s at loss 4.7e-5 (complexity 38). The
  author had run this search before and seen the loss plateau there.
- Raw results (not in git): `results/yield_benchmark/hill_ladder/seed42/hill/hill_full/baseline_pysr/`.

## Result

    y = sqrt((s12^2 + (0.92787 s23)^2 + s13^2
              + (s33 + 0.44322 (s11 - s22) - s11)^2 / 2.33075
              + ((s22 - s11) / 1.59656)^2) / 2.38535)

| | |
|---|---|
| Fresh-point relative L2 | 8.8e-3 (test 7.4e-3) |
| Recovered (< 1e-4) | no |

Quadratic form of the equation vs the true Hill M (tensor components, so
shear entries should be 2x the Mandel ones): normal-stress block within 0.1%;
s23 coefficient within 0.3%; s13 and s12 share one coefficient (1/2.385),
while the truth has 0.20 and 0.22, i.e. 4.8% too large and 4.7% too small.
That one missing constant is the whole error.

## Conclusion

PySR alone finds the right shape (Hill form: squared stress differences plus
squared shears) but stops one constant short: complexity 38 is needed even
for this form, and a separate coefficient for s12 or s13 would push it
further. Our pipeline gives the exact form at complexity 4 in one invariant
(HILL-A, error 1.1e-5). One seed only: run more seeds before claiming
"PySR alone fails on aligned Hill".
