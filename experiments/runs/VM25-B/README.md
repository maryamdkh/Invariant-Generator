# VM25-B: von Mises 25 (all directions), PySR alone

- Goal: ladder rung 1; PySR on the 6 raw stress components is expected to
  solve von Mises.
- Data/config: as VM25-A (`config.toml` = `configs/hill_ladder/vm25.toml`).
- Run (Arrhenius interactive node, 2026-10-26, commit 5d7c710 or later; the
  baseline code did not change after it):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm25.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm25 \
          --arms baseline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 136 s at loss 5.1e-15 (complexity 36).

## Result

    y = sqrt( (0.0282843 (s11 - s33))^2 + (0.0282843 (s33 - s22))^2
            + (0.0282843 (s11 - s22))^2 + 0.0048 (s23^2 + s13^2 + s12^2) )

Fresh-point relative L2 7.4e-8 (test 7.0e-8): **recovered**. This is the
von Mises form: 0.0282843^2 = 0.0008 = (1/2)/25^2 and 0.0048 = 3/25^2, so
y = sigma_eq / 25 with sigma_eq^2 = 1/2 sum (si - sj)^2 + 3 sum tau^2.
The first exact form appeared at complexity 32 (loss 1.9e-13).

## Conclusion

As expected, PySR alone solves von Mises from the raw components, in about
2 minutes (more PySR time than VM25-A's 62 s).
