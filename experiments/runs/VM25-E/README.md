# VM25-E: von Mises 25 (all directions), PySR on the feature library, no encoder

- Goal: does the invariant library alone (structural tensors fixed to
  identity, so isotropic: F1 = I1, F2 = I2, F3 = I3, F4 = I13, homogenized)
  make PySR succeed without the encoder?
- Run (Arrhenius interactive node, 2026-10-26, commit 501c5f9 or later, before
  the time-to-recovery change; library code unchanged since):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm25.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm25 \
          --arms library_pysr --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 62 s. Timing to be rerun (author).

## Result

Exact form first at complexity 12 (loss 1.3e-13):

    y = sqrt(0.0024 * F2^2 - 0.0008 * F1^2) = sqrt(0.0024 tr(sigma^2) - 0.0008 (tr sigma)^2)

which is von Mises: sigma_eq^2 = 3/2 tr(sigma^2) - 1/2 (tr sigma)^2, divided by
25^2. PySR's selected equation (complexity 24, the same form plus tiny
constants) has fresh relative L2 1.7e-8: **recovered**.

## Conclusion

For the isotropic von Mises surface the fixed isotropic invariants are
enough; PySR needs only a 12-node formula (vs 32-36 on raw components, VM25-B).
