# HILL-E: Hill with aligned axes (all directions), PySR on the feature library, no encoder

- Goal: does the invariant library alone (structural tensors fixed to
  identity: F1 = I1, F2 = I2, F3 = I3, F4 = I13, homogenized) make PySR
  succeed without the encoder and without learned structure?
- Run (Arrhenius interactive node, 2026-10-27, commit `733a827`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/hill.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/hill \
          --arms library_pysr --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 1330 s at loss 3.1e-3 (complexity 39).

## Result

Selected equation (complexity 12): y = -0.09389 F1^2 / F4 + F2 / 2.03189;
fresh relative L2 6.1e-2 (test 5.8e-2). Not recovered. Longer equations
(up to complexity 39) only lowered the loss from 3.6e-3 to 3.1e-3.

## Conclusion

Expected, and the reason it matters: with identity tensors every feature is
an isotropic invariant (unchanged by any rotation of sigma), while aligned
Hill is anisotropic. No formula in F1-F4 can represent it, however long, so
the loss plateaus at ~3e-3 (6% error). The encoder succeeds (HILL-A) because
it learns the structural tensor A; the fixed library does not. This is the
ablation for contribution 2 (learnable physical structure). On von Mises,
which is isotropic, the same arm recovered (VM25-E).
