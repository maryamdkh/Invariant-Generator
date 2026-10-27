# RH3-E: rotated Hill 3D (all directions), PySR on the feature library, no encoder

- Goal: as HILL-E / RH2-E: fixed isotropic library (identity tensors:
  F1 = I1, F2 = I2, F3 = I3, F4 = I13, homogenized), no encoder.
- Run (Arrhenius interactive node, 2026-10-27, commit `1618c9b`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3 \
          --arms library_pysr --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 1086 s at loss 2.8e-3 (complexity 39).

## Result

Selected equation (complexity 12, loss 3.0e-3):

    y = sqrt(-0.07849 F1^2 + F2^2 / 4.25102)

i.e. a von Mises form, the best isotropic fit to the surface. Fresh
relative L2 6.2e-2 (test 5.7e-2). Not recovered; complexity 12-39 all
between loss 2.8e-3 and 3.0e-3.

## Conclusion

As for HILL-E and RH2-E: isotropic features cannot express an anisotropic
surface. Here PySR returns the closest von Mises surface, with the same 6%
error as PySR alone on raw components (RH3-B), which converged to a
von Mises-like form as well. The learned tensor (RH3-A) is what closes the
gap: 1.1e-5.
