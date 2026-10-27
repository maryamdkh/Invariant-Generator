# RH2-E: rotated Hill 2D (all directions), PySR on the feature library, no encoder

- Goal: as HILL-E: fixed isotropic library (identity tensors: F1 = I1,
  F2 = I2, F3 = I3, F4 = I13, homogenized), no encoder.
- Run (Arrhenius interactive node, 2026-10-27, commit `4e4e007`, assumed):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh2.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh2 \
          --arms library_pysr --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 506 s (no improvement), loss 2.5e-3
  (complexity 34).

## Result

Selected equation (complexity 8, loss 3.7e-3): y = F4 / (F1/F3 + 1.41374); fresh relative L2 6.8e-2
(test 6.6e-2). Not recovered. Complexity 17-34 all sit at loss 2.48-2.51e-3.

## Conclusion

Same as HILL-E: isotropic features cannot express an anisotropic surface,
so the loss plateaus at ~2.5e-3 (7% error) however long the formula. The
learned tensor is what makes RH2-A work (1.1e-6).
