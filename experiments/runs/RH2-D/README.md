# RH2-D: rotated Hill 2D (all directions), sparse linear (SINDy-style)

- STLSQ on a degree-3 polynomial library (20 terms) for y^2.
- Run locally on the Mac (no PySR, 0.01 s), commit `b229b38`:

      uv run python scripts/run_yield_benchmark.py --config configs/hill_ladder/rh2.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh2 \
          --arms sparse_linear --seed 42 --device cpu

## Result

y^2 = the 6-term plane-stress quadratic (s11^2, s11 s22, s11 s12, s22^2,
s22 s12, s12^2); fresh relative L2 4.4e-11 (test 4.5e-11). Recovered.
Expected: the surface is a quadratic form (see HILL-D).
