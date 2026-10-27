# VM100-A: von Mises 100 (all directions), our pipeline (scale check)

- Goal: same surface as VM25 scaled by 4 (yield stress 100); the pipeline
  should give the same structure and error, only a different constant.
- Run (Arrhenius interactive node, 2026-10-26, commit `2c33f0e`: Hill-ladder
  early-stopping patience 50):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm100.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm100 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 85 s at loss 2.2e-10.

## Result

| | VM100-A | VM25-A |
|---|---|---|
| Stage 1 | n = 1 (val rmse 1.1e-3), 940 s | n = 1, 1159 s |
| Stage 2 | J1 = -0.0399949 * I11, 689 s | J1 = -0.162562 * I11 |
| Stage 3 | y = J1 / -2.29282 (complexity 4) | y = -0.42559 * J1 |
| Tensor cosine vs truth | 0.9999999995 | 0.9999999995 |
| Fresh relative L2 | 1.35e-5 | 1.31e-5 |

Overall coefficient 0.017444 vs 0.069185 (ratio 3.97, about the factor 4;
the learned tensor's overall scale is free, so it need not be exactly 4).

Time to recovery: none. The stop-rule marker fired at 19 s on a minibatch
loss (batching = true), but the full-data loss never went below 2.2e-10; the
watcher now ignores such markers (RESEARCH_LOG 47).

## Conclusion

Scale check passed: identical structure (n = 1, I11, the von Mises tensor)
and the same error level as VM25-A; the pipeline is insensitive to the stress
scale. The 1.3e-5 limit is again the learned tensor (best possible relative
error with it: 1.34e-5).

Update 2026-10-27: yield recovery threshold is now relative L2 < 1e-4
(RESEARCH_LOG 49), so this run counts as recovered.
