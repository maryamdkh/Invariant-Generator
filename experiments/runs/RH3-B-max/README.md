# RH3-B-max: rotated Hill 3D, PySR alone with maxsize 150 (cap check)

- Goal: RH3-B used maxsize 40, under which the exact formula on raw
  components (21 quadratic terms, ~143 nodes) cannot be written. Does PySR
  alone succeed when the cap is not binding?
- Config `configs/hill_ladder/rh3_maxsize150.toml` = rh3.toml with
  maxsize 150 and heap_size_hint_gb 6 (first attempt ran out of memory,
  RESEARCH_LOG 64).
- Run (Arrhenius interactive node, srun --mem=128G, 2026-10-27/28, commit
  `6e229e7`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/rh3_maxsize150.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/rh3_maxsize150 \
          --arms baseline --seed 42 --numprocs 8 --device cpu

- PySR stopped by hand after 5502 s (92 min; RH3-B: 1056 s).

## Result

| | RH3-B-max | RH3-B |
|---|---|---|
| maxsize | 150 | 40 |
| PySR time | 5502 s | 1056 s |
| Largest complexity in hall of fame | 80 | 39 |
| Best loss | 7.8e-4 (complexity 80) | 2.2e-3 |
| Selected equation complexity | 42 (loss 9.3e-4) | 34 |
| Fresh relative L2 | 3.3e-2 | 6.0e-2 |
| Test relative L2 | 3.4e-2 | 5.8e-2 |
| Recovered (< 1e-4) | no | no |

Selected equation (complexity 42):

    y = sqrt(((s12 + s22 - s11 + s23)(s12 - s13) + (s33 - s11)^2 + (s11 - s22)^2
              + (s22 - s33)^2) / 12.609 + 0.45065 (s13^2 + s23^2 + s12^2))

## Conclusion

The cap was not what stopped PySR. With 3.75x the room and 5x the time,
the search never went beyond complexity 80 (the exact form needs ~143),
the loss went from 2.2e-3 to 7.8e-4, and the error halved (6% -> 3.3%).
It now picks up a few cross terms, but it is still far from the exact form
and four orders of magnitude from the threshold. Our pipeline: 1.1e-5 at
complexity 4, PySR at loss 1e-8 after 20 s (RH3-A).
