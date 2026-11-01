# SA-D: Synthetic SA (70 problems), sparse linear, seed 42

- STLSQ: y^m = polynomial of the raw inputs, degree 3, m = 1, 2, 3 (same arm as all D runs).
- Run on the Mac, commit `478d0f3`, all 70 problems, no errors:

      uv run python scripts/run_feynman_benchmark.py --config configs/synthetic_sa.toml \
          --manifest data/synthetic/sa/manifest.json --data-dir data/synthetic/sa \
          --results-dir results/feynman_benchmark/sa_D/seed42 --arms sparse_linear \
          --timeout-minutes 15 --seed 42 --numprocs 8

- Scored by `experiments/probes/sa_d_compare.py` -> `experiments/outputs/sa_d_score.csv`.
- 6 problems had a NaN fresh score (the polynomial under the sqrt is negative at 2-724 of
  10000 points). They are rescored with the clipping the test score already uses. No
  shared code changed. All 6 stay far above the threshold.

## Result (fresh relative L2; "under" = < 3.2e-4, provisional)

| Group | Ours under | PySR alone under | Sparse linear under | Median ours | Median sparse linear | Ours better (paired) |
|---|---|---|---|---|---|---|
| all (70) | 36 | 7 | 21 | 1.6e-4 | 1.9e-2 | 38/52 |
| sqrt(q1) (21) | 14 | 6 | 21 | 2.7e-5 | 1.5e-11 | 0/14 |
| q1/(1+q2) (28) | 13 | 0 | 0 | 2.1e-4 | 5.6e-2 | 21/21 |
| sqrt(q1)+q2 (21) | 9 | 1 | 0 | 2.4e-4 | 1.1e-2 | 17/17 |

- sqrt(q1): exact, symbolic check true on all 21, fit under 0.5 s. y^2 is polynomial by design.
- The two non-polynomial forms: 0/49 for sparse linear, 22/49 for ours; ours is more
  accurate on all 38 paired problems (median 235x and 38x).

## Conclusion

Sparse linear wins whenever y^m is polynomial and fails otherwise; ours is the only
method here that does well on both. Caveat: an implicit library (SINDy-PI) could write
q1/(1+q2) as y + y q2 = q1; it was not tested.
