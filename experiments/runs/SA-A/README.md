# SA-A: Synthetic SA (70 rotated quadratic-form problems), ours, seed 42

- Stage 1 + graded Stage 2 + PySR on the encoded variables, `configs/synthetic_sa.toml`.
- Arrhenius job `4669470`, commit `44f44db`, 2026-10-30, 15-minute PySR limit,
  `KEEP_ARTIFACTS=full`. Results rsynced to `results/feynman_benchmark/sa_A/` (gitignored).
- Scored by `experiments/probes/sa_a_score.py` -> `experiments/outputs/sa_a_score.csv`.
- "Recovered" = fresh relative L2 < 3.2e-4, the yield threshold; provisional for this
  benchmark. No problem passed the symbolic check (19 false, 29 SymPy time-outs).

## Result

| Group | Problems | n chosen | n correct | Under 3.2e-4 | Median fresh error (n chosen) |
|---|---|---|---|---|---|
| all | 70 | 52 | 52 | 36 | 1.6e-4 |
| sqrt(q1) | 21 | 14 | 14 | 14 | 2.7e-5 |
| q1/(1+q2) | 28 | 21 | 21 | 13 | 2.1e-4 |
| sqrt(q1)+q2 | 21 | 17 | 17 | 9 | 2.4e-4 |

| d | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| n chosen | 0 | 6 | 10 | 9 | 9 | 8 | 10 |
| under 3.2e-4 | 0 | 2 | 6 | 5 | 8 | 7 | 8 |

- 18 problems have no accepted n. 7 are correct (d = 2 with true n = 2, nothing to
  reduce); 11 are false rejections: validation or test NRMSE just above 5e-3 at the
  true n (3 at d = 2, 4 at d = 3, sa_d06_03, sa_d08_02, sa_d10_03, sa_d10_04).
- 4 tasks have no `benchmark_summary.json` (sa_d06_04, sa_d06_05, sa_d10_06,
  sa_d12_04): PySR finished, the task ended during scoring. Probably the 1.5 h limit
  (logs not yet read). Rescored offline from the saved equation; the same code
  reproduces a stored score (sa_d12_03, 2.916606e-4).

## Conclusion

Never a wrong n, and accuracy does not fall with d. Weak points: the 5e-3 acceptance
rejects 11 reducible problems, and errors are small but the formulas are not the
true symbolic form. One arm only: no comparison claim before SA-B, C, D, E.
