# SA-B: Synthetic SA (70 problems), PySR alone on the raw inputs, seed 42

- Same PySR config as SA-A (`configs/synthetic_sa.toml`, maxsize 35, 15-minute limit).
- Arrhenius, commit `a930e16`, array of 70 tasks, `KEEP_ARTIFACTS=full`, no failed tasks.
  Results in `results/feynman_benchmark/sa_B/` (gitignored).
- Scored by `experiments/probes/sa_b_compare.py` -> `experiments/outputs/sa_b_score.csv`.

## Result (fresh relative L2; "under" = < 3.2e-4, provisional)

| Group | Ours under | PySR alone under | Median ours | Median PySR alone | Ours better (paired) | Median ratio |
|---|---|---|---|---|---|---|
| all (70) | 36 | 7 | 1.6e-4 | 6.2e-2 | 52/52 | 605x |
| sqrt(q1) (21) | 14 | 6 | 2.7e-5 | 3.5e-2 | 14/14 | 2103x |
| q1/(1+q2) (28) | 13 | 0 | 2.1e-4 | 8.1e-2 | 21/21 | 564x |
| sqrt(q1)+q2 (21) | 9 | 1 | 2.4e-4 | 5.3e-2 | 17/17 | 223x |

| d | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| Ours under (of 10) | 0 | 2 | 6 | 5 | 8 | 7 | 8 |
| PySR alone under (of 10) | 4 | 3 | 0 | 0 | 0 | 0 | 0 |
| PySR alone median | 1.7e-3 | 1.7e-2 | 2.6e-2 | 5.9e-2 | 8.1e-2 | 1.1e-1 | 1.1e-1 |

- Paired = the 52 problems where ours chose an n; the other 18 have no formula from ours.
- All 7 PySR-alone successes are in those 18 (d <= 3). Ours with PySR alone as the
  fallback: 43/70.

## Conclusion

PySR alone degrades steadily with d and never gets under 3.2e-4 from d = 4; ours does
not degrade. Large caveat: maxsize 35 cannot hold the expanded form from d = 4 (10
terms, about 68 nodes by the RH3-B-max estimate of 143 nodes for 21 terms), so part of
the gap is the cap, not the search. A larger-maxsize check is needed before claiming it.
