# SA-E: Synthetic SA (70 problems), PySR on the fixed term list, no encoder, seed 42

- PySR searches over the raw inputs and all their degree-2 products (5 terms at d = 2, 90 at
  d = 12), with no encoder. Same PySR config and 15-minute limit as SA-A and SA-B.
- Arrhenius, array of 70 tasks, `KEEP_ARTIFACTS=full`, no failed tasks. Branch head at
  submission was `9030a1b`; the commit was not recorded on the cluster. Results in
  `results/feynman_benchmark/sa_E/` (gitignored).
- Scored by `experiments/probes/sa_e_compare.py` -> `experiments/outputs/sa_e_score.csv`.

## Result (fresh relative L2; "under" = < 3.2e-4, provisional)

| Group | Ours | PySR alone | Sparse linear | Fixed terms | Median ours | Median fixed terms | Ours better (paired) | Median ratio |
|---|---|---|---|---|---|---|---|---|
| all (70) | 36 | 7 | 21 | 8 | 1.6e-4 | 4.6e-2 | 52/52 | 453x |
| sqrt(q1) (21) | 14 | 6 | 21 | 6 | 2.7e-5 | 2.0e-2 | 14/14 | 1493x |
| q1/(1+q2) (28) | 13 | 0 | 0 | 0 | 2.1e-4 | 8.6e-2 | 21/21 | 454x |
| sqrt(q1)+q2 (21) | 9 | 1 | 0 | 2 | 2.4e-4 | 3.6e-2 | 17/17 | 154x |

| d | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| Ours under (of 10) | 0 | 2 | 6 | 5 | 8 | 7 | 8 |
| Fixed terms under (of 10) | 5 | 3 | 0 | 0 | 0 | 0 | 0 |
| Median fixed terms | 2.9e-4 | 1.2e-2 | 2.2e-2 | 3.8e-2 | 6.7e-2 | 8.5e-2 | 1.1e-1 |

- All 8 successes are at d <= 3, among the 18 problems where ours chose no n. It never beats
  ours where ours chose an n.
- Against PySR alone: better on 55/70, median 4.6e-2 against 6.2e-2, and 8 against 7 under.
- With fixed terms as the fallback ours reaches 44/70. Of the 18 no-n problems, 10 are under
  with at least one of PySR alone, sparse linear or fixed terms.

## Conclusion

Having the right terms is not enough. Fixed terms and ours draw on the same quadratic
monomials, and only the method with the encoder stays accurate as d grows. Caveat: maxsize 35
cannot hold a weighted sum of all the terms from about d = 4 for this arm either, so the
larger-maxsize check must cover it.
