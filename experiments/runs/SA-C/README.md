# SA-C: Synthetic SA (70 problems), symbolic distillation, seed 42

- A neural network with an L1 penalty on 8 hidden variables; PySR explains each kept variable
  from the raw inputs, then y from the kept variables. The 15-minute limit is split over those
  searches, so the total matches one PySR run. Network settings are the defaults used in all
  C runs, not tuned on SA.
- Arrhenius, array of 70 tasks, `KEEP_ARTIFACTS=full`, no failed tasks. Branch head at
  submission was `1e8ef7c`; the commit was not recorded on the cluster. Results in
  `results/feynman_benchmark/sa_C/` (gitignored).
- Scored by `experiments/probes/sa_c_compare.py` -> `experiments/outputs/sa_c_score.csv`.
- 4 problems have no finite fresh score (the composed formula takes a sqrt of a negative value).
  They are not rescored: their test errors are 9.1e-3 to 1.5e-1, far above the threshold.

## Result (fresh relative L2; "under" = < 3.2e-4, provisional)

| Group | Ours | PySR alone | Distillation | Sparse linear | Fixed terms | Median ours | Median distillation | Ours better (paired) | Median ratio |
|---|---|---|---|---|---|---|---|---|---|
| all (70) | 36 | 7 | 0 | 21 | 8 | 1.6e-4 | 6.3e-2 | 49/49 | 603x |
| sqrt(q1) (21) | 14 | 6 | 0 | 21 | 6 | 2.7e-5 | 5.4e-2 | 14/14 | 2656x |
| q1/(1+q2) (28) | 13 | 0 | 0 | 0 | 0 | 2.1e-4 | 8.1e-2 | 19/19 | 429x |
| sqrt(q1)+q2 (21) | 9 | 1 | 0 | 0 | 2 | 2.4e-4 | 5.8e-2 | 16/16 | 198x |

| d | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| Median distillation | 6.9e-3 | 2.9e-2 | 4.3e-2 | 6.3e-2 | 8.2e-2 | 1.1e-1 | 1.1e-1 |
| Median network NRMSE | 2.5e-3 | 5.4e-3 | 5.2e-3 | 3.2e-3 | 1.4e-3 | 1.2e-3 | 1.1e-3 |

- Best distillation result on any problem: 1.5e-3, about five times the threshold.
- The network does not degrade with d; the symbolic error does. Median final error is about
  21 times the network's NRMSE (training NRMSE against fresh relative L2, so only indicative).
- Against PySR alone: better on 21/66; better on 16/26 for q1/(1+q2), 1/20 for sqrt(q1).
- Hidden variables kept match the true n on 30/70 (10/21 at n = 1, 20/49 at n = 2). The count
  depends on our keep rule (importance > 0.2 of the largest).

## Conclusion

The network learns y well at every d, but its hidden variables have to be written as formulas
from the raw inputs, and that search fails in the same way as PySR alone. Same pattern as RH3-C
and CB-C, now over 70 problems. Caveat: maxsize 35 limits the hidden-variable searches too, and
each search has a smaller share of the time.
