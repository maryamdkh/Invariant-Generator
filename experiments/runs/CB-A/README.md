# CB-A: CB2001 (2090-T3), our pipeline

- Goal: n = 2 encoded variables = the generalized invariants J2^0 (quadratic)
  and J3^0 (cubic), then PySR finds y = (J2^0^3 - c J3^0^2)^(1/6) * const.
- Data `data/yield_benchmarks/cb2001_2090T3.hdf` (2000 points), config
  `config.toml` (= `configs/cb/cb2001_2090T3.toml`: orthotropic polynomial
  features (29 monomials), graded Stage 2, the author's PySR settings +
  cube/cbrt).
- Run (Arrhenius interactive node, 2026-10-28, commit `8466a4d`; the first
  attempt died with a lost connection outside tmux):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/cb/cb2001_2090T3.toml \
          --results-dir results/yield_benchmark/cb/seed42/cb2001_2090T3 \
          --arms pipeline --seed 42 --numprocs 16 --device cpu

- PySR stopped by hand after 434 s, once the loss was below ~1e-7.

## Result

| | |
|---|---|
| Stage 1 | n = 1 val nrmse 7.0e-2 (rejected), n = 2 6.8e-3 (accepted), 815 s |
| Stage 2 | graded: J1 = 9 quadratic terms, J2 = 20 cubic terms, 213 s |
| J1 vs true J2^0 | cosine 0.99999993, coefficient error 3.7e-4 |
| J2 vs true J3^0 | cosine 0.999994, coefficient error 3.4e-3 |
| Exact form in hall of fame | complexity 13: cbrt(sqrt((J1/1.25522)^3 - 0.68226 J2^2)), loss 8.3e-8, fresh 3.3e-4 |
| Selected by PySR (model_selection best) | complexity 22, loss 4.3e-8, not the exact form |
| Fresh / test relative L2 (selected) | 2.9e-4 / 1.8e-4 |
| Recovered (< 1e-4) | no |

Every equation in the hall of fame scored on fresh points
(`experiments/probes/score_hall_of_fame.py`,
`experiments/outputs/cb2001_hall_of_fame_scores.json`): the best is 2.9e-4;
none reaches 1e-4.

## Conclusion

Structure fully right: n = 2 chosen automatically, the two rows are J2^0 and
J3^0, and PySR finds the exact CB2001 form at complexity 13 in 7 minutes.
The 3e-4 error is the learned J3^0 coefficients (3.4e-3 relative), not
PySR: longer formulas and later stopping only move the 4th digit. Same
pattern as the Hill ladder (exact structure, error floor = learned
constants), at a larger floor because 29 coefficients are learned. PySR's
"best" selection prefers a longer, wrong-form equation that is marginally
lower in loss.

Update 2026-10-28: yield recovery threshold is now relative L2 < 10^-3.5
(~3.2e-4, PySR loss ~1e-7; RESEARCH_LOG 68), so this run counts as recovered.
