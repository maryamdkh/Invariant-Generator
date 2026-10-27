# VM25-C: von Mises 25 (all directions), symbolic distillation (Cranmer 2020)

- Method: MLP x -> 8 latents -> y with the scale-invariant latent L1
  (RESEARCH_LOG 58); kept components fitted by PySR from x, then y by PySR
  from the kept components; no constant refit after composition.
- Run (Arrhenius interactive node, 2026-10-27, commit `5e21412`):

      "$PYTHON" scripts/run_yield_benchmark.py --config configs/hill_ladder/vm25.toml \
          --results-dir results/yield_benchmark/hill_ladder/seed42/vm25 \
          --arms distillation --seed 42 --numprocs 16 --device cpu

- Network 104 s (train nrmse 2.5e-3); one component kept (z7; the next is
  15% of it). Two PySR searches, each stopped by hand; 1444 s in total.

## Result

Inner (z7 from stresses, loss 0.024 on the standardised latent):

    z7 = 1.70941 - 0.0020560 [(s11 - s22)^2 + (s33 - s11)(s33 - s22)]
                 - 0.0062077 (s12^2 + s13^2 + s23^2)

The bracket is (1/2) sum of squared stress differences and 0.0062077 /
0.0020560 = 3.02 (von Mises: 3), so z7 ~ 1.709 - 0.002056 sigma_eq^2: the
right invariant, shear weight 0.7% off.

Outer (y from z7, selected): y = z7 (0.010234 z7^2 - 0.35589) + 1.10347,
a cubic approximating y = sqrt((1.709 - z7) / 0.002056) / 25; the hall of
fame has sqrt forms at loss 1.3e-4.

| | |
|---|---|
| Fresh-point relative L2 | 3.86e-2 (test 3.89e-2) |
| Recovered (< 1e-4) | no |

## Conclusion

Distillation finds the von Mises invariant in its latent, but the latent is
only approximately affine in sigma_eq^2 (MLP warp; inner loss 0.024), the
outer search approximates the inverse with a cubic, and the two errors
compound to 4%. Our pipeline needs no inner fit: its latent is closed form
by construction (VM25-A, 1.3e-5). A final constant refit of the composed
formula (not part of Cranmer et al.) would likely help here; kept out so
the baseline follows the paper.
