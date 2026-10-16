# Paper plan: adaptive interpretable bottleneck for symbolic regression

## Claim

One adaptive bottleneck finds the few interpretable variables a target
depends on, returns them in closed form, and hands them to symbolic
regression (SR), which then recovers equations it misses on raw inputs at
matched compute. The same framework works with tensor invariants (anisotropic
yield surfaces) and with power-law groups (physics laws), without units.

## Positioning (closest work; verify before citing)

| Line of work | What it does | Our difference |
|---|---|---|
| AI Feynman (Udrescu & Tegmark 2020) | Dimensional analysis + NN-guided recursion before SR | Unit-free; adaptive n; works on tensor invariants |
| Dimensionless learning: Bakarji et al. 2022 (Nat. Comput. Sci.), Xie et al. 2022 (Nat. Commun.), IT-pi (Yuan & Lozano-Duran 2025), Hi-pi (2025), arXiv:2411.15919 | Learns power-law groups, mostly from a dimension matrix or with a fixed group count | No units; minimal n chosen from data; exact rational exponents; coupled to SR |
| FePySR (Yu, Lu, Lai, arXiv:2605.12704, 2026) | Neural feature extraction before PySR | Our features are closed-form groups with selected n; must be a baseline |
| Mechanics model discovery: EUCLID (Flaschel et al. 2022), neural-polynomial elastoplasticity (CMAME 2024), ICNN yield functions | Sparse libraries or networks for yield/plasticity | We learn the invariants themselves, then SR |

## Method revisions (all opt-in; yield-surface defaults unchanged)

1. Stage 1: choose n from the gradient spectrum of a full-width network
   (row-normalized d f / d features), instead of a fixed nrmse threshold.
2. Stage 2: exact rational exponents (snap) in the sparsest integer basis.
3. Stage 3: portfolio fallback (part of the budget on raw inputs), so the
   pipeline is never worse than SR alone at matched compute.
4. Sign-aware log features to cover all 119 Feynman equations.

## Protocol (fixed before the full run)

- Development set: the 21 pilot equations (already seen). Test set: the other
  98 AI Feynman equations and all SRSD-Feynman tiers (different sampling
  ranges). Report them separately; tune only on development.
- Noise 0, 1e-3, 1e-2 (Gaussian, relative to target std); 5-10 seeds.
- Matched total compute per arm; recovery vs budget (5, 15, 60 min).
- Arms: PySR alone; PySR with units; pipeline; pipeline + fallback; PySR on
  the true groups (upper bound); ablations (no snapping, rref vs sparsest
  basis, n fixed to the number of inputs); FePySR if its code runs.
- Metrics: SRBench symbolic solution, numerical recovery (fresh-sample
  relative L2 < 1e-6), R^2 > 0.999 rate, complexity, time to solution.
  Paired per-equation tests (McNemar / Wilcoxon), bootstrap intervals.

## Yield-surface experiments

- Rotated Hill (existing): exact recovery from noisy data.
- Second criterion: open; see the representability note in the week-1 report.
- Real data: DX56D and single-crystal gold (existing adapters).
