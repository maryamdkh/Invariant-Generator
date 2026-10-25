# Paper plan (CS/ML venue)

Target: ML venue (NeurIPS/ICML/ICLR main track or AI4Science workshop, TMLR,
Machine Learning: Science and Technology). The paper is method-first; yield
surfaces are one application domain, not the story.

## Working title

Adaptive interpretable bottlenecks: learning a minimal set of closed-form
variables to scale symbolic regression

## Claim

Symbolic regression (SR) fails mainly because its search space grows
combinatorially with the number of inputs. A small neural model can find,
from data alone, the minimal number of input combinations the target depends
on and express them in closed form; any SR backend then searches over those
few variables. At matched compute this recovers more equations, faster, and
(with a portfolio fallback) never fewer than the backend alone.

## Contributions

1. **Method.** A linear bottleneck over an interpretable feature map, with
   (a) data-driven choice of the bottleneck width n relative to an
   unconstrained reference model, (b) closed-form variables: exact rational
   exponents in the sparsest integer basis for power-law maps, invariant
   combinations for tensor inputs, (c) a backend-agnostic hand-off to SR with a
   portfolio fallback.
2. **Analysis.** n* equals the rank of the gradient span in feature space;
   this gives an a priori structure count per benchmark equation (98/110 AI
   Feynman equations are reducible in log space) and a diagnostic for the
   learned model.
3. **Evidence.** Standard SR benchmarks with matched compute, seeds, noise,
   and ablations; plus a structured-input domain (anisotropic plasticity) where
   the feature map is a tensor-invariant basis.

## Positioning (verify each before citing)

| Line of work | Gap we address |
|---|---|
| SR systems: PySR (Cranmer 2023), AI Feynman 1/2 (Udrescu & Tegmark 2020; Udrescu et al. 2020), uDSR, LLM-guided SR: LLM-SR (2024/25), LaSR (NeurIPS 2024) | Search still over raw inputs; our front end is orthogonal and can wrap any of them |
| Feature/dimension reduction before SR: FePySR (2026), DeepPySR (2026), latent-variable SR (McConaghy 2010) | Their features are not closed form or the width is not chosen from data |
| Dimensionless learning: Bakarji et al. 2022, Xie et al. 2022, IT-pi 2025, Hi-pi 2025 | Need units or a fixed group count; ours is unit-free with adaptive n |
| Active subspaces / sufficient dimension reduction (Constantine 2015) | Linear directions, not closed-form groups, and not coupled to SR |
| Benchmarks: SRBench (La Cava et al. 2021), SRSD-Feynman (Matsubara et al.), LLM-SRBench (ICML 2025) | Evaluate on all three; LLM-SRBench addresses memorized textbook equations |
| ML yield functions (application): convex neural yield discovery (arXiv:2606.19375, 2026), neural-polynomial + SR (CMAME 2024), EUCLID (2022-2025), survey "AI Meets Plasticity" (arXiv:2602.01215, 2026) | Black-box or fixed libraries; ours learns the invariants and returns closed forms |

Closest prior method (added 2026-10-25, must be discussed): symbolic
distillation, Cranmer et al. 2020 (NeurIPS, arXiv:2006.11287): NN with an
L1/bottleneck latent, then SR on the latent; its latent is a black box, so it
needs a second SR step (inputs -> latent). Ours is closed form by
construction. Also SINDy (Brunton et al. 2016) and SINDy-autoencoder
(Champion et al. 2019): sparse linear library regression is the final model
there; ours uses the library as a bottleneck before a nonlinear regressor.

Classic yield criteria are used as ground truths, as in the recent ML
papers above: Hill (1948), Cazacu & Barlat (2001) CB2001, Barlat et al.
(2005) Yld2004-18p; PDFs in `references/`.

## Comparison methods (agreed 2026-10-25, RESEARCH_LOG 33)

Chosen to test the core contributions: closed-form latent, learnable
physical structure (a, A) in the library, automatic n. Supersedes "Arms" below.

| | Method | Tests | Status |
|---|---|---|---|
| A | Ours + PySR | - | required |
| B | PySR alone, raw inputs | does the front end help | required |
| C | Symbolic distillation (Cranmer 2020): MLP with L1 latent, PySR on latent | closed-form latent vs black-box latent | required |
| D | SINDy on the same feature library (linear, no network) | need for bottleneck + nonlinear step | required |
| E | PySR on the full feature library, no encoder | features alone vs encoder | required |
| F' | Learned A vs true Hill tensor (up to scale) | anisotropy is actually learned | required (replaces fixed A = I ablation) |
| G | PySR `select_k_features` | selecting vs combining inputs; only meaningful with distractor inputs | optional |
| H | AI Feynman (cite reported numbers, do not run) | unit-based reduction vs unit-free groups | optional |
| I | Operon with and without our encoder | front end helps a second SR backend | required, run last |

F' generalised to "structure recovery": learned A / encoder rows / exponents /
n compared with the ground truth.

## Experiment matrix (agreed 2026-10-25, RESEARCH_LOG 34)

R = required, o = optional, - = not applicable, done = have results.
All runs: 5 seeds, same PySR budget for every arm.

| Dataset | A ours | B PySR | C distil | D SINDy | E PySR+lib | F' structure | G select_k | H AI Feyn | I Operon |
|---|---|---|---|---|---|---|---|---|---|
| Hill ladder: von Mises, aligned Hill, rotated Hill 2D (3 in), rotated Hill 3D (6 in) | R | R | R | R | R | R (A vs true) | - | - | o |
| CB2001 (2090-T3) | R (1 seed done) | R | R | R | R | R (rows vs J2^0, J3^0) | - | - | o |
| Yld2004-18p | R | R | R | o | o | - | - | - | o |
| Synthetic (a): rotated quadratic forms, d = 2..12 | R | R | R | R | R | R (A vs true) | - | - | R |
| Synthetic (b): power-law groups, d = 4..12, k = 1..3 | R | R | R | o | R | R (n, exponents) | o (distractors) | - | R |
| Feynman (110) | done (1 seed) | done (1 seed) | R | o | o | done (n: 93/102, 8/8) | o | o (cite) | o |
| LLM-SRBench LSR-Transform (91) | done (5 seeds) | done (5 seeds) | o | - | - | o | - | - | o |
| Noise: rotated Hill 3D + Feynman subset, 1-2 levels | R | R | o | - | - | o | - | - | - |

Notes:
- Synthetic (a): y = sqrt(x^T A x) alone is solved by linear regression of
  y^2 on quadratic monomials (SINDy), as is rotated Hill. Include outer
  functions that break this (e.g. two quadratic forms combined
  nonlinearly) so the family tests more than the SINDy case.
- E on tensor inputs: structural tensors cannot be learned without the
  encoder, so E uses the library with a, A fixed (identity).

Run order: (1) B on the Hill ladder; (2) A + F' on the Hill ladder;
(3) synthetic generators (a), (b) with A, B; (4) implement C, run on Hill,
CB2001, synthetic; (5) D and E (cheap); (6) Feynman to 5 seeds + C;
(7) noise; (8) I Operon.

## Experiments

1. **Main SR benchmarks:** SRBench AI Feynman (119; 21 development, rest
   test), SRSD-Feynman (Easy/Medium/Hard), LLM-SRBench (LSR-Transform,
   LSR-Synth). Noise 0, 1e-3, 1e-2; 5 seeds; matched total compute; recovery
   vs budget (5/15/60 min).
2. **Arms:** backend alone (PySR; LaSR or LLM-SR if feasible), pipeline,
   pipeline + portfolio, oracle groups (upper bound), FePySR if runnable.
3. **Ablations:** fixed-threshold vs reference-relative n; no snapping; RREF vs
   sparsest basis; no encoder-LR scaling; n fixed to the input count.
4. **Structured inputs (application section):** rotated Hill48 (exact
   recovery), CB2001 with published 2090-T3 coefficients (needs a quadratic
   and a cubic orthotropic invariant: two encoder rows over an orthotropic
   polynomial integrity basis), Yld2004-18p (stress test: not polynomial; report
   accuracy and compactness honestly).
5. **Metrics:** SRBench symbolic solution, numerical recovery (< 1e-6 on fresh
   points), R^2 > 0.999 rate, complexity, time to solution; paired tests and
   bootstrap intervals.

## Compute

Development runs on the laptop; full runs on Vera/Arrhenius via
`cluster/` (the author submits jobs; two-factor login).
