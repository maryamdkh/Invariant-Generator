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

Classic yield criteria are used as ground truths, as in the recent ML
papers above: Hill (1948), Cazacu & Barlat (2001) CB2001, Barlat et al.
(2005) Yld2004-18p; PDFs in `references/`.

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
