# Analysis notes for the paper

One section per experiment (or group), written right after the run, in the
form the paper needs: what the result means, why it came out that way,
caveats a reviewer may raise, and which contribution it supports. Numbers
and commands are in `experiments/runs/<ID>/README.md`; the step-by-step
record is in `RESEARCH_LOG.md`. Contributions: (1) closed-form latent, one SR
step; (2) learnable physical structure (a, A) in the library; (3) automatic
choice of n.

All results below are seed 42 only unless stated; claims need 5 seeds.

## Cross-cutting decisions

### Recovery criterion for yield surfaces (RESEARCH_LOG 49, 68)

- Recovered = relative L2 on 5000 fresh surface points < 10^-3.5 (~3.2e-4),
  the same for every method. Equation datasets keep 1e-6 (exponents
  snapped there).
- Why: the PySR community treats a loss below ~1e-7 as "found". With our
  `inverse_target_squared` weights the loss is ~ (relative error)^2, so
  loss 1e-7 = relative error ~3.2e-4. Author's decision (entry 68).
- History, to state openly: 1e-6 first (failed our pipeline only because of
  float32 precision, structure exact), then 1e-4 (entries 49-67), then
  3.2e-4. Only two of 21 seed-42 results depend on the choice between 1e-4
  and 3.2e-4: CB-A (2.9e-4, exact CB2001 form) and RH2-B (PySR alone, 3
  inputs, 1.4e-4). Both become recovered; the first is right (true
  equation found), the second matches the author's earlier observation that
  PySR alone solves rotated Hill with 3 inputs. Every RH3 result is >= 100x
  above either threshold, so the core claim does not depend on it.
- Caveat for the paper: loss thresholds in the PySR docs refer to plain MSE
  in the target's units and depend on scale; state the criterion as a
  relative error, not as a loss value.
- For the paper: main tables at 3.2e-4; appendix table with recovery at
  1e-4, 1e-5, 1e-6 next to the structure score (F'), so no reviewer can say
  the threshold was chosen for us. Symbolic recovery (right expression up
  to constants) is reported separately from numerical recovery.

Sensitivity, seed 42 (fresh relative L2; Y = recovered):

| run | fresh | 1e-6 | 1e-5 | 1e-4 | 3.2e-4 |
|---|---|---|---|---|---|
| VM25-A | 1.3e-5 | n | n | Y | Y |
| VM100-A | 1.4e-5 | n | n | Y | Y |
| HILL-A | 1.1e-5 | n | n | Y | Y |
| RH2-A | 1.2e-6 | n | Y | Y | Y |
| RH3-A | 1.0e-5 | n | n | Y | Y |
| CB-A | 2.9e-4 | n | n | n | Y |
| VM25-B | 7.4e-8 | Y | Y | Y | Y |
| VM100-B | 2.3e-6 | n | Y | Y | Y |
| HILL-B | 8.8e-3 | n | n | n | n |
| RH2-B | 1.4e-4 | n | n | n | Y |
| RH3-B | 6.0e-2 | n | n | n | n |
| RH3-B-max | 3.3e-2 | n | n | n | n |
| all C | >= 1.1e-2 | n | n | n | n |
| all D | <= 1e-8 | Y | Y | Y | Y |
| VM25-E | 1.7e-8 | Y | Y | Y | Y |
| HILL/RH2/RH3-E | >= 6e-2 | n | n | n | n |

At 1e-5 and 1e-6 our 6D results drop because of the learned-constant floor
(see next section), while the structure score stays at cosine >= 0.99999;
this is the case for a final constant refit before submission.

### Error floor of our pipeline on 6D surfaces

- A errors are ~1e-5 on all 6D Hill-ladder surfaces and ~1e-6 in 2D.
  `experiments/probes/structure_recovery.py` shows this is the learned
  tensor's own precision: the best c * sqrt(sigma:A_learned:sigma) cannot do
  better (floor 9.5e-6 HILL, 8.9e-6 RH3, 1.1e-6 RH2). PySR stopping time is
  irrelevant; longer equations change the loss only in the 4th digit.
- 2D is 10x more precise because only a 3x3 block of A is learned.
- Possible improvement (not done): a float64 refit of A at the end of Stage 2.

### Sparse linear (D) solves the whole Hill ladder

- Every Hill-ladder surface is a quadratic form, so y^2 is linear in the
  quadratic monomials (9 terms aligned, 21 rotated) and STLSQ is exact
  (~1e-11) in 0.03 s, even on RH3 where PySR alone fails.
- Consequence for the paper: on the Hill ladder our advantage over D is
  interpretability (one tensor A with physical eigenvalues, one invariant,
  a 4-node formula vs 21 unrelated coefficients), not accuracy. Accuracy
  over D must be shown on surfaces that are not polynomial in sigma
  (CB2001, Yld2004, CPFEM). A reviewer will raise this; address it upfront.

### Sparse linear (D) also solves CB2001 (RESEARCH_LOG 66)

- CB2001 is y^6 proportional to J2^0^3 - c J3^0^2, a degree-6 polynomial in
  sigma. With powers up to 6 and a degree-6 library (924 terms), STLSQ is
  exact (1.6e-9) with 138 active terms. Earlier "D fails on CB2001" (2.1e-2,
  RESEARCH_LOG 40) came from capping D at m = 3, plus a bug in the inverse
  of even powers (fixed).
- So CB2001 does not show an accuracy advantage over D either. Our
  advantage there is interpretability and compactness: n = 2 encoded
  variables (J2^0, J3^0) and a short nested formula vs 138 coefficients.
- Accuracy over D must come from surfaces that are not polynomial in sigma:
  Yld2004-18p (exponent 8 on differences of principal values of linear
  transforms: not a polynomial) and CPFEM data. Plan the paper's argument
  around that; do not claim "D cannot do CB2001".

## CB2001, 2090-T3 (seed 42)

| | CB2001 |
|---|---|
| A ours | 1.8e-4 (fresh 2.9e-4), exact form at complexity 13 |
| B PySR alone | 2.4e-2 x |
| D sparse linear | 1.6e-9 (138 terms, m = 6) |
| F' row cosines | 0.99999993 (J2^0), 0.999994 (J3^0) |

### CB-B (PySR alone)

- Not recovered (2.4e-2; loss plateau 3.6e-4 after 697 s). PySR returns a
  sqrt-of-quadratic surface, i.e. a Hill/von Mises-like form; the cubic
  invariant J3^0, which carries CB2001's tension/compression asymmetry,
  never appears (only a tiny cbrt(s12) s23 s13 term in the longest
  equations).
- CB2001 in raw components is a degree-6 polynomial under a sixth root with
  hundreds of terms; PySR cannot assemble it. With the two encoded
  invariants it is a 13-node formula (CB-A). Second case after RH3 where
  the encoder turns an unsolvable search into an easy one; this one is
  non-quadratic and needs n = 2 (contributions 1 and 3).

### CB-A / CB-F (ours)

- Contribution 3 shown: Stage 1 rejects n = 1 (7.0e-2) and accepts n = 2
  (6.8e-3), the true number of generalized invariants.
- Contribution 1: the two rows are J2^0 and J3^0 (cosines 0.99999993,
  0.999994) and PySR finds the exact CB2001 form
  cbrt(sqrt((J1/c)^3 - k J2^2)) at complexity 13 within 7 minutes.
- Recovered at the main threshold 3.2e-4 (fresh 2.9e-4, selected); not at
  1e-4. The exact form itself scores 3.3e-4.
  Scoring the whole hall of fame shows no equation below 2.9e-4, so it is
  not a PySR stopping issue: the floor is the learned J3^0 coefficients
  (3.4e-3 relative error). Same pattern as the Hill ladder (exact structure,
  floor = precision of learned constants), larger here because 29
  coefficients are learned instead of a 6x6 tensor.
- Under a "loss < 1e-7 means found" reading (loss 8e-8 at the exact form)
  this counts as found; under our relative-error criterion it does not. The
  author's decision on the criterion settles it (see the criterion section).
- PySR's model_selection = "best" picked complexity 22 (loss 4.3e-8, not the
  exact form) over the exact form at 13 (8.3e-8). For the paper, report the
  hall of fame, not only the selected equation.
- Remedy worth considering for the method: a final joint refit of the
  encoder coefficients and formula constants once the structure is found
  ("constant polishing", RESEARCH_LOG 22). It would likely push CB-A (and
  the Hill-ladder 1e-5 floor) well below 1e-4 without changing structure.

## Hill ladder (seed 42)

| | VM25 | VM100 | HILL | RH2 | RH3 |
|---|---|---|---|---|---|
| A ours | 1.3e-5 | 1.4e-5 | 1.3e-5 | 1.1e-6 | 1.1e-5 |
| B PySR alone | 7e-8 | 2.3e-6 | 7.4e-3 x | 1.5e-4 | 5.8e-2 x |
| C distillation | 3.9e-2 x | - | 1.4e-2 x | 1.1e-2 x | 5.4e-2 x |
| D sparse linear | 4e-11 | - | 5e-11 | 4e-11 | 4e-11 |
| E fixed library | 2e-8 | - | 5.8e-2 x | 6.6e-2 x | 5.7e-2 x |
| F' tensor cosine | 0.9999999995 | 0.9999999995 | 0.9999999996 | 0.99999999999 | 0.9999999997 |

(test relative L2; x = not recovered at 3.2e-4.)

Story: A recovers every surface with the exact tensor. B degrades with
anisotropy and dimension. E fails as soon as the surface is anisotropic,
which isolates the learned tensor. C never recovers: on VM, HILL, RH2 it
finds the right invariant but loses accuracy in the two-step chain or in
the 6-input inner search; on RH3 its inner search fails exactly like B. D is exact because the surfaces are
quadratic.

### VM25-A / VM100-A (ours, von Mises)

- n = 1, one invariant I11, y = c * J1 (complexity 4). Learned A equals the
  deviatoric projector (cosine 0.9999999995; hydrostatic eigenvalue 5e-5
  instead of 0). Error 1.3e-5 = tensor precision (see floor above).
- VM100 (yield stress 4x) gives the same structure and error; the overall
  constant ratio is 3.97 (the tensor's scale is free, so not exactly 4).
  The pipeline is scale-independent.

### VM25-B / VM100-B (PySR alone, von Mises)

- Recovered: exact von Mises form (constants sqrt 3, 1/3) on raw components,
  but needs complexity 32-36 vs 4 for A. Control case: PySR alone is fine on
  the isotropic surface.

### VM25-C (distillation)

- One latent kept. Inner PySR found z = 1.709 - 0.002056 [(1/2) sum of
  squared stress differences] - 0.006208 shear^2; the shear/normal ratio is
  3.02 (von Mises: 3), so the latent IS the von Mises invariant.
- But the MLP latent is only approximately affine in sigma_eq^2 (inner loss
  0.024 on the standardised latent); the outer search approximates the
  inverse sqrt with a cubic; errors compound to 3.9e-2.
- Supports contribution 1: our latent is closed form by construction, so no
  inner fit and no compounding (A: 1.3e-5, one SR step).
- Honest note: a final constant refit of the composed formula would likely
  improve C. It is not part of Cranmer et al. (2020), so it is left out to
  keep the baseline faithful; offer it as a variant if reviewers ask.
- Implementation note: the original L1 on z was dodged by rescaling (all
  latents shrank, decoder weights grew); fixed with the scale-invariant
  penalty |z_j| * ||W_j|| (RESEARCH_LOG 58). Describe this in the baselines
  section so the comparison is seen as fair.

### HILL-C (distillation, aligned Hill)

- One latent kept; inner PySR found z = c - sqrt(Hill quadratic) and the
  outer map is linear, so the latent is the right invariant and is affine
  in y: here the two-step chain costs nothing.
- The error (1.4e-2) is entirely the inner search on 6 raw components:
  complexity reaches the maxsize 40 cap before every term gets its own
  constant (s13 and s23 share one; others rounded, 5-20% off the truth).
  This is the same limit PySR alone hits (HILL-B).
- Point for the paper: distillation still has to solve a 6-input symbolic
  problem (z from x), so it inherits PySR's difficulty with dimension. Our
  encoder replaces that step with a learned tensor in a fixed invariant
  form, which is exact by construction (A: 1.3e-5). Supports contribution 1
  (one SR step on a closed-form latent) and 2 (structure learned by gradient
  descent, not searched).

### RH2-C (distillation, rotated Hill 2D)

- One latent kept; inner PySR found the rotated Hill quadratic (affine in
  y^2) with coefficients 0.6-2.7% off (s11 s22 pinned to -1: one constant
  short, as PySR alone in RH2-B); outer y = sqrt(z0 + 2.123) - 0.315 has the
  right shape but its offset does not match the latent's (1.728). Composed
  1.1e-2, worse than PySR alone (1.5e-4) on the same 3 inputs.
- Point for the paper: even in low dimension, splitting one SR problem into
  two (z from x, then y from z) lets each step's small errors compound;
  distillation is worse than PySR alone here. Our latent needs no inner
  search (RH2-A 1.1e-6). Supports contribution 1.

### RH3-C (distillation, rotated Hill 3D)

- One latent kept; the network itself is accurate (predicts y to 2.2e-3
  nrmse), so the latent contains the right information.
- The inner PySR search (z from 6 raw inputs) hits the maxsize 40 cap and
  returns a near-isotropic Hill-like form (s13, s23 share a constant; no
  rotation cross terms), exactly like PySR alone (RH3-B). Composed 5.4e-2,
  the same level as B (5.8e-2) and E (5.7e-2).
- Point for the paper: distillation moves the difficulty, it does not remove
  it: a symbolic search over all raw inputs is still needed to explain the
  latent. Our latent is a learned tensor inside a fixed invariant form, so
  there is nothing to search on the raw inputs (RH3-A 1.1e-5). This is the
  sharpest evidence for contribution 1 against the closest prior work.
- Caveat as for RH3-B: maxsize 40 binds the inner search; RH3-B-max checks
  whether a larger cap changes the picture for PySR on raw inputs (the same
  conclusion would apply to C's inner search).

### VM25-E (fixed library, von Mises)

- Recovered: y = sqrt(0.0024 F2^2 - 0.0008 F1^2) at complexity 12, i.e. von
  Mises in isotropic invariants. For isotropic surfaces the fixed library is
  enough; it shortens the formula (12 vs 32-36 on raw components).

### HILL-A / HILL-F (ours, aligned Hill)

- n = 1, I11, complexity 4, 1.3e-5. Learned A matches the Hill tensor:
  cosine 0.9999999996, all five non-zero eigenvalues to 4-5 digits.
  Anisotropy is absorbed into A without any change to the method.

### HILL-B (PySR alone, aligned Hill)

- Not recovered (7.4e-3, loss plateau 4.7e-5; the author knows from earlier
  runs that it does not go lower). The equation has the Hill shape; its
  quadratic form matches the truth within 0.1% (normal block) and 0.3%
  (s23), but s12 and s13 share one coefficient (truth 0.22 vs 0.20). One
  missing constant is the whole error; adding it costs complexity beyond
  the plateau at 38.
- First surface where A succeeds and B fails.

### HILL-E / RH2-E / RH3-E (fixed library, anisotropic surfaces)

- Not recovered (6-7%), loss plateaus at 2.5-3e-3 however long the formula.
- Expected by construction: with identity tensors every feature is
  isotropic (unchanged by rotating sigma), so no formula in them can express
  an anisotropic surface. On RH3 PySR returns exactly the closest von Mises
  form.
- E vs A is the ablation for contribution 2 (learned structural tensor):
  same library, same PySR, only the tensor is fixed. This replaces the
  "fixed A = I" ablation.

### HILL-D / RH2-D / RH3-D (sparse linear)

- Exact (see "Sparse linear solves the whole Hill ladder" above).

### RH2-A / RH2-F (ours, rotated Hill 2D)

- n = 1, I11, complexity 4, 1.1e-6 (10x better than 6D: 3x3 block).
  Cosine 0.99999999999. Time to recovery 1306 s, of which PySR 20 s: the
  cost is the encoder, PySR's part is trivial.

### RH2-B (PySR alone, rotated Hill 2D)

- Recovered at 3.2e-4, not at 1e-4 (1.4e-4 fresh, 1.5e-4 test; loss 2.1e-8).
  Same failure as HILL-B: right quadratic shape, but s11^2 and s22^2 forced
  equal while the truth differs by 0.12%. The author had seen PySR alone
  solve this case; another seed may cross the threshold. Report honestly as
  borderline.

### RH3-A / RH3-F (ours, rotated Hill 3D, the core case)

- n = 1, I11, complexity 4, 1.1e-5; PySR reached loss 1e-8 after 20 s.
  Learned A vs truth cosine 0.9999999997; eigenvalues equal HILL-A's (same
  tensor, rotated): the rotation is absorbed into the eigenvectors of A.

### RH3-B (PySR alone, rotated Hill 3D, the core comparison)

- Not recovered (5.8e-2), loss plateau 2.2e-3 after 1056 s.
- The best equations are a von Mises-like form (equal-weight squared stress
  differences + squared shears, one shear rescaled) with a fourth root and
  an offset; the rotation's cross terms (s11 s23, s13 s12, ...) never appear.
  PySR alone does not find the right form, not just the constants.
- Head-to-head: A 1.1e-5 at complexity 4 with PySR at loss 1e-8 in 20 s vs
  B 5.8e-2 at complexity 34 after 1056 s.
- PySR alone across the ladder: recovered (VM) -> one constant short (HILL,
  RH2) -> wrong form (RH3). Clean degradation with anisotropy and dimension.
- Priority: more seeds for RH3-B (and A) before the claim goes in the paper.
- Caveat (maxsize): all arms share the author's PySR config with
  maxsize = 40. The exact rotated Hill formula on raw components has 21
  quadratic terms (complexity ~80+), so B cannot express it under the cap;
  ours needs 4. Shortening the formula is part of our point, but a reviewer
  may say PySR alone only needed a larger maxsize. Planned check: RH3-B with
  maxsize 150 (RH3-B-max; the expanded exact form needs ~143 with
  constants costing 2, and fits the other constraints). If it still fails, the claim is much stronger.
- RH3-B-max (done, RESEARCH_LOG 65): with maxsize 150 and 92 min (5x
  RH3-B's time), PySR alone never went beyond complexity 80, best loss
  7.8e-4, test error 3.4e-2 (RH3-B: 5.8e-2). Not recovered, four orders of
  magnitude from the threshold. The cap was not the reason: the search does
  not find its way to a 143-node formula. Answer to the reviewer: raising
  the cap and the time helps a little; it does not solve the problem. The
  encoder turns it into a 4-node problem PySR solves in 20 s.
- Same settings for every arm (fairness). Distillation's inner searches use
  plain MSE instead of relative weights because the latents cross zero; its
  outer search uses the config unchanged.
