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

### Recovery criterion for yield surfaces (RESEARCH_LOG 49)

- Recovered = relative L2 on 5000 fresh surface points < 1e-4, the same for
  every method. Equation datasets keep 1e-6 (exponents snapped there).
- Why: the PySR community treats a loss below ~1e-7 as "found". With our
  `inverse_target_squared` weights the loss is ~ (relative error)^2, so
  1e-7 is ~3e-4 relative error; 1e-4 (loss ~1e-8) leaves a margin.
- Caveat for the paper: loss thresholds in the PySR docs refer to plain MSE
  in the target's units and depend on scale; state the criterion as a
  relative error, not as a loss value.
- The earlier 1e-6 failed our pipeline only because of float32 precision in
  the learned tensor (structure exact, error ~1e-5). The structure score
  (F') is reported alongside as a second, structural criterion.
- Borderline case to mention: RH2-B (1.4e-4) passes a loss < 1e-7 rule but
  not our criterion. The criterion was fixed before that run and not
  changed per run.

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

## Hill ladder (seed 42)

| | VM25 | VM100 | HILL | RH2 | RH3 |
|---|---|---|---|---|---|
| A ours | 1.3e-5 | 1.4e-5 | 1.3e-5 | 1.1e-6 | 1.1e-5 |
| B PySR alone | 7e-8 | 2.3e-6 | 7.4e-3 x | 1.5e-4 x | 5.8e-2 x |
| C distillation | 3.9e-2 x | - | 1.4e-2 x | 1.1e-2 x | |
| D sparse linear | 4e-11 | - | 5e-11 | 4e-11 | 4e-11 |
| E fixed library | 2e-8 | - | 5.8e-2 x | 6.6e-2 x | 5.7e-2 x |
| F' tensor cosine | 0.9999999995 | 0.9999999995 | 0.9999999996 | 0.99999999999 | 0.9999999997 |

(test relative L2; x = not recovered at 1e-4.)

Story: A recovers every surface with the exact tensor. B degrades with
anisotropy and dimension. E fails as soon as the surface is anisotropic,
which isolates the learned tensor. C finds the right invariant but loses
accuracy in the two-step chain. D is exact because the surfaces are
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

- Borderline, not recovered (1.4e-4 fresh, 1.5e-4 test; loss 2.1e-8).
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
  maxsize ~100 (RH3-B-max). If it still fails, the claim is much stronger.
- Same settings for every arm (fairness). Distillation's inner searches use
  plain MSE instead of relative weights because the latents cross zero; its
  outer search uses the config unchanged.
