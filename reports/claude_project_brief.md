# Project brief (paste into the Claude project's instructions)

**Who and what.** I am Maryam, a PhD student. My project, Invariant-Generator
(GitHub: maryamdkh/Invariant-Generator, branch `feynman-generalization`), is
an adaptive interpretable encoder that sits in front of symbolic regression
(PySR). Stage 1 finds the smallest number n of encoded variables a neural
model needs; Stage 2 turns the encoder rows into interpretable, exact
combinations (power-law groups with rational exponents, or tensor
invariants); Stage 3 runs PySR on those few variables and expands the
result back to the original inputs.

**Core result (must never break).** The pipeline was built for anisotropic
yield surfaces. On rotated Hill it beats PySR alone. Everything else
(AI Feynman, LLM-SRBench, noise, scaling) is generalization work, added only
as opt-in options; rotated Hill is checked to give identical results after
every code change.

**Paper.** CS/ML venue, method-first: "an adaptive interpretable bottleneck
that hands symbolic regression a few closed-form variables". We do not
compete with PySR; we set the scene for it. Yield surfaces are an
application section. Prefer recent citations.

**Results so far (seed 42).**
- AI Feynman, 110 equations, 15-min PySR budget: pipeline 85, portfolio 88,
  PySR alone 86 recovered. No difference: PySR alone solves most in seconds,
  and single-seed counts vary by +-2 between identical runs.
- Structure discovery: Stage 1 finds the true number of groups on 93 of 102
  reducible Feynman equations (and correctly no reduction on all 8 others).
- CB2001 yield criterion: Stage 1 found n = 2, rows match the true
  anisotropic invariants (cosine 0.999999); PySR got close (error 2e-4).
- Noise smoke test (1 equation, 1% noise): pipeline returned the exact
  formula; PySR alone fitted a slightly wrong constant (1.00038).

**Running now / next.**
1. LLM-SRBench LSR-Transform (111 harder problems), seed 42, on the
   Arrhenius cluster.
2. AI Feynman with 1% target noise, seed 42, on the cluster.
3. Scaling experiment (to build): synthetic equations with 4-12 inputs that
   depend on a few groups; hypothesis: PySR alone degrades as inputs grow,
   the pipeline does not. Likely the key figure.
4. Multiple seeds wherever an effect shows.

**How I work.** I run the cluster jobs myself (two-factor login) and share
outputs as screenshots; the code work happens in Claude Code on my Mac,
which keeps `HANDOFF.md` and `RESEARCH_LOG.md` (the full record with
numbers) in the repo.
