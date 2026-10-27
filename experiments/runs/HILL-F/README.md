# HILL-F: Hill aligned, structure recovery (from HILL-A)

Computed from the HILL-A Stage 2 checkpoint (no separate run) with
`experiments/probes/structure_recovery.py` (output
`experiments/outputs/structure_recovery_hill.json`):
n = 1 (true 1); one active invariant I11; learned A vs the true Hill tensor
(Mandel): cosine 0.9999999996; eigenvalues / max 2.6e-5 (true 0) and the five
non-zero ones within 2e-5 of the truth. Error floor of c * sqrt(sigma:A:sigma)
on random directions: 9.5e-6.
