# RH3-F: rotated Hill 3D, structure recovery (from RH3-A)

`experiments/probes/structure_recovery.py` on the RH3-A Stage 2 checkpoint
(output `experiments/outputs/structure_recovery_rh3.json`): n = 1 (true 1),
one active invariant I11; learned A vs the true rotated Hill tensor (full
6x6 Mandel, 21 non-zero entries): cosine 0.9999999997; eigenvalues / max
2.5e-5 (true 0) and the five non-zero ones within 2e-5. Error floor of
c * sqrt(sigma:A:sigma) on random directions 8.9e-6.
