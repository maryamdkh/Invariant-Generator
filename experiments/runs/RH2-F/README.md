# RH2-F: rotated Hill 2D, structure recovery (from RH2-A)

`experiments/probes/structure_recovery.py` on the RH2-A Stage 2 checkpoint
(output `experiments/outputs/structure_recovery_rh2.json`); with plane-stress
data only the s11, s22, s12 block of A is identifiable, so that block is
compared. n = 1 (true 1), one active invariant I11; cosine 0.99999999999;
eigenvalues / max 0.259761, 0.694907, 1 (true 0.259762, 0.694911, 1). Error
floor of c * sqrt(sigma:A:sigma) on random directions 1.1e-6.
