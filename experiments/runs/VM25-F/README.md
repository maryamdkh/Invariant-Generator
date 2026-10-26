# VM25-F: von Mises 25, structure recovery (from VM25-A)

Computed from the VM25-A Stage 2 checkpoint (no separate run):
n = 1 (true 1); one active invariant I11 = sigma:A:sigma; learned A vs the
true von Mises tensor (Mandel): cosine 0.9999999995; eigenvalues / max 5e-5
(true 0, hydrostatic) and 0.99994-1.0 (true 1). Script: the check in
RESEARCH_LOG 42 (load_model_for_checkpoint, effective_fourth_order_tensor,
Mandel conversion, compare with `data/hill_ladder_full/vonMises25_full.json`).
