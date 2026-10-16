"""Hypothesis probe: refit every 1-3 term encoder support from the Stage 1 checkpoint."""
import itertools, json, sys, time
from copy import deepcopy
import numpy as np, torch
from invariant_generator.config import load_config
from invariant_generator.adaptive import config_for_adaptive_n
from invariant_generator.data import prepare_training_data
from invariant_generator.evaluation import evaluate_model
from invariant_generator.sparsify import load_model_for_checkpoint, apply_encoder_mask
from invariant_generator.input_features import input_feature_names

cfg = config_for_adaptive_n(load_config("configs/adaptive_encoder_feynman_gaussian_sum.toml"), 1)
data = prepare_training_data(cfg)
names = input_feature_names(cfg)
dev = torch.device("cpu")
base = load_model_for_checkpoint(cfg, "results/feynman_i_6_20a_sum/stage1/n01/checkpoint_best.pt").to(dev)
dense_state = deepcopy(base.state_dict())
X = torch.as_tensor(data.X_train, dtype=torch.float32); y = torch.as_tensor(data.y_train, dtype=torch.float32)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
torch.manual_seed(0)

def refit(mask):
    model = deepcopy(base); model.load_state_dict(dense_state); apply_encoder_mask(model, mask)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    for _ in range(EPOCHS):
        opt.zero_grad(); loss = (model(X) - y).pow(2).mean(); loss.backward(); opt.step(); sched.step()
        apply_encoder_mask(model, mask)
    tr = evaluate_model(model, data.X_train, data.y_train, device=dev)["rmse"]
    va = evaluate_model(model, data.X_validation, data.y_validation, device=dev)["rmse"]
    return tr, va, model.encoder_matrix().detach().numpy()[0]

rows = []
t0 = time.time()
for k in (1, 2, 3):
    for support in itertools.combinations(range(len(names)), k):
        mask = np.zeros((1, len(names))); mask[0, list(support)] = 1
        tr, va, S = refit(mask)
        rows.append({"k": k, "support": [names[i] for i in support], "train_rmse": tr, "val_rmse": va,
                     "S": [round(float(S[i]), 4) for i in support]})
print(f"{len(rows)} refits in {time.time()-t0:.0f}s; y std {data.y_validation.std():.4f}")
for k in (1, 2, 3):
    best = sorted([r for r in rows if r["k"] == k], key=lambda r: r["val_rmse"])[:4]
    for r in best: print(k, r["support"], f"val={r['val_rmse']:.2e} train={r['train_rmse']:.2e}", r["S"])
json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else "/dev/null", "w"), indent=1)
