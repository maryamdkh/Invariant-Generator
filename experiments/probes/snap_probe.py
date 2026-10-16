import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from probe_config import base_config, SP
from invariant_generator.adaptive import config_for_adaptive_n
from invariant_generator.sparsify import sparsify_encoder_from_checkpoint
from invariant_generator.feynman import parse_pmlb_metadata
for name in sys.argv[2].split(","):
    pr = parse_pmlb_metadata(f"data/feynman_tables/pmlb_metadata/{name}.yaml")
    cfg = base_config(name, pr.variables)
    s1 = json.loads((SP / "log_probe" / name / "stage1" / "adaptive_stage1_summary.json").read_text())
    n = s1["selected_n"]; cfg = config_for_adaptive_n(cfg, n)
    cfg.sparsification.method = "snap"; cfg.sparsification.masked_refit_epochs = 1500
    cfg.sparsification.masked_refit_learning_rate = 3e-3; cfg.sparsification.masked_refit_schedule = "cosine"
    cfg.sparsification.masked_refit_min_learning_rate = 1e-5
    from invariant_generator.adaptive import adaptive_sparsification_run_id
    r = sparsify_encoder_from_checkpoint(cfg, checkpoint_path=s1["selected_checkpoint"])
    d = json.loads(r.summary_path.read_text())
    print("RESULT", name, pr.variables, "| snapped", d["snapped"], "1/", d["selected_denominator"])
    for c in d["candidate_results"]:
        print("RESULT    D", c["denominator"], "resid", round(c["rounding_residual"], 4), "val nrmse", round(c["test_metrics"]["nrmse"], 5), c["exponents"])
