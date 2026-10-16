from pathlib import Path
from invariant_generator.config import Config
import sys
SP = Path(sys.argv[1])
def base_config(name, variables):
    c = Config()
    c.data.kind = "supervised"; c.data.data_dir = Path("data/feynman").resolve(); c.data.dataset_name = name
    c.data.dataset_key = "X"; c.data.target_key = "y"; c.data.feature_names = variables
    c.augmentation.enabled = False
    c.input_transform.kind = "log"
    c.encoder.enabled = True; c.encoder.init = "random"
    c.normalization.enabled = True; c.normalization.mode = "standard"
    c.model.output_activation = "none"; c.model.standardize_target = True
    c.loss.lambda_structure = 0.0; c.loss.lambda_encoder_l1_ratio = 0.0; c.loss.lambda_encoder_l2 = 0.0
    c.train.results_dir = SP / "log_probe"; c.train.split_dir = SP / "log_probe" / "splits"; c.train.device = "cpu"
    c.train.epochs = 3000; c.train.learning_rate = 1e-3; c.train.encoder_learning_rate_scale = 30.0
    c.train.lr_plateau_min_delta = 0.0; c.train.early_stopping_min_delta = 0.0; c.train.log_every = 25
    c.adaptive.metric = "nrmse"; c.adaptive.nrmse_threshold = 0.01; c.adaptive.restarts = 3
    c.adaptive.n_min = 1; c.adaptive.n_max = len(variables) - 1; c.adaptive.results_subdir = name
    return c
