import h5py
import numpy as np

from invariant_generator.config import Config
from invariant_generator.train import train_from_config


def test_successful_training_keeps_only_best_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25, 0.1, 0.0, 0.0],
            [0.5, 1.0, 0.25, 0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    with h5py.File(data_dir / "toy.h5", "w") as f:
        f.create_dataset("stress", data=X)

    config = Config()
    config.data.data_dir = data_dir
    config.data.dataset_name = "toy"
    config.data.dataset_key = "stress"
    config.data.test_size = 0.25
    config.data.random_state = 0
    config.augmentation.n_aug_per_sample = 1
    config.augmentation.random_state = 0
    config.model.hidden_dims = [4]
    config.train.results_dir = tmp_path / "results"
    config.train.split_dir = tmp_path / "splits"
    config.train.run_id = "toy_run"
    config.train.epochs = 2
    config.train.save_every = 1
    config.train.log_every = 1
    config.train.batch_size = 0
    config.train.device = "cpu"

    result = train_from_config(config)

    checkpoints = sorted(path.name for path in result.experiment_dir.glob("checkpoint*.pt"))
    assert checkpoints == ["checkpoint_best.pt"]
    assert result.best_checkpoint.exists()
    assert not result.recovery_checkpoint.exists()
    assert result.history_path.exists()
