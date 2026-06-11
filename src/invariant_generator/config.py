from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRESS_NAMES = ["s11", "s22", "s33", "s23", "s13", "s12"]
INVARIANT_NAMES = [f"I{i}" for i in range(1, 14)]


@dataclass(slots=True)
class DataConfig:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    dataset_name: str = "rotatedhill"
    dataset_key: str = "stress"

    # Default order: [s11, s22, s33, s23, s13, s12].
    # Set stress_format="plane_stress_2d" only for old 3-column datasets
    # ordered as [s11, s22, s12].
    stress_format: str = "voigt_3d"
    feature_names: list[str] = field(default_factory=lambda: STRESS_NAMES.copy())

    test_size: float = 0.1
    random_state: int = 42
    shuffle: bool = True


@dataclass(slots=True)
class NoiseConfig:
    enabled: bool = False
    scale: float = 0.02
    random_state: int = 42
    relative_to_feature_std: bool = True


@dataclass(slots=True)
class AugmentationConfig:
    enabled: bool = True
    include_original: bool = True
    augment_test: bool = False

    # If k_values is not None, each listed scaling factor is applied to every
    # sample. If it is None, random k values are sampled from k_range.
    k_values: list[float] | None = None
    n_aug_per_sample: int = 5
    k_range: tuple[float, float] = (0.5, 1.75)
    random_state: int = 42
    shuffle: bool = True

    # For yield-surface effective stress, f(k*sigma)=|k|^1 f(sigma).
    homogeneity_degree: float = 1.0
    surface_target: float = 1.0


@dataclass(slots=True)
class InvariantConfig:
    # Candidate invariant pool. By default, use all available invariants I1 ... I13
    # and let the sparse encoder/PySR ranking decide which ones matter most.
    selected: list[str] = field(default_factory=lambda: INVARIANT_NAMES.copy())

    # These are enabled by default because the automatic pool includes
    # structural invariants that use a and A.
    enable_second_order: bool = True
    enable_fourth_order: bool = True

    # Optional transformation sign(I)*|I|^(1/degree) to make each invariant
    # first-degree homogeneous in stress, as suggested in the notes.
    homogenize: bool = False

    init_scale: float = 0.05
    eps: float = 1e-12


@dataclass(slots=True)
class EncoderConfig:
    enabled: bool = False

    # 0 means "same as number of selected invariants".
    output_dim: int = 0
    init: str = "identity"


@dataclass(slots=True)
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "silu"

    # "softplus" keeps the predicted effective stress non-negative.
    # Use "none" if you want the network output unconstrained.
    output_activation: str = "softplus"


@dataclass(slots=True)
class LossConfig:
    # Exact PDF structure:
    # L = L_data + L_param + L_structure + L_enc
    lambda_param: float = 0.0
    lambda_structure: float = 1e-3
    lambda_encoder_l1_ratio: float = 1e-4
    lambda_encoder_l2: float = 1e-4
    eps: float = 1e-12


@dataclass(slots=True)
class TrainConfig:
    run_id: str = "debug_invariant_generator"
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    split_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "splits")

    use_saved_split: bool = True
    save_split_if_missing: bool = True

    seed: int = 42
    epochs: int = 1000

    # 0 means full-batch training. This matches the dataset-sum L_data in the
    # notes exactly. Positive values enable mini-batch training.
    batch_size: int = 0

    learning_rate: float = 1e-3
    grad_clip_norm: float | None = 10.0
    device: str = "auto"

    log_every: int = 50
    # Positive values overwrite checkpoint_latest.pt at this interval. A
    # successful run removes it, leaving checkpoint_best.pt as the saved model.
    save_every: int = 100

    # Optional manual stop control. Training stops gracefully when stop_file
    # exists and contains one of stop_keywords, e.g. "stop".
    stop_file: Path = field(default_factory=lambda: PROJECT_ROOT / "STOP_TRAINING.txt")
    stop_keywords: list[str] = field(default_factory=lambda: ["stop", "quit", "exit", "halt"])

    # Plateau scheduler settings. Patience is counted in evaluation events
    # (epochs where logging/saving evaluates test metrics), not raw epochs.
    lr_plateau_enabled: bool = True
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 5
    lr_plateau_min_lr: float = 1e-6
    lr_plateau_min_delta: float = 1e-6

    # Early stopping settings, also counted in evaluation events.
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6


@dataclass(slots=True)
class SymbolicConfig:

    top_k: int = 3
    random_state: int = 42
    output_subdir: str = "symbolic"

    # Saving / resuming
    output_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    run_id: str = "rotatedhill_symbolic" #f"vonMises25_noise_{NOISE_SCALE}"

    # Search
    niterations: int = 10_000_000 # the number of outer iterations; how many times islands evolve + communicate
    timeout_hours: float = 24.0
    warm_start: bool = True
    parallelism: str = "multithreading" #"multiprocessing" #"multithreading" 
    numprocs: int = 4
    batching: bool = True
    batch_size: int = 100
    """
    If my dataset is more than 1000 points, I either subsample it (low-dimensional and not much noise) 
    or set batching=True (high-dimensional or very noisy, so it needs to evaluate on all the data).
    """


    populations: int = 24 # the number of islands; Increase populations to 3*num_cores.
    population_size: int = 50 # live search pool = populations × population_size
    ncycles_per_iteration: int = 5000 # the number of generations considered per iteration;
    """
    Larger ncyclesperiteration = threads spend more time working before talking to the leader thread.
    that means there's going to be less job on the main thread, no bottleneck, and more multithreading.
    """

    model_selection: str = "best"
    elementwise_loss: str = "loss(prediction, target, weight) = weight * (prediction - target)^2" #Weighted MSE
    weight_mode: str = "inverse_target_squared"
    
    """
    MSE:
    "loss(prediction, target) = (prediction - target)^2"

    Weighted MSE:
    "loss(prediction, target, weight) = weight * (prediction - target)^2" 

    Positivity-aware MSE:
    "loss(prediction, target, weight) = weight * (prediction - target)^2 + 1.0 * ifelse(prediction < 0.0, prediction^2, 0.0)" 

    """

    """
    what weights are and why use them?
    When you augment by scaling the input stress as kX, the target also scales:
    for your current first-degree homogeneous yield function,f′(kX)=k. 
    So an absolute error of 0.02 is not equally important for all k.
    For k = 0.5, error 0.02 is a 4% relative error.
    For k = 2.0, error 0.02 is only a 1% relative error.
    you should use a relative / weighted MSE
    weight * (prediction - target)^2
    where weight = 1 / target^2
    """

    binary_operators: list[str] = field(default_factory=lambda: ["+", "-", "*","/"])
    unary_operators: list[str] = field(default_factory=lambda: ["square","sqrt"])

    maxsize: int = 120

    
    """
    Set maxsize a bit larger than the final size you want. e.g., if you want a final equation of size 30,
    you might set this to 35, so that it has a bit of room to explore.
    """
    maxdepth: int | None = None #8
    parsimony: float = 0.0 #1e-3 # Multiplicative factor for how much to punish complexity.  default = 0.0
    complexity_of_constants: int = 2

    """
    NOTES:
    Do not add constraint for the sqrt, it prevents PySR find the actual equation. 
    """
    constraints: dict = field(
        default_factory=lambda: {
            "square": 10,
            "sqrt": 90, #
            "*": (8, 8) #
        }
    )

    nested_constraints: dict = field(
        default_factory=lambda: {
            "square": {"square": 1},
            "sqrt": { #
                "sqrt": 0,
            },
            }
    )

    precision: int = 64
    progress: bool = True
    early_stop_condition:str = "stop_if(loss, complexity) = loss < 1e-17" #(loss < 0.1) && (complexity < 10)



@dataclass(slots=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    invariants: InvariantConfig = field(default_factory=InvariantConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)

    @property
    def experiment_dir(self) -> Path:
        return self.train.results_dir / self.train.run_id

    @property
    def split_path(self) -> Path:
        test_pct = int(round(self.data.test_size * 100))
        seed = self.data.random_state
        name = f"{self.data.dataset_name}_{self.data.stress_format}_test{test_pct}_seed{seed}.npz"
        return self.train.split_dir / name


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _constraint_value_from_toml(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_constraint_value_from_toml(item) for item in value)
    if isinstance(value, dict):
        return {key: _constraint_value_from_toml(item) for key, item in value.items()}
    return value


def _set_known_fields(section_obj: Any, values: dict[str, Any], *, section: str) -> None:
    known = getattr(section_obj, "__dataclass_fields__", {})
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key [{section}].{key}")

        if key in {"data_dir", "results_dir", "split_dir", "stop_file", "output_directory"}:
            value = _resolve_project_path(value)
        elif key == "k_values":
            value = None if value == [] else [float(v) for v in value]
        elif key == "k_range":
            if len(value) != 2:
                raise ValueError("[augmentation].k_range must contain exactly two values.")
            value = (float(value[0]), float(value[1]))
        elif key in {
            "selected",
            "feature_names",
            "hidden_dims",
            "stop_keywords",
            "binary_operators",
            "unary_operators",
        }:
            value = list(value)
        elif key == "maxdepth" and int(value) < 0:
            value = None
        elif key in {"constraints", "nested_constraints"}:
            value = _constraint_value_from_toml(value)

        setattr(section_obj, key, value)


def load_config(path: str | Path | None = None) -> Config:
    """
    Load a TOML config into the dataclass defaults.

    If path is None, returns Config() with built-in defaults.
    """
    config = Config()
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        payload = tomllib.load(f)

    for section, values in payload.items():
        if not hasattr(config, section):
            raise KeyError(f"Unknown config section [{section}]")
        section_obj = getattr(config, section)
        if not isinstance(values, dict):
            raise TypeError(f"Config section [{section}] must be a table.")
        _set_known_fields(section_obj, values, section=section)

    return config
