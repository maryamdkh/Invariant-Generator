from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from invariant_generator.config import Config, coerce_config_dataclasses
from invariant_generator.invariants import InvariantPool


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name!r}")


class SparseInvariantEncoder(nn.Module):
    """
    Linear encoder S from the notes.

    It maps selected invariant features to a smaller or equal feature vector:
        I_tilde = S I

    Sparsity is not enforced here; it is encouraged by L_enc in loss.py.
    """

    def __init__(self, input_dim: int, output_dim: int, *, init: str = "identity") -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.register_parameter("gate_logits", None)
        self.reset_parameters(init=init)

    @property
    def weight(self) -> torch.Tensor:
        if self.gate_logits is not None:
            return self.linear.weight * torch.sigmoid(self.gate_logits)
        return self.linear.weight

    @property
    def raw_weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def gates(self) -> torch.Tensor | None:
        if self.gate_logits is None:
            return None
        return torch.sigmoid(self.gate_logits)

    def enable_gates(self, *, init_probability: float = 0.95) -> None:
        probability = float(init_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("init_probability must be in (0, 1).")
        value = torch.logit(
            self.linear.weight.detach().new_full(self.linear.weight.shape, probability)
        )
        self.gate_logits = nn.Parameter(value)

    def collapse_gates(self) -> None:
        if self.gate_logits is None:
            return
        with torch.no_grad():
            self.linear.weight.mul_(torch.sigmoid(self.gate_logits))
        self.register_parameter("gate_logits", None)

    def reset_parameters(self, *, init: str) -> None:
        init = init.lower()
        with torch.no_grad():
            if init == "identity":
                self.linear.weight.zero_()
                diag = min(self.linear.weight.shape)
                self.linear.weight[:diag, :diag] = torch.eye(diag)
            elif init == "random":
                nn.init.xavier_uniform_(self.linear.weight)
            else:
                raise ValueError("encoder init must be 'identity' or 'random'.")

    def forward(self, invariants: torch.Tensor) -> torch.Tensor:
        return F.linear(invariants, self.weight)


class InvariantStandardizer(nn.Module):
    """Non-trainable normalization for invariant features."""

    VALID_MODES = {"standard", "scale_only"}

    def __init__(
        self,
        input_dim: int,
        *,
        mode: str = "standard",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(
                "normalization mode must be 'standard' or 'scale_only' "
                f"when normalization is enabled, got {mode!r}."
            )
        self.mode = mode
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(input_dim, dtype=torch.float32))

    @property
    def center(self) -> bool:
        return self.mode == "standard"

    @property
    def effective_mean(self) -> torch.Tensor:
        if self.center:
            return self.mean
        return torch.zeros_like(self.mean)

    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError(
                "normalization statistics must match invariant dimension. "
                f"Got mean={tuple(mean.shape)}, std={tuple(std.shape)}, "
                f"expected={tuple(self.mean.shape)}."
            )
        with torch.no_grad():
            self.mean.copy_(mean.detach().to(dtype=self.mean.dtype, device=self.mean.device))
            self.std.copy_(
                std.detach()
                .to(dtype=self.std.dtype, device=self.std.device)
                .clamp_min(self.eps)
            )

    def forward(self, invariants: torch.Tensor) -> torch.Tensor:
        return (invariants - self.effective_mean) / self.std


class MLPRegressor(nn.Module):
    """Small feed-forward network used after invariant generation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "silu",
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers.")
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(_activation(activation))
            current_dim = int(hidden_dim)

        layers.append(nn.Linear(current_dim, 1))

        output_activation = output_activation.lower()
        if output_activation == "softplus":
            layers.append(nn.Softplus())
        elif output_activation in {"none", "linear", ""}:
            pass
        else:
            raise ValueError("output_activation must be 'softplus' or 'none'.")

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class InvariantYieldModel(nn.Module):
    """
    Full pipeline:

        stress -> invariant pool -> optional S encoder -> MLP -> f_hat
    """

    def __init__(
        self,
        invariant_pool: InvariantPool,
        regressor: MLPRegressor,
        *,
        encoder: SparseInvariantEncoder | None = None,
        normalizer: InvariantStandardizer | None = None,
    ) -> None:
        super().__init__()
        self.invariant_pool = invariant_pool
        self.normalizer = normalizer
        self.encoder = encoder
        self.regressor = regressor

    @classmethod
    def from_config(cls, config: Config) -> "InvariantYieldModel":
        coerce_config_dataclasses(config)
        A_psd = config.constraints.A_psd
        A_psd_mode = A_psd.mode.lower()
        A_psd_target = A_psd.target.lower()
        A_psd_basis = A_psd.basis.lower()
        if A_psd.enabled:
            if A_psd_target != "fourth_order_a":
                raise ValueError("constraints.A_psd.target must be 'fourth_order_A'.")
            if A_psd_basis != "mandel":
                raise ValueError("constraints.A_psd.basis must be 'mandel'.")
            if A_psd_mode not in {"hard", "penalty", "check"}:
                raise ValueError(
                    "constraints.A_psd.mode must be 'hard', 'penalty', or 'check'."
                )
            if not config.invariants.enable_fourth_order:
                raise ValueError(
                    "constraints.A_psd requires invariants.enable_fourth_order=true."
                )

        invariant_pool = InvariantPool(
            config.invariants.selected,
            enable_second_order=config.invariants.enable_second_order,
            enable_fourth_order=config.invariants.enable_fourth_order,
            homogenize=config.invariants.homogenize,
            init_scale=config.invariants.init_scale,
            eps=config.invariants.eps,
            fourth_order_psd_mode="hard"
            if A_psd.enabled and A_psd_mode == "hard"
            else "off",
            fourth_order_psd_min_eigenvalue=A_psd.min_eigenvalue,
        )

        invariant_dim = invariant_pool.output_dim
        normalizer: InvariantStandardizer | None = None
        encoder: SparseInvariantEncoder | None = None
        regressor_input_dim = invariant_dim

        normalization_mode = config.normalization.mode.lower()
        if normalization_mode not in {"standard", "scale_only", "none"}:
            raise ValueError(
                "normalization.mode must be 'standard', 'scale_only', or 'none', "
                f"got {config.normalization.mode!r}."
            )

        if config.normalization.enabled and normalization_mode != "none":
            normalizer = InvariantStandardizer(
                invariant_dim,
                mode=normalization_mode,
                eps=config.normalization.eps,
            )

        if config.encoder.enabled:
            output_dim = config.encoder.output_dim or invariant_dim
            encoder = SparseInvariantEncoder(
                invariant_dim,
                output_dim,
                init=config.encoder.init,
            )
            regressor_input_dim = output_dim

        regressor = MLPRegressor(
            regressor_input_dim,
            config.model.hidden_dims,
            activation=config.model.activation,
            output_activation=config.model.output_activation,
        )

        return cls(invariant_pool, regressor, encoder=encoder, normalizer=normalizer)

    def raw_invariant_features(self, stress: torch.Tensor) -> torch.Tensor:
        return self.invariant_pool(stress)

    def normalized_invariant_features(self, stress: torch.Tensor) -> torch.Tensor:
        features = self.raw_invariant_features(stress)
        if self.normalizer is not None:
            features = self.normalizer(features)
        return features

    def invariant_features(self, stress: torch.Tensor) -> torch.Tensor:
        features = self.normalized_invariant_features(stress)
        if self.encoder is not None:
            features = self.encoder(features)
        return features

    def forward(self, stress: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.invariant_features(stress))

    @staticmethod
    def _parameter_count(module: nn.Module | None, *, trainable_only: bool) -> int:
        if module is None:
            return 0
        return sum(
            param.numel()
            for param in module.parameters()
            if not trainable_only or param.requires_grad
        )

    def parameter_counts(self) -> dict[str, int]:
        """Return total/trainable parameter counts by model component."""
        counts = {
            "total": self._parameter_count(self, trainable_only=False),
            "trainable": self._parameter_count(self, trainable_only=True),
            "invariant_pool_total": self._parameter_count(
                self.invariant_pool,
                trainable_only=False,
            ),
            "invariant_pool_trainable": self._parameter_count(
                self.invariant_pool,
                trainable_only=True,
            ),
            "encoder_total": self._parameter_count(self.encoder, trainable_only=False),
            "encoder_trainable": self._parameter_count(self.encoder, trainable_only=True),
            "normalizer_total": self._parameter_count(
                self.normalizer,
                trainable_only=False,
            ),
            "normalizer_trainable": self._parameter_count(
                self.normalizer,
                trainable_only=True,
            ),
            "regressor_total": self._parameter_count(self.regressor, trainable_only=False),
            "regressor_trainable": self._parameter_count(
                self.regressor,
                trainable_only=True,
            ),
        }
        return counts

    def structural_norms(self) -> dict[str, torch.Tensor]:
        return self.invariant_pool.structural_norms()

    def encoder_matrix(self) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        return self.encoder.weight

    def raw_encoder_matrix(self) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        return self.encoder.raw_weight

    def encoder_gates(self) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        return self.encoder.gates

    def invariant_normalization_state(self) -> dict[str, object] | None:
        if self.normalizer is None:
            return None
        return {
            "mode": self.normalizer.mode,
            "mean": [float(value) for value in self.normalizer.mean.detach().cpu()],
            "std": [float(value) for value in self.normalizer.std.detach().cpu()],
            "eps": self.normalizer.eps,
        }

    def neural_parameter_l2(self) -> torch.Tensor:
        """
        L2 penalty for the neural-network regressor only.

        Structural tensors and S have their own dedicated loss terms in the
        notes, so they are intentionally excluded here.
        """
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for param in self.regressor.parameters():
            total = total + param.pow(2).sum()
        return total
