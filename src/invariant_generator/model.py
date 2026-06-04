from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from invariant_generator.config import Config
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
        self.reset_parameters(init=init)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

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
        return self.linear(invariants)


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
    ) -> None:
        super().__init__()
        self.invariant_pool = invariant_pool
        self.encoder = encoder
        self.regressor = regressor

    @classmethod
    def from_config(cls, config: Config) -> "InvariantYieldModel":
        invariant_pool = InvariantPool(
            config.invariants.selected,
            enable_second_order=config.invariants.enable_second_order,
            enable_fourth_order=config.invariants.enable_fourth_order,
            homogenize=config.invariants.homogenize,
            init_scale=config.invariants.init_scale,
            eps=config.invariants.eps,
        )

        invariant_dim = invariant_pool.output_dim
        encoder: SparseInvariantEncoder | None = None
        regressor_input_dim = invariant_dim

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

        return cls(invariant_pool, regressor, encoder=encoder)

    def invariant_features(self, stress: torch.Tensor) -> torch.Tensor:
        features = self.invariant_pool(stress)
        if self.encoder is not None:
            features = self.encoder(features)
        return features

    def forward(self, stress: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.invariant_features(stress))

    def structural_norms(self) -> dict[str, torch.Tensor]:
        return self.invariant_pool.structural_norms()

    def encoder_matrix(self) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        return self.encoder.weight

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
