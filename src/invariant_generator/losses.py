from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from invariant_generator.config import ConstraintsConfig, LossConfig
from invariant_generator.constraints import fourth_order_to_mandel_matrix, psd_penalty
from invariant_generator.model import InvariantYieldModel


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    param: torch.Tensor
    structure: torch.Tensor
    encoder: torch.Tensor
    constraint: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "data": float(self.data.detach().cpu()),
            "param": float(self.param.detach().cpu()),
            "structure": float(self.structure.detach().cpu()),
            "encoder": float(self.encoder.detach().cpu()),
            "constraint": float(self.constraint.detach().cpu()),
        }


class YieldSurfaceLoss(nn.Module):
    """
    Loss from the attached notes:

        L = L_data + L_param + L_structure + L_enc

    with:
        L_data      = sum_i [f_hat(k_i*sigma_i) - k_i]^2
        L_structure = lambda_structure * ((||a|| - 1)^2 + (||A|| - 1)^2)
        L_enc       = lambda_enc,1 * ||S||_1 / ||S||_2
                    + lambda_enc,2 * ||S||_2

    L_param is standard neural-network L2 regularization on the MLP weights.
    """

    def __init__(
        self,
        config: LossConfig,
        constraints: ConstraintsConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.constraints = constraints

    def forward(
        self,
        model: InvariantYieldModel,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> LossBreakdown:
        if prediction.shape != target.shape:
            target = target.reshape_as(prediction)

        zero = prediction.new_zeros(())

        # Equation (3b): dataset sum of squared homogeneous-target errors.
        data = (prediction - target).pow(2).sum()

        # Standard neural-network regularization, e.g. L2.
        param = self.config.lambda_param * model.neural_parameter_l2()

        structure_base = zero
        for norm in model.structural_norms().values():
            structure_base = structure_base + (norm - 1.0).pow(2)
        structure = self.config.lambda_structure * structure_base

        S = model.encoder_matrix()
        if S is None:
            encoder = zero
        else:
            flat = S.reshape(-1)
            l1 = torch.linalg.vector_norm(flat, ord=1)
            l2 = torch.linalg.vector_norm(flat, ord=2).clamp_min(self.config.eps)
            encoder = (
                self.config.lambda_encoder_l1_ratio * (l1 / l2)
                + self.config.lambda_encoder_l2 * l2
            )

        constraint = zero
        if self.constraints is not None:
            A_psd = self.constraints.A_psd
            if A_psd.enabled and A_psd.mode.lower() == "penalty":
                A = model.invariant_pool.effective_fourth_order_tensor()
                mandel = fourth_order_to_mandel_matrix(A)
                constraint = float(A_psd.penalty_weight) * psd_penalty(
                    mandel,
                    min_eigenvalue=A_psd.min_eigenvalue,
                )

        total = data + param + structure + encoder + constraint
        return LossBreakdown(
            total=total,
            data=data,
            param=param,
            structure=structure,
            encoder=encoder,
            constraint=constraint,
        )
