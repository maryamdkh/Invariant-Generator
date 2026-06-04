import torch

from invariant_generator.config import Config
from invariant_generator.losses import YieldSurfaceLoss
from invariant_generator.model import InvariantYieldModel


def test_pdf_loss_terms_match_formula():
    config = Config()
    config.invariants.selected = ["I4", "I11"]
    config.invariants.enable_second_order = True
    config.invariants.enable_fourth_order = True
    config.encoder.enabled = True
    config.encoder.output_dim = 2
    config.model.hidden_dims = []
    config.model.output_activation = "none"
    config.loss.lambda_param = 0.0
    config.loss.lambda_structure = 0.25
    config.loss.lambda_encoder_l1_ratio = 0.5
    config.loss.lambda_encoder_l2 = 0.75

    model = InvariantYieldModel.from_config(config)
    criterion = YieldSurfaceLoss(config.loss)

    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.5, 1.5, 2.0])
    loss = criterion(model, prediction, target)

    expected_data = ((prediction - target) ** 2).sum()

    expected_structure = torch.zeros(())
    for norm in model.structural_norms().values():
        expected_structure = expected_structure + (norm - 1.0) ** 2
    expected_structure = config.loss.lambda_structure * expected_structure

    S = model.encoder_matrix()
    assert S is not None
    flat = S.reshape(-1)
    l1 = torch.linalg.vector_norm(flat, ord=1)
    l2 = torch.linalg.vector_norm(flat, ord=2).clamp_min(config.loss.eps)
    expected_encoder = (
        config.loss.lambda_encoder_l1_ratio * (l1 / l2)
        + config.loss.lambda_encoder_l2 * l2
    )

    torch.testing.assert_close(loss.data, expected_data)
    torch.testing.assert_close(loss.structure, expected_structure)
    torch.testing.assert_close(loss.encoder, expected_encoder)
    torch.testing.assert_close(
        loss.total,
        expected_data + expected_structure + expected_encoder,
    )
