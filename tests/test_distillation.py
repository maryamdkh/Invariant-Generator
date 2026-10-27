import torch

from invariant_generator.distillation import DistillationNet, DistillationSettings


def test_latent_penalty_is_unchanged_by_rescaling_latent_and_decoder():
    torch.manual_seed(0)
    net = DistillationNet(3, DistillationSettings(latent_dim=4, hidden=8))
    x = torch.randn(16, 3)

    def penalty() -> torch.Tensor:
        _, z = net(x)
        return torch.mean(torch.abs(z) * net.latent_weight_norms())

    before = penalty()
    prediction_before, _ = net(x)
    with torch.no_grad():  # z -> z / 10, decoder input weights -> * 10: same network output
        net.encoder[-1].weight /= 10
        net.encoder[-1].bias /= 10
        net.decoder[0].weight *= 10
    prediction_after, _ = net(x)
    assert torch.allclose(prediction_before, prediction_after, atol=1e-5)
    assert torch.allclose(before, penalty(), rtol=1e-5)
