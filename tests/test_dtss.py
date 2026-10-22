from __future__ import annotations

import pytest
import torch

from forestpersons.dtss import (
    DTSSLIFLinear,
    DTSSSpikingMLP,
    DTSSSpikingTransformerBlock,
    DynamicTimeStepMask,
)


def assert_prefix_mask(mask: torch.Tensor) -> None:
    values = mask.detach().cpu().tolist()
    assert values[0] == 1.0
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_dtss_scores_and_mask_form_consecutive_prefix() -> None:
    dtss = DynamicTimeStepMask(t_max=4, initial_timesteps=2)
    diagnostics = dtss()
    assert diagnostics.scores.shape == (4,)
    assert torch.all(diagnostics.scores[:-1] >= diagnostics.scores[1:])
    assert torch.equal(diagnostics.mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert_prefix_mask(diagnostics.mask)


def test_dtss_remains_prefix_after_parameter_change() -> None:
    dtss = DynamicTimeStepMask(t_max=4, initial_timesteps=2)
    with torch.no_grad():
        dtss.raw_timestep_parameters.copy_(torch.tensor([-4.0, 2.0, -3.0, 0.5]))
    diagnostics = dtss()
    assert torch.all(diagnostics.scores[:-1] >= diagnostics.scores[1:])
    assert_prefix_mask(diagnostics.mask)


def test_dtss_spiking_mlp_shapes_and_gradients_cpu() -> None:
    torch.manual_seed(20261020)
    model = DTSSSpikingMLP(6, 8, 3, t_max=4, initial_timesteps=2)
    inputs = torch.randn(4, 5, 6)
    output, diagnostics = model(inputs)
    assert output.shape == (4, 5, 3)
    for item in diagnostics:
        assert item.mask.shape == (4,)
        assert_prefix_mask(item.mask)

    loss = output.mean() + 0.01 * model.mask_loss()
    loss.backward()
    timestep_gradients = [
        model.hidden.dtss.raw_timestep_parameters.grad,
        model.output.dtss.raw_timestep_parameters.grad,
    ]
    assert all(gradient is not None for gradient in timestep_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in timestep_gradients if gradient is not None)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in timestep_gradients if gradient is not None)
    assert model.hidden.linear.weight.grad is not None
    assert torch.isfinite(model.hidden.linear.weight.grad).all()


def test_dtss_lif_rejects_wrong_time_dimension() -> None:
    layer = DTSSLIFLinear(3, 4, t_max=4)
    with pytest.raises(ValueError, match="expected T=4"):
        layer(torch.randn(3, 2, 3))


def test_spiking_transformer_shape_masks_and_cpu_gradients() -> None:
    torch.manual_seed(20261020)
    block = DTSSSpikingTransformerBlock(16, 4, t_max=4, initial_timesteps=2)
    inputs = torch.randn(4, 2, 5, 16, requires_grad=True)
    output, diagnostics = block(inputs)
    assert output.shape == inputs.shape
    assert set(diagnostics) == {
        "query", "key", "value", "attention", "projection", "mlp_hidden", "mlp_output"
    }
    for item in diagnostics.values():
        assert_prefix_mask(item.mask)
        assert torch.equal(item.mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    loss = output.square().mean() + 0.001 * block.mask_loss()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    parameter_gradients = [parameter.grad for parameter in block.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients if gradient is not None)
    timestep_gradients = [
        parameter.grad
        for name, parameter in block.named_parameters()
        if name.endswith("raw_timestep_parameters")
    ]
    assert len(timestep_gradients) == 7
    assert all(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in timestep_gradients)


def test_spiking_transformer_rejects_invalid_embedding() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DTSSSpikingTransformerBlock(15, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_dtss_spiking_mlp_mps_forward_backward() -> None:
    torch.manual_seed(20261020)
    device = torch.device("mps")
    model = DTSSSpikingMLP(6, 8, 3, t_max=4, initial_timesteps=2).to(device)
    inputs = torch.randn(4, 2, 6, device=device)
    output, diagnostics = model(inputs)
    loss = output.square().mean() + 0.01 * model.mask_loss()
    loss.backward()
    torch.mps.synchronize()
    assert output.shape == (4, 2, 3)
    assert all(item.mask.device.type == "mps" for item in diagnostics)
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)
    assert all(torch.equal(item.mask, expected) for item in diagnostics)
    gradient = model.hidden.dtss.raw_timestep_parameters.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_spiking_transformer_mps_forward_backward() -> None:
    torch.manual_seed(20261020)
    device = torch.device("mps")
    block = DTSSSpikingTransformerBlock(16, 4, t_max=4, initial_timesteps=2).to(device)
    inputs = torch.randn(4, 2, 5, 16, device=device, requires_grad=True)
    output, diagnostics = block(inputs)
    loss = output.square().mean() + 0.001 * block.mask_loss()
    loss.backward()
    torch.mps.synchronize()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(item.mask.device.type == "mps" for item in diagnostics.values())
    timestep_gradients = [
        parameter.grad
        for name, parameter in block.named_parameters()
        if name.endswith("raw_timestep_parameters")
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in timestep_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in timestep_gradients if gradient is not None)

