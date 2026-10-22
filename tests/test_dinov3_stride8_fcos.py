import math

import torch

from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3Stride8FCOSHead,
    PreprocessInfo,
    Stride8FeatureAdapter,
    analyze_fcos_positive_locations,
    build_fcos_targets,
    decode_fcos_predictions,
    load_compatible_fcos_weights_into_stride8,
)


def make_info(width: int = 128, height: int = 96) -> PreprocessInfo:
    return PreprocessInfo(width, height, width, height, 1.0, width // 16, height // 16)


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


def test_stride8_head_doubles_feature_map_dimensions() -> None:
    head = DINOv3Stride8FCOSHead(input_dim=384, hidden_dim=16)
    logits, boxes, centerness = head(torch.randn(5, 7, 384))

    assert logits.shape == (10, 14)
    assert boxes.shape == (10, 14, 4)
    assert centerness.shape == (10, 14)


def test_stride8_adapter_starts_as_exact_feature_replication() -> None:
    adapter = Stride8FeatureAdapter(channels=4)
    features = torch.randn(1, 4, 3, 5)

    actual = adapter(features)
    expected = features.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    assert torch.allclose(actual, expected)


def test_stride8_head_reuses_all_compatible_fcos_weights() -> None:
    source = DINOv3FCOSHead(input_dim=384, hidden_dim=16)
    target = DINOv3Stride8FCOSHead(input_dim=384, hidden_dim=16)
    with torch.no_grad():
        source.classification.bias.fill_(0.125)
        source.box.bias.fill_(0.5)

    summary = load_compatible_fcos_weights_into_stride8(target, source.state_dict())

    assert set(summary["loaded"]) == set(source.state_dict())
    assert all(name.startswith("stride8_adapter.") for name in summary["new_or_unloaded"])
    assert torch.allclose(target.classification.bias, source.classification.bias)
    assert torch.allclose(target.box.bias, source.box.bias)


def test_stride8_assignment_increases_locations_and_keeps_tiny_gt_positive() -> None:
    info = make_info()
    ground_truth = [
        {"annotation_id": 1, "box": [20.0, 16.0, 48.0, 44.0], "visibility_level": 20},
        {"annotation_id": 2, "box": [100.0, 70.0, 104.0, 75.0], "visibility_level": 40},
    ]

    stride16 = analyze_fcos_positive_locations(ground_truth, info, stride=16)
    stride8 = analyze_fcos_positive_locations(ground_truth, info, stride=8)

    assert stride8[0]["positive_locations"] > stride16[0]["positive_locations"]
    assert stride16[1]["positive_locations"] >= 1
    assert stride8[1]["positive_locations"] >= 1


def test_stride8_target_encoding_decodes_back_to_original_box() -> None:
    info = make_info(128, 128)
    gt_box = [24.0, 20.0, 70.0, 86.0]
    ground_truth = [{"box": gt_box, "visibility_level": 20}]
    classification, targets, centerness, mask, _stats = build_fcos_targets(
        ground_truth,
        info,
        stride=8,
        grid_height=16,
        grid_width=16,
    )
    positive = torch.nonzero(mask, as_tuple=False)[0]
    gy, gx = [int(value) for value in positive]
    logits = torch.full((16, 16), -20.0)
    center_logits = torch.full((16, 16), -20.0)
    raw_box = torch.zeros((16, 16, 4))
    logits[gy, gx] = 20.0
    center_logits[gy, gx] = 20.0
    raw_box[gy, gx] = inverse_softplus(targets[gy, gx])

    predictions = decode_fcos_predictions(
        logits,
        raw_box,
        center_logits,
        info,
        conf_threshold=0.5,
        stride=8,
    )

    assert len(predictions) == 1
    assert torch.allclose(torch.tensor(predictions[0]["box"]), torch.tensor(gt_box), atol=1e-4)


def test_stride8_decode_uses_independent_xy_scales() -> None:
    info = PreprocessInfo(200, 100, 800, 800, 4.0, 50, 50)
    logits = torch.full((100, 100), -20.0)
    center_logits = torch.full((100, 100), -20.0)
    raw_box = torch.zeros((100, 100, 4))
    logits[20, 40] = 20.0
    center_logits[20, 40] = 20.0

    prediction = decode_fcos_predictions(
        logits,
        raw_box,
        center_logits,
        info,
        conf_threshold=0.5,
        stride=8,
    )[0]
    x1, y1, x2, y2 = prediction["box"]

    assert 78.0 < x1 < 81.0
    assert 19.0 < y1 < 21.0
    assert 81.0 < x2 < 83.0
    assert 21.0 < y2 < 22.0


def test_stride8_losses_are_finite_and_reach_adapter_and_reused_head() -> None:
    torch.manual_seed(3)
    head = DINOv3Stride8FCOSHead(input_dim=32, hidden_dim=16)
    features = torch.randn(4, 4, 32, requires_grad=True)
    logits, raw_box, center_logits = head(features)
    info = make_info(64, 64)
    classification, targets, centerness, mask, _stats = build_fcos_targets(
        [{"box": [12.0, 10.0, 45.0, 54.0], "visibility_level": 20}],
        info,
        stride=8,
        grid_height=8,
        grid_width=8,
    )
    predicted_distances = torch.nn.functional.softplus(raw_box[mask])
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, classification)
        + torch.nn.functional.smooth_l1_loss(predicted_distances, targets[mask])
        + torch.nn.functional.binary_cross_entropy_with_logits(center_logits[mask], centerness[mask])
    )
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert head.stride8_adapter.mix.weight.grad is not None
    assert torch.isfinite(head.stride8_adapter.mix.weight.grad).all()
    assert float(head.stride8_adapter.mix.weight.grad.abs().sum()) > 0
    assert head.classification.weight.grad is not None
    assert float(head.classification.weight.grad.abs().sum()) > 0
    assert features.grad is not None


def test_stride16_defaults_remain_checkpoint_and_decode_compatible() -> None:
    source = DINOv3FCOSHead(input_dim=32, hidden_dim=16)
    restored = DINOv3FCOSHead(input_dim=32, hidden_dim=16)
    restored.load_state_dict(source.state_dict(), strict=True)
    info = make_info(64, 64)
    logits = torch.randn(4, 4)
    raw_box = torch.randn(4, 4, 4)
    centerness = torch.randn(4, 4)

    default = decode_fcos_predictions(logits, raw_box, centerness, info, conf_threshold=0.0)
    explicit = decode_fcos_predictions(logits, raw_box, centerness, info, conf_threshold=0.0, stride=16)

    assert default == explicit
