import torch

from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3FPNFCOSHead,
    PreprocessInfo,
    build_fpn_fcos_targets,
    decode_fpn_fcos_predictions,
    load_compatible_fcos_weights_into_fpn,
)


def test_fpn_fcos_head_outputs_three_level_shapes() -> None:
    head = DINOv3FPNFCOSHead(input_dim=384, hidden_dim=16)
    features = torch.randn(5, 7, 384)

    outputs = head(features)

    assert [output["level"] for output in outputs] == ["fine", "medium", "coarse"]
    assert outputs[0]["class_logits"].shape == (5, 7)
    assert outputs[0]["box_raw"].shape == (5, 7, 4)
    assert outputs[1]["class_logits"].shape == (3, 4)
    assert outputs[1]["box_raw"].shape == (3, 4, 4)
    assert outputs[2]["class_logits"].shape == (2, 2)
    assert outputs[2]["box_raw"].shape == (2, 2, 4)


def test_fpn_fcos_targets_use_scale_aware_assignment() -> None:
    info = PreprocessInfo(
        original_width=256,
        original_height=256,
        resized_width=256,
        resized_height=256,
        scale=1.0,
        grid_width=16,
        grid_height=16,
    )
    ground_truth = [
        {"box": [32.0, 32.0, 56.0, 56.0], "visibility_level": 20},
        {"box": [48.0, 48.0, 224.0, 224.0], "visibility_level": 100},
    ]

    targets, stats = build_fpn_fcos_targets(
        ground_truth,
        info,
        level_shapes=[("fine", 16, 16, 16), ("medium", 32, 8, 8), ("coarse", 64, 4, 4)],
    )
    by_level = {target["level"]: target for target in targets}

    assert stats["annotations_with_positive"] == 2
    assert int(by_level["fine"]["positive_mask"].sum()) > 0
    assert int(by_level["coarse"]["positive_mask"].sum()) > 0
    assert torch.all(by_level["fine"]["box_targets"][by_level["fine"]["positive_mask"]] > 0)
    assert torch.all(by_level["coarse"]["box_targets"][by_level["coarse"]["positive_mask"]] > 0)


def test_fpn_fcos_decode_uses_separate_xy_scales() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=50,
        grid_height=50,
    )
    output = {
        "level": "fine",
        "stride": 16,
        "class_logits": torch.full((50, 50), -20.0),
        "box_raw": torch.zeros((50, 50, 4)),
        "centerness_logits": torch.full((50, 50), -20.0),
    }
    output["class_logits"][10, 20] = 20.0
    output["centerness_logits"][10, 20] = 20.0

    predictions = decode_fpn_fcos_predictions([output], info, conf_threshold=0.5)

    assert len(predictions) == 1
    x1, y1, x2, y2 = predictions[0]["box"]
    assert 70.0 < x1 < 90.0
    assert 18.0 < y1 < 21.0
    assert 75.0 < x2 < 90.0
    assert 21.0 < y2 < 23.0
    assert predictions[0]["confidence"] > 0.99


def test_fpn_can_reuse_compatible_single_scale_fcos_weights() -> None:
    single = DINOv3FCOSHead(input_dim=384, hidden_dim=16)
    fpn = DINOv3FPNFCOSHead(input_dim=384, hidden_dim=16)
    with torch.no_grad():
        single.classification.bias.fill_(0.123)
        single.box.bias.fill_(0.456)

    summary = load_compatible_fcos_weights_into_fpn(fpn, single.state_dict())

    assert "classification.bias" in summary["loaded"]
    assert "box.bias" in summary["loaded"]
    assert "pyramid_adapters.medium.0.weight" in summary["new_or_unloaded"]
    assert torch.allclose(fpn.classification.bias, torch.full_like(fpn.classification.bias, 0.123))
    assert torch.allclose(fpn.box.bias, torch.full_like(fpn.box.bias, 0.456))
