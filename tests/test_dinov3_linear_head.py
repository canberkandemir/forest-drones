import torch

from forestpersons.dinov3_detector import (
    DINOv3LinearPatchHead,
    PreprocessInfo,
    build_dense_targets,
    decode_predictions,
)


def test_linear_patch_head_output_shapes_and_parameter_count() -> None:
    head = DINOv3LinearPatchHead(input_dim=384)
    features = torch.zeros((3, 4, 384), dtype=torch.float32)

    logits, raw_box = head(features)

    assert logits.shape == (3, 4)
    assert raw_box.shape == (3, 4, 4)
    assert sum(parameter.numel() for parameter in head.parameters()) == 1925


def test_dense_targets_use_separate_xy_scales() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=50,
        grid_height=50,
    )
    ground_truth = [{"box": [50.0, 25.0, 70.0, 45.0], "visibility_level": 100}]

    objectness, box_targets, box_mask, collisions = build_dense_targets(ground_truth, info)

    assert collisions == 0
    assert int(box_mask.sum()) == 1
    gy, gx = torch.nonzero(box_mask, as_tuple=False)[0].tolist()
    assert objectness[gy, gx] == 1.0
    assert gx == 15
    assert gy == 17
    assert torch.isclose(box_targets[gy, gx, 2].exp(), torch.tensor(5.0))
    assert torch.isclose(box_targets[gy, gx, 3].exp(), torch.tensor(10.0))


def test_decode_predictions_returns_original_image_coordinates() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=2,
        grid_height=2,
    )
    logits = torch.full((2, 2), -10.0)
    logits[0, 0] = 10.0
    raw_box = torch.zeros((2, 2, 4))
    raw_box[0, 0, 2] = torch.log(torch.tensor(4.0))
    raw_box[0, 0, 3] = torch.log(torch.tensor(2.0))

    predictions = decode_predictions(logits, raw_box, info, conf_threshold=0.5, nms_threshold=0.5)

    assert len(predictions) == 1
    box = predictions[0]["box"]
    assert box == [0.0, 0.0, 10.0, 3.0]
    assert predictions[0]["confidence"] > 0.99


def test_decode_predictions_filters_low_confidence() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=2,
        grid_height=2,
    )
    logits = torch.full((2, 2), -10.0)
    raw_box = torch.zeros((2, 2, 4))

    assert decode_predictions(logits, raw_box, info, conf_threshold=0.5) == []
