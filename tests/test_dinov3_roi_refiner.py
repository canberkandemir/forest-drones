from pathlib import Path

import torch

from forestpersons.dinov3_detector import PreprocessInfo
from forestpersons.dinov3_roi_refiner import (
    DINOv3RoIRefinementHead,
    decode_box_deltas,
    decode_fcos_pre_nms_proposals,
    encode_box_deltas,
    ground_truth_boxes_resized,
    load_roi_refiner_checkpoint,
    match_and_sample_proposals,
    refine_and_nms,
    roi_refinement_loss,
    save_roi_refiner_checkpoint,
)


def make_info() -> PreprocessInfo:
    return PreprocessInfo(200, 100, 800, 800, 4.0, 50, 50)


def test_ground_truth_transform_uses_independent_xy_scales() -> None:
    boxes = ground_truth_boxes_resized([{"box": [10.0, 10.0, 30.0, 40.0]}], make_info())
    assert torch.allclose(boxes, torch.tensor([[40.0, 80.0, 120.0, 320.0]]))


def test_box_delta_round_trip() -> None:
    proposals = torch.tensor([[10.0, 20.0, 50.0, 80.0], [4.0, 8.0, 20.0, 24.0]])
    targets = torch.tensor([[12.0, 18.0, 58.0, 86.0], [5.0, 6.0, 24.0, 30.0]])
    decoded = decode_box_deltas(proposals, encode_box_deltas(proposals, targets))
    assert torch.allclose(decoded, targets, atol=1e-5)


def test_proposal_matching_samples_positive_negative_and_ignores_middle() -> None:
    proposals = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 7.0, 7.0], [20.0, 20.0, 30.0, 30.0]]
    )
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    sample = match_and_sample_proposals(proposals, gt, samples_per_image=8)
    sampled_labels = sample["labels"].tolist()
    assert 1 in sampled_labels
    assert 0 in sampled_labels
    assert 1 not in sample["indices"].tolist()


def test_refinement_head_and_losses_have_finite_gradients() -> None:
    head = DINOv3RoIRefinementHead(input_dim=32, hidden_dim=16, output_size=3)
    features = torch.randn(8, 8, 32)
    proposals = torch.tensor([[8.0, 8.0, 64.0, 64.0], [32.0, 32.0, 96.0, 96.0]])
    logits, deltas = head(features, proposals)
    labels = torch.tensor([1, 0])
    matched_gt = torch.tensor([[10.0, 10.0, 60.0, 62.0], [10.0, 10.0, 60.0, 62.0]])
    loss, parts = roi_refinement_loss(logits, deltas, proposals, labels, matched_gt)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["positive"] == 1
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_pre_nms_decode_and_refined_inference_preserve_coordinates() -> None:
    info = PreprocessInfo(64, 64, 64, 64, 1.0, 2, 2)
    cls = torch.full((2, 2), -20.0)
    center = torch.full((2, 2), -20.0)
    cls[0, 0] = 20.0
    center[0, 0] = 20.0
    raw = torch.zeros((2, 2, 4))
    proposals = decode_fcos_pre_nms_proposals(cls, raw, center, info, min_score=0.5)
    assert proposals["boxes"].shape == (1, 4)
    detections = refine_and_nms(
        proposals["boxes"],
        proposals["scores"],
        torch.tensor([20.0]),
        torch.zeros((1, 4)),
        info,
        confidence_threshold=0.5,
    )
    assert len(detections) == 1
    assert torch.allclose(torch.tensor(detections[0]["box"]), proposals["boxes"][0], atol=1e-5)


def test_refiner_checkpoint_round_trip(tmp_path: Path) -> None:
    head = DINOv3RoIRefinementHead(input_dim=32, hidden_dim=16, output_size=3)
    path = tmp_path / "refiner.pt"
    save_roi_refiner_checkpoint(
        path,
        head,
        source_detector_checkpoint=Path("source.pt"),
        settings={"seed": 7},
        history=[{"epoch": 1, "loss": 0.5}],
    )
    loaded, checkpoint = load_roi_refiner_checkpoint(path, torch.device("cpu"))
    assert checkpoint["settings"]["seed"] == 7
    assert checkpoint["history"][0]["loss"] == 0.5
    for name, tensor in head.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


def test_proposal_score_mode_does_not_use_quality_logit() -> None:
    info = PreprocessInfo(64, 64, 64, 64, 1.0, 4, 4)
    proposals = torch.tensor([[4.0, 4.0, 28.0, 28.0], [32.0, 32.0, 60.0, 60.0]])
    proposal_scores = torch.tensor([0.8, 0.2])
    detections = refine_and_nms(
        proposals,
        proposal_scores,
        torch.tensor([-20.0, 20.0]),
        torch.zeros((2, 4)),
        info,
        score_mode="proposal",
        confidence_threshold=0.0,
    )
    assert [round(item["confidence"], 3) for item in detections] == [0.8, 0.2]
