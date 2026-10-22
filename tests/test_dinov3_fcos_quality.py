from types import SimpleNamespace

import torch

from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3FCOSQualityHead,
    PreprocessInfo,
    decode_fcos_predictions,
)
from scripts.train_dinov3_lora_fcos import (
    aligned_iou_from_ltrb,
    compute_fcos_losses,
    fcos_quality_loss,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        focal_alpha=0.75,
        focal_gamma=2.0,
        focal_normalization="positive_count",
        giou_loss_weight=0.0,
        box_loss_weight=2.0,
        quality_loss_weight=1.0,
    )


def _info() -> PreprocessInfo:
    return PreprocessInfo(64, 64, 64, 64, 1.0, 4, 4)


def test_quality_head_reuses_legacy_state_and_starts_neutral() -> None:
    legacy = DINOv3FCOSHead(input_dim=32, hidden_dim=16)
    quality = DINOv3FCOSQualityHead(input_dim=32, hidden_dim=16)

    missing, unexpected = quality.load_state_dict(legacy.state_dict(), strict=False)
    outputs = quality(torch.randn(4, 4, 32))

    assert set(missing) == {"quality.weight", "quality.bias"}
    assert unexpected == []
    assert len(outputs) == 4
    assert outputs[3].shape == (4, 4)
    assert torch.allclose(outputs[3].sigmoid(), torch.full((4, 4), 0.5))


def test_aligned_ltrb_iou_identical_and_partial_boxes() -> None:
    target = torch.tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0]])
    pred = torch.tensor([[2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0]])

    iou = aligned_iou_from_ltrb(pred, target)

    assert torch.allclose(iou[0], torch.tensor(1.0))
    assert 0.0 < float(iou[1]) < 1.0


def test_quality_targets_are_detached_and_quality_gradients_are_finite() -> None:
    raw_box = torch.zeros((2, 2, 4), requires_grad=True)
    quality_logits = torch.zeros((2, 2), requires_grad=True)
    box_targets = torch.ones((2, 2, 4))
    positive = torch.tensor([[True, False], [False, True]])

    loss, stats = fcos_quality_loss(quality_logits, raw_box, box_targets, positive)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["count"] == 2
    assert quality_logits.grad is not None and torch.isfinite(quality_logits.grad).all()
    assert raw_box.grad is None


def test_quality_loss_is_positive_only_and_zero_safe() -> None:
    quality_logits = torch.randn((2, 2), requires_grad=True)
    raw_box = torch.zeros((2, 2, 4), requires_grad=True)
    targets = torch.ones((2, 2, 4))
    mask = torch.zeros((2, 2), dtype=torch.bool)

    loss, stats = fcos_quality_loss(quality_logits, raw_box, targets, mask)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert stats["count"] == 0
    assert quality_logits.grad is not None
    assert torch.count_nonzero(quality_logits.grad) == 0


def test_compute_losses_updates_quality_branch_with_finite_gradient() -> None:
    logits = torch.zeros((2, 2), requires_grad=True)
    raw_box = torch.zeros((2, 2, 4), requires_grad=True)
    centerness_logits = torch.zeros((2, 2), requires_grad=True)
    quality_logits = torch.zeros((2, 2), requires_grad=True)
    classification = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.ones((2, 2, 4))
    centerness = torch.ones((2, 2))
    mask = classification.bool()

    loss, parts = compute_fcos_losses(
        logits,
        raw_box,
        centerness_logits,
        classification,
        targets,
        centerness,
        mask,
        _args(),
        quality_logits=quality_logits,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["quality"] > 0.0
    assert quality_logits.grad is not None and torch.isfinite(quality_logits.grad).all()


def test_quality_power_zero_preserves_canonical_decoder_exactly() -> None:
    logits = torch.tensor([[4.0, 2.0], [-3.0, -4.0]])
    centerness = torch.tensor([[3.0, 1.0], [-2.0, -3.0]])
    quality = torch.tensor([[-8.0, 8.0], [8.0, 8.0]])
    raw_box = torch.zeros((2, 2, 4))

    legacy = decode_fcos_predictions(logits, raw_box, centerness, _info(), conf_threshold=0.01)
    ablation = decode_fcos_predictions(
        logits,
        raw_box,
        centerness,
        _info(),
        conf_threshold=0.01,
        quality_logits=quality,
        quality_score_power=0.0,
    )

    assert legacy == ablation


def test_quality_aware_scoring_changes_candidate_order_before_nms() -> None:
    logits = torch.tensor([[4.0, 3.0]])
    centerness = torch.tensor([[4.0, 3.0]])
    quality = torch.tensor([[-5.0, 5.0]])
    raw_box = torch.zeros((1, 2, 4))
    info = PreprocessInfo(32, 16, 32, 16, 1.0, 2, 1)

    predictions = decode_fcos_predictions(
        logits,
        raw_box,
        centerness,
        info,
        conf_threshold=0.0,
        nms_threshold=1.0,
        topk=1,
        quality_logits=quality,
        quality_score_power=1.0,
    )

    assert len(predictions) == 1
    assert predictions[0]["box"][0] > 8.0
