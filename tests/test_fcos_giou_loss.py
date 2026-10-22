import torch

from scripts.train_dinov3_lora_fcos import (
    fcos_box_loss,
    generalized_iou_loss_from_ltrb,
    generalized_iou_loss_xyxy,
)


def test_giou_loss_identical_boxes_is_zero() -> None:
    distances = torch.tensor([[2.0, 3.0, 4.0, 5.0]])

    loss = generalized_iou_loss_from_ltrb(distances, distances)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_giou_loss_partial_overlap_between_zero_and_two() -> None:
    pred = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    target = torch.tensor([[1.0, 1.0, 4.0, 4.0]])

    loss = generalized_iou_loss_from_ltrb(pred, target)

    assert 0.0 < float(loss) < 2.0


def test_giou_loss_disjoint_centered_boxes_is_finite() -> None:
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[3.0, 0.0, 4.0, 1.0]])

    loss = generalized_iou_loss_xyxy(pred, target)

    assert torch.isfinite(loss)
    assert float(loss) > 1.0


def test_fcos_box_loss_has_finite_gradients_with_giou() -> None:
    raw_box = torch.zeros((2, 2, 4), requires_grad=True)
    targets = torch.ones((2, 2, 4))
    mask = torch.tensor([[True, False], [False, True]])

    loss, parts = fcos_box_loss(raw_box, targets, mask, giou_loss_weight=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["smooth_l1"] >= 0.0
    assert parts["giou"] >= 0.0
    assert raw_box.grad is not None
    assert torch.isfinite(raw_box.grad).all()
    assert bool((raw_box.grad[mask].abs() > 0).any())


def test_lambda_zero_matches_original_smooth_l1_distance_loss() -> None:
    raw_box = torch.tensor([[[0.2, -0.3, 0.4, -0.5]]])
    targets = torch.tensor([[[1.2, 0.8, 1.5, 0.7]]])
    mask = torch.tensor([[True]])

    loss, parts = fcos_box_loss(raw_box, targets, mask, giou_loss_weight=0.0)
    pred = torch.nn.functional.softplus(raw_box[mask])
    expected = torch.nn.functional.smooth_l1_loss(torch.log1p(pred), torch.log1p(targets[mask]), beta=0.25)

    assert torch.allclose(loss, expected)
    assert parts["giou"] == 0.0
