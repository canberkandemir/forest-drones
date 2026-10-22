# DINOv3 ViT-S LoRA + FCOS GIoU Localization Loss Pilot

Date: 2026-10-13

## Scope

This experiment tested one controlled localization-loss change for the existing
single-scale DINOv3 ViT-S + LoRA + FCOS detector.

Change tested:

```text
box_loss = smooth_l1_log_distance_loss + lambda_iou * (1 - GIoU)
```

The original detector behavior is preserved when `lambda_iou = 0`.

No train5000 or overnight training was started. Both comparison runs used the
existing ForestPersons train400 subset and were initialized from the same
trained ViT-S LoRA + FCOS train5000 checkpoint.

## Implementation

Files:

- `scripts/train_dinov3_lora_fcos.py`
- `tests/test_fcos_giou_loss.py`

Implementation details:

- Predicted FCOS L/T/R/B distances are transformed with `softplus`.
- Target L/T/R/B distances are kept in the same patch-grid distance coordinate
  system.
- Predicted and target distances are decoded to centered `xyxy` boxes before
  computing GIoU.
- Degenerate boxes are clamped with a small epsilon for numerical stability.
- The GIoU path is differentiable and contributes gradients to the box
  prediction branch.
- `--giou-loss-weight 0.0` exactly preserves the previous Smooth L1-only loss.
- Training logs now record total box loss, Smooth L1 component, and GIoU
  component separately.

Unit tests cover:

- Identical boxes.
- Partial overlap.
- Disjoint boxes.
- Finite gradients.
- `lambda_iou = 0` equivalence with the original Smooth L1 distance loss.

## Training Configuration

Shared configuration for both controlled runs:

- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Detector: single-scale FCOS head.
- LoRA: rank 4, q/v projections.
- Original DINOv3 weights frozen.
- Dataset: existing ForestPersons train400 subset.
- Evaluation: existing val600 subset.
- Resolution: short side 672.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Device: Apple MPS.
- Initialization checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

## Smoke Test

A 2-image MPS forward/backward smoke test with `lambda_iou = 1` completed
successfully.

- Loss: 0.9958.
- Smooth L1 box component: 0.0120.
- GIoU box component: 0.2035.
- LoRA gradients: 48/48 trainable tensors nonzero.
- Head gradients: 18/18 trainable tensors nonzero.
- Peak MPS driver memory: approximately 5.52 GiB.
- Runtime: 3.37 seconds.

## Training Results

| Run | Lambda | Epoch 1 loss | Epoch 2 loss | Epoch 2 Smooth L1 | Epoch 2 GIoU | Optimizer steps | Total time | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train400 continuation | 0 | 0.6174 | 0.6101 | 0.0100 | 0.0000 | 200 | 854.3 s | 10.62 GiB |
| Train400 continuation | 1 | 0.9365 | 0.9028 | 0.0109 | 0.1452 | 200 | 936.4 s | 10.53 GiB |

No MPS out-of-memory or non-finite loss occurred.

## Val600 Results

Evaluation used the canonical ForestPersons framework. The matched-FP row is
the best evaluated threshold with FP/image <= 0.425.

| Model | AP | AP50 | AP75 | AR@100 | Conf | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-scale LoRA + FCOS train5000 reference | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Train400 continuation, lambda=0 | 0.3495 | 0.8117 | 0.2326 | 0.4657 | 0.60 | 0.8171 | 0.7574 | 0.3267 | 0.6092 | 0.7765 | 0.8912 | 0.9528 |
| Train400 continuation, lambda=1 | 0.3516 | 0.8125 | 0.2321 | 0.4685 | 0.60 | 0.8198 | 0.7599 | 0.3233 | 0.6322 | 0.7706 | 0.8860 | 0.9528 |

## Interpretation

Against the matched `lambda=0` continuation run, `lambda=1` produced:

- AP: +0.0021.
- AP50: +0.0008.
- AP75: -0.0005.
- AR@100: +0.0028.
- Recall at FP/image <= 0.425: +0.0027.
- Precision at the matched operating point: +0.0025.
- FP/image at the matched operating point: -0.0033.
- 20%-visibility recall: +0.0230.

So the GIoU term gave a small recall and 20%-visibility gain in this bounded
train400 continuation, but it did not improve AP75. Because AP75 was the main
localization target, this is a mixed result rather than a clear win.

Both train400 continuation runs remain below the original train5000 LoRA
reference in AP, AP75, matched-FP recall, and 20%-visibility recall. That is
expected because this was a short train400 continuation experiment, not a new
matched train5000 run from the original protocol.

## Artifacts

- Lambda 0 checkpoint:
  `outputs/dinov3_lora_giou/train400_lambda0_seed20261013_from_lora5k_e2/dinov3_lora_fcos.pt`
- Lambda 1 checkpoint:
  `outputs/dinov3_lora_giou/train400_lambda1_seed20261013_from_lora5k_e2/dinov3_lora_fcos.pt`
- Lambda 0 evaluation:
  `outputs/dinov3_lora_giou/train400_lambda0_seed20261013_from_lora5k_e2/eval_val600/summary.json`
- Lambda 1 evaluation:
  `outputs/dinov3_lora_giou/train400_lambda1_seed20261013_from_lora5k_e2/eval_val600/summary.json`
- Comparison CSV:
  `outputs/dinov3_lora_giou/train400_lambda_comparison_seed20261013.csv`

## Decision

Do not scale this exact GIoU setting to train5000 solely for AP75. The pilot did
not show the desired localization improvement. If the next priority is
20%-visibility recall rather than AP75, a longer or lower-weight GIoU run could
be considered, but the evidence is not strong enough to replace the current
single-scale train5000 LoRA baseline.
