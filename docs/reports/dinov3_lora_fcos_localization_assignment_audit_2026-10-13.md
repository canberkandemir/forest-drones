# DINOv3 ViT-S LoRA + FCOS Localization and Assignment Audit

Date: 2026-10-13

## Scope

This is a read-only audit of the current single-scale DINOv3 ViT-S + LoRA +
FCOS detector. No training, inference, checkpoint modification, or dataset
download was performed.

Current train5000/val600 baseline:

- AP: 0.3587
- AP50: 0.8139
- AP75: 0.2379
- Recall at FP/image <= 0.425: 0.8465
- Recall at 20% visibility: 0.6667

## Regression Loss

The current box regression loss is not IoU, GIoU, DIoU, or CIoU.

Implementation:

- `scripts/train_dinov3_lora_fcos.py:87-92`
- The model predicts raw FCOS distances.
- Distances are transformed with `softplus`.
- The loss is `SmoothL1Loss(log1p(pred_distance), log1p(target_distance), beta=0.25)`.
- The box loss is weighted by `box_loss_weight = 2.0`.

This is a stable distance-regression objective, but it does not directly
optimize box overlap. That is a likely reason AP50 is strong while AP75 remains
weak.

## Positive Assignment

Implementation:

- `src/forestpersons/dinov3_detector.py:407-483`

The current single-scale FCOS assignment:

- Uses one DINO patch grid at stride 16.
- Scales ground-truth boxes to resized image coordinates with separate x/y
  scaling.
- Marks candidate cells whose centers are inside the GT box.
- Applies center sampling with radius `1.5 * PATCH_SIZE`, i.e. 24 resized
  pixels.
- If no candidate exists, assigns the nearest center patch as a fallback.
- If multiple GT boxes compete for a cell, the smaller-area object wins.
- Encodes box targets as left/top/right/bottom distances divided by 16.
- Computes centerness from the distance balance:
  `sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))`.

Lightweight train5000 assignment audit from manifests and annotation geometry:

| Visibility | GT | Avg positives / GT | Median positives / GT | Fallback GT | Small max-side <32 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 1,422 | 7.10 | 9 | 4 | 4 |
| 40 | 1,448 | 7.77 | 9 | 1 | 0 |
| 70 | 1,593 | 8.25 | 9 | 0 | 0 |
| 100 | 1,968 | 8.29 | 9 | 2 | 0 |

Overall train5000 assignment audit:

- Images: 5,000
- GT annotations: 6,431
- Positive cells: 50,600
- Center fallbacks: 7
- Multi-GT overwrites: 121
- Annotations with at least one positive: 6,431

Interpretation:

- The assignment is not dropping 20%-visibility people entirely.
- 20%-visibility people do get fewer positives on average than clearer people.
- The current assignment looks reasonable for recall, but the small drop in
  positives for 20%-visibility cases may make them more sensitive to loss
  weighting and confidence calibration.

## Centerness

Centerness is used in two places:

- Training: binary cross-entropy on positive cells only,
  `scripts/train_dinov3_lora_fcos.py:229-235`.
- Inference: score is `sqrt(class_score * centerness_score)`,
  `src/forestpersons/dinov3_detector.py:695-697`.

This can suppress useful boxes whose classification score is high but whose
centerness is low. That is expected FCOS behavior, not a confirmed bug. It may
hurt heavily occluded people if the most discriminative patch lies away from the
geometric center, but changing centerness first would also risk increasing false
positives.

## Confirmed Issues vs Limitations

Confirmed implementation issues:

- None found in the audited single-scale FCOS path.
- Coordinate scaling uses separate `scale_x` and `scale_y`.
- All train5000 GT annotations in the current subset receive at least one
  positive assignment.

Architectural/training limitations:

- Localization is trained with distance Smooth L1, not an overlap-aware loss.
- AP75 is much weaker than AP50, suggesting localization precision is the main
  gap.
- 20%-visibility GT receives fewer positive cells on average.
- Centerness may suppress off-center evidence for occluded people, but this is
  not yet proven to be the dominant failure mode.

## Recommended Single Controlled Improvement

Add an IoU-family localization term for positive FCOS cells while keeping the
existing Smooth L1 distance loss.

Recommended first variant:

```text
box_loss = smooth_l1_log_distance_loss + lambda_iou * (1 - GIoU)
```

Start with `lambda_iou = 1.0`, keep the existing `box_loss_weight = 2.0`, and
evaluate whether AP75 improves without damaging matched-FP recall.

Why this is the best next controlled change:

- It directly targets AP75/localization quality.
- It preserves the existing single-scale architecture and assignment.
- It adds little complexity and no new dataset/model dependency.
- It is less likely than centerness changes to destabilize false positives.
- It isolates localization from the failed FPN experiment.

## Fair Comparison Protocol

Compare original vs modified model using:

- Same train5000 subset.
- Same val600 subset.
- Same DINOv3 ViT-S backbone.
- Same LoRA rank 4 q/v targets.
- Same short side 672.
- Same seed: 20261012 or a clearly documented matched seed.
- Same optimizer, learning rates, gradient accumulation, and 2-epoch budget.
- Same initialization:
  - FCOS head initialized from
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`.
  - LoRA freshly initialized, unless explicitly comparing continuation from the
    existing LoRA checkpoint.
- Same canonical val600 evaluation framework.

Primary metrics:

- AP, AP50, AP75, AR@100.
- Recall at FP/image <= 0.425.
- Precision at the matched operating point.
- Recall by visibility, especially 20%.

Success criterion:

- AP75 improves over 0.2379.
- Recall at FP/image <= 0.425 stays close to or above 0.8465.
- 20%-visibility recall stays close to or above 0.6667.

## Decision

Do not change architecture yet. The most promising next experiment is a
single-scale LoRA + FCOS run with an added GIoU localization term.
