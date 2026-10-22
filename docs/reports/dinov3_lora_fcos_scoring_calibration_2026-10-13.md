# DINOv3 ViT-S LoRA + FCOS Classification/Centerness Scoring Diagnostic

Date: 2026-10-13

## Scope

This experiment tested whether the FCOS confidence score is suppressing useful
detections of heavily occluded people. No model weights were changed, no
training was run, and no datasets were downloaded.

The analysis used the best existing DINOv3 ViT-S + LoRA + single-scale FCOS
checkpoint:

`outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`

## Inference Formula Inspected

The current FCOS inference path is in:

`src/forestpersons/dinov3_detector.py`

For each patch location:

```text
class_score = sigmoid(class_logit)
centerness_score = sigmoid(centerness_logit)
score = sqrt(class_score * centerness_score)
```

The decoder then:

1. Filters candidates by score threshold.
2. Keeps the top 300 scored candidates per image.
3. Decodes L/T/R/B distances to image-space boxes.
4. Applies torchvision NMS with IoU threshold 0.50.

## Candidate Export

Decoded pre-NMS candidates were exported with:

- Box coordinates.
- Classification probability.
- Centerness probability.
- Patch-grid flat index.

Candidate files:

- Train400 calibration:
  `outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013/pre_nms_candidates/train400_candidates.jsonl.gz`
- Val600:
  `outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013/pre_nms_candidates/val600_candidates.jsonl.gz`

Export summary:

| Split | Images | Candidates | Runtime | Seconds/image | File size |
| --- | ---: | ---: | ---: | ---: | ---: |
| train400 calibration | 400 | 1,281,168 | 213.2 s | 0.533 | 64 MiB |
| val600 | 600 | 1,917,636 | 310.6 s | 0.518 | 96 MiB |

The train400 calibration subset is from the official training split and has zero
overlap with val600.

## Tested Scoring Rules

The scoring rules were fixed before evaluation:

| Name | Formula |
| --- | --- |
| `current_sqrt_cls_center` | `sqrt(class_score * centerness_score)` |
| `class_alpha_0` | `class_score` |
| `class_center_alpha_0.5` | `class_score * centerness_score^0.5` |
| `class_center_alpha_1` | `class_score * centerness_score` |

For each rule, candidates were re-ranked, filtered, top-300 limited, and passed
through the same NMS implementation.

## Calibration Selection

Selection used train400 only. The objective was highest recall among evaluated
thresholds with FP/image <= 0.425. Ties were resolved with 20%-visibility recall
and then precision.

Best calibration rule:

| Rule | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current rule | 0.55 | 0.9219 | 0.7732 | 0.3550 | 0.8036 | 0.9412 | 0.9318 | 0.9815 |
| Selected class-only rule | 0.50 | 0.9352 | 0.7519 | 0.4050 | 0.8571 | 0.9328 | 0.9621 | 0.9691 |

On train400, removing centerness from the score improved recall and 20% recall
while staying inside the FP budget. This motivated one locked val600 evaluation
of `class_alpha_0` at threshold 0.50.

## Val600 Result

AP/AP50/AP75/AR@100 below are computed from low-threshold ranked predictions for
each scoring formula. Recall/precision/FP/image use the operating threshold.

| Rule | Threshold | AP | AP50 | AP75 | AR@100 | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current rule | 0.55 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Calibrated class-only rule | 0.50 | 0.3564 | 0.8213 | 0.2299 | 0.4689 | 0.8531 | 0.6733 | 0.5167 | 0.6724 | 0.8118 | 0.9275 | 0.9670 |

The calibrated rule recovered:

- +5 true positives overall.
- +1 true positive at 20% visibility.
- +60 false positives overall.

But it exceeded the target FP/image budget on val600:

```text
target budget:  <= 0.425 FP/image
observed value: 0.517 FP/image
```

Therefore, the selected class-only rule is not acceptable as a replacement
under the required operating budget.

## Budget-Constrained Observation

This was not used for model selection, but it helps interpret the result:

- `class_alpha_0` at threshold 0.55 gives FP/image 0.365, but recall drops to
  0.8318 and R20 drops to 0.6322.
- `class_center_alpha_0.5` at threshold 0.40 gives FP/image 0.380, recall
  0.8425 and R20 0.6667.
- `class_center_alpha_1` at threshold 0.30 gives FP/image 0.4183, recall
  0.8478 and R20 0.6667.
- Current rule at threshold 0.55 gives FP/image 0.4167, recall 0.8465 and R20
  0.6667.

So under the val600 FP budget, the alternative formulas do not show a meaningful
20%-visibility recall improvement over the current rule.

## Interpretation

Measured result:

- Removing or weakening centerness can recover a few low-confidence true
  positives.
- On val600, the gain is mostly paid for by additional false positives.
- The train400-selected class-only rule did not generalize to the val600 FP
  budget.
- AP and AP75 do not improve.

Hypothesis:

- The current centerness formulation is not the dominant bottleneck at the
  accepted FP/image operating point.
- The remaining 20%-visibility gap is more likely a combined localization and
  representation problem than a simple score-suppression problem.

## Limitations

- Calibration used train400, which is small.
- Val600 is still a selected pilot subset, not the full official validation
  split.
- The diagnostic recomputes NMS from decoded pre-NMS grid candidates, but does
  not modify model weights.
- The selected threshold was not tuned on val600; its FP budget failure is a
  real generalization issue, not something corrected after the fact.

## Decision

Do not replace the current FCOS score with class-only scoring. The selected
class-only rule slightly increases recall but violates the FP/image budget and
does not improve AP75 or low-visibility recall enough to justify the extra false
positives.

Recommended next experiment:

- Focus on localization/assignment rather than scoring.
- A practical next controlled test is to increase positive supervision for
  20%-visibility and small persons, for example by adjusting center-sampling
  radius or adding a small-object/low-visibility loss weight, while keeping the
  same checkpoint/evaluation protocol.
