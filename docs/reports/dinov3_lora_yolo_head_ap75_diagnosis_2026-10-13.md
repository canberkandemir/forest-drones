# DINOv3 LoRA YOLO-Style Head AP75 Diagnosis

Date: 2026-10-13

## Scope

This report diagnoses why the DINOv3 ViT-S + LoRA + YOLO-style DFL head has
very weak AP75 after the train400 2-epoch pilot.

No train400/train5000 retraining, architecture changes, or checkpoint
modifications were performed.

## Inputs

- Checkpoint:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt`
- Saved val600 predictions:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/coco_results.json`
- Evaluation summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/summary.json`

## Geometry and DFL Audit

No confirmed x/y scaling or DFL encoding bug was found.

Verified from code and tests:

- DFL targets are encoded as L/T/R/B distances in stride units.
- `reg_max=32` means bins 0..32, so maximum unclipped side distance is about
  `32 * 16 = 512` resized pixels.
- Training clamps DFL targets to `[0, reg_max)`.
- The train400 pilot recorded `clipped_target_distances = 0` in both epochs,
  so `reg_max` was sufficient for the assigned positives.
- Decoding uses expected distance bins and multiplies by stride 16.
- Original-image conversion uses separate `scale_x` and `scale_y`.
- No padding is used; images are resized directly to a patch-aligned grid.
- Existing tests cover separate x/y scaling and DFL decoding.

The saved val600 boxes also do not show a simple global size bug:

| Quantity | Median | Mean |
| --- | ---: | ---: |
| Predicted/GT width ratio | 1.043 | 1.079 |
| Predicted/GT height ratio | 0.898 | 0.932 |
| Predicted/GT area ratio | 0.905 | 1.034 |
| Normalized center error | 0.090 | 0.097 |

There is a mild tendency toward boxes that are too wide and too short, but not
a catastrophic coordinate transform error.

## Assignment and Loss Audit

Task-aligned assignment produced meaningful positives:

| Epoch | Positives/image | Annotated GT with positives | Center fallbacks | Overwrites | Clipped distances |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.70 | 524 | 0 | 32 | 0 |
| 2 | 11.71 | 525 | 0 | 28 | 0 |

The losses were finite and decreased:

| Epoch | Total | Classification | IoU | DFL |
| --- | ---: | ---: | ---: | ---: |
| 1 | 44.8272 | 36.6765 | 0.6439 | 2.2143 |
| 2 | 9.0459 | 3.3684 | 0.4191 | 1.6896 |

Gradients were nonzero for all LoRA and head tensors in the final epoch.

Potential limitation:

- The classifier target uses a normalized task-alignment metric. It is not a
  direct localization-quality score at inference time.
- Inference ranks boxes with class probability only. There is no centerness,
  predicted IoU, or quality score.

## Best-Candidate IoU From Saved Post-NMS Predictions

Using saved val600 predictions generated at `prediction_conf_min=0.001`, each
ground-truth person was matched to its highest-IoU saved candidate.

| Metric | Value |
| --- | ---: |
| GT persons | 749 |
| Mean best IoU | 0.627 |
| Median best IoU | 0.641 |
| GT with best IoU >= 0.50 | 89.05% |
| GT with best IoU >= 0.75 | 9.88% |
| Among GT with IoU >= 0.50, score >= 0.60 | 35.23% |

Visibility breakdown:

| Visibility | GT | Best IoU >= 0.50 | Best IoU >= 0.75 | Score >= 0.60 given IoU >= 0.50 |
| --- | ---: | ---: | ---: | ---: |
| 20 | 174 | 80.46% | 4.60% | 30.71% |
| 40 | 170 | 85.29% | 10.59% | 32.41% |
| 70 | 193 | 90.67% | 8.81% | 43.43% |
| 100 | 212 | 97.64% | 14.62% | 33.33% |

This explains the metric pattern:

- AP50 is moderate because many GT persons have at least one candidate above
  IoU 0.50.
- AP75 is extremely low because few post-NMS candidates reach IoU 0.75.
- The fixed-threshold recall is also low because many IoU >= 0.50 candidates
  have scores below the accepted operating threshold.

## Comparison With FCOS Localization

Using the same saved-prediction best-candidate analysis:

| Model | Best IoU >= 0.50 | Best IoU >= 0.75 | Median best IoU |
| --- | ---: | ---: | ---: |
| DINOv3 LoRA + FCOS train5000 | 92.12% | 41.26% | 0.727 |
| DINOv3 LoRA + YOLO-style train400 | 89.05% | 9.88% | 0.641 |

FCOS is not just better at confidence calibration; it has far more high-IoU
surviving candidates.

## Pre-NMS Check

Because the saved COCO predictions are post-NMS, a small bounded pre-NMS check
was run on the first 30 val600 images. No training was performed.

Result:

| Subset | GT | Pre-NMS IoU >= 0.50 | Post-NMS IoU >= 0.50 | Pre-NMS IoU >= 0.75 | Post-NMS IoU >= 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First 30 val600 images | 33 | 93.94% | 93.94% | 36.36% | 3.03% |

In this subset, 11 GT instances had a pre-NMS candidate with IoU >= 0.75 that
was lost after score-ranked NMS.

This is the strongest confirmed finding:

- The model can sometimes produce well-localized boxes.
- Those boxes are often not the highest class-scored boxes in their local
  neighborhood.
- Class-score-only NMS suppresses high-IoU candidates in favor of worse boxes.

## Confirmed Findings

1. No confirmed DFL bin, stride, coordinate scale, or clipping bug was found.
2. The head trains stably and produces meaningful positives.
3. Post-NMS localization quality is poor: only 9.88% of GT have a saved
   candidate above IoU 0.75.
4. A bounded pre-NMS check shows that well-localized candidates can exist before
   NMS but are often suppressed.
5. The current class-only confidence score is not localization-aware enough for
   NMS or AP75.

## Hypotheses

- The classification branch is learning object presence but not box quality.
- Normalized task-aligned labels are not enough to make the class score a
  reliable localization-quality score after only 2 epochs.
- Single-scale stride-16 features still limit localization for small/heavily
  occluded people, but the pre-NMS result shows scoring/NMS is currently a more
  immediate bottleneck than raw box generation.

## Recommendation

Do not scale this exact YOLO-style head to train5000 yet.

The single most justified correction is to add localization-aware scoring before
the next train400 pilot. A minimal controlled option:

- add a lightweight quality/objectness branch trained to predict IoU of the
  assigned box, or change the classification target/loss to a quality-focal or
  varifocal-style target tied directly to IoU;
- rank and NMS detections with `class_score * quality_score`;
- keep the backbone, LoRA setup, train400 split, resolution, optimizer, and
  training budget unchanged for comparison.

This directly targets the confirmed failure: high-IoU pre-NMS boxes are being
discarded because class score alone is a poor ranking signal.

## Artifacts

- Analysis script:
  `scripts/analyze_dinov3_yolo_head_failures.py`
- Summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/summary.json`
- Per-GT CSV:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/gt_best_candidate_analysis.csv`
- Pre-NMS subset check:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/pre_nms_subset30.json`
- Qualitative examples:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/examples/`
