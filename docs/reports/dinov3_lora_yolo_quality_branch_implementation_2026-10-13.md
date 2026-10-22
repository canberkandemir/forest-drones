# DINOv3 LoRA YOLO-Style Quality Branch Implementation

Date: 2026-10-13

## Scope

This task verified the NMS ranking failure in the DINOv3 ViT-S + LoRA +
YOLO-style DFL head, then implemented a lightweight localization-quality branch.

No train400 or train5000 training was run. No checkpoint was modified.

## Verified NMS Failure

The previous diagnostic showed that well-localized pre-NMS candidates were lost
after NMS. This was rechecked on the same first 30 val600 images using the
actual decoder sequence:

1. confidence threshold `0.001`,
2. class-score top-k `300`,
3. NMS IoU `0.50`.

Result:

| Quantity | Count |
| --- | ---: |
| Images | 30 |
| GT persons | 33 |
| GT with pre-NMS best IoU >= 0.75 | 12 |
| Lost at confidence threshold | 0 |
| Lost at top-k | 0 |
| Lost at NMS | 12 |
| Kept after NMS | 0 |

Examples confirmed the mechanism: the suppressed candidate had higher IoU with
the GT person, but an overlapping lower-IoU candidate had a higher class score.

Example:

| Field | Value |
| --- | ---: |
| Candidate GT IoU | 0.7836 |
| Candidate class score | 0.0687 |
| Suppressor GT IoU | 0.5869 |
| Suppressor class score | 0.3171 |
| Candidate/suppressor IoU | 0.7561 |

This confirms the failure is not threshold filtering or top-k pruning. It is
score-ranked NMS with a class-only score.

## Implementation

Updated module:

- `src/forestpersons/dinov3_yolo.py`

Changes:

- Added optional `include_quality` flag to `DINOv3YOLOHead`.
- Added a small quality branch parallel to classification/regression.
- The quality branch predicts one localization-quality logit per patch cell.
- Existing class-only head behavior is unchanged by default.
- Existing class-only checkpoints remain loadable because `include_quality`
  defaults to `False` when checkpoint metadata is absent.

Training target:

- For positive cells, quality target is the detached IoU between the decoded
  predicted L/T/R/B box and the assigned GT target box.
- For unmatched/background cells, quality target is `0`.
- Quality loss is BCE-with-logits averaged over the grid.
- Ground-truth IoU is used only as a training target, not during inference.

Inference:

- Default score mode remains `class`.
- New optional mode: `class_quality`, using:

```text
score = class_score * predicted_quality
```

This score is applied before top-k selection and NMS.

Updated scripts:

- `scripts/smoke_dinov3_lora_yolo_head.py`
- `scripts/train_dinov3_lora_yolo_head.py`
- `scripts/evaluate_forestpersons_framework.py`

New/updated flags:

- `--include-quality`
- `--quality-loss-weight`
- `--score-mode class_quality` for smoke.
- `--yolo-score-mode class_quality` for evaluation.

## Smoke Test

Command:

```bash
PYTHONPATH=src .venv-dinov3/bin/python scripts/smoke_dinov3_lora_yolo_head.py \
  --output-dir outputs/dinov3_lora_yolo_head/smoke_quality_train2_seed20261013_v2 \
  --max-images 2 \
  --short-side 672 \
  --device mps \
  --include-quality \
  --quality-loss-weight 1.0 \
  --score-mode class_quality
```

Result:

- Device: MPS.
- Images: 2 real train400 images.
- Non-finite losses: none.
- Nonzero gradients:
  - LoRA tensors: 48/48.
  - Head tensors: 26/26.
- Runtime: 2.75 s.
- Peak MPS driver memory: 5,937,463,296 bytes, about 5.53 GiB.

Parameter count:

| Component | Trainable | Total |
| --- | ---: | ---: |
| DINOv3 backbone with LoRA | 73,728 | 21,670,272 |
| YOLO-style head + quality branch | 776,774 | 776,774 |
| Total | 850,502 | 22,447,046 |

Smoke losses:

| Image | Total | Classification | IoU | DFL | Quality | Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `088_FPV_Person_summer_forest-0006.jpg` | 1556.2864 | 1543.1132 | 0.9949 | 3.4400 | 0.5512 | 4 |
| `336_FPV_Person_summer_forest-0043.jpg` | 479.9232 | 467.3584 | 0.9955 | 3.2771 | 0.1825 | 8 |

The first attempted quality loss normalization was too large because it summed
all background cells and normalized by positives. It was corrected before this
validated smoke run by averaging BCE over the grid.

## Tests

Added/updated tests in `tests/test_dinov3_yolo_head.py` for:

- quality output shape,
- quality loss finite gradients,
- class-only state-dict compatibility,
- quality-aware score ordering,
- quality-aware NMS survivor changes,
- existing DFL, IoU, assignment, no-positive, and coordinate-scaling behavior.

Targeted tests:

```text
PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests/test_dinov3_yolo_head.py tests/test_detection_evaluation.py -q
21 passed
```

## Readiness

The quality-aware head is ready for a controlled train400 pilot.

Recommended pilot:

- Use the same train400/val600 splits.
- Use the same LoRA initialization and YOLO head initialization protocol as the
  previous YOLO-style train400 pilot.
- Enable `--include-quality`.
- Evaluate twice if useful:
  - class-only score for backward comparison,
  - class-quality score for the intended correction.

Success criterion:

- Improve AP75 and recall at FP/image <= 0.425 over the previous YOLO-style
  head.
- Ideally approach the FCOS train400 AP75 of 0.2211 before considering
  train5000 scaling.

## Artifacts

- NMS verification:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/nms_ranking_verification_subset30.json`
- Smoke summary:
  `outputs/dinov3_lora_yolo_head/smoke_quality_train2_seed20261013_v2/summary.json`
- Smoke checkpoint:
  `outputs/dinov3_lora_yolo_head/smoke_quality_train2_seed20261013_v2/smoke_yolo_head.pt`
