# DINOv3 ViT-S LoRA + YOLO-Style Head Train400 Pilot

Date: 2026-10-13

## Scope

This was the first controlled pilot for the DINOv3 ViT-S + LoRA + YOLO-style
one-stage detection head on ForestPersons.

No train5000 training, dataset download, FCOS modification, Faster R-CNN, or
Deformable DETR work was performed.

## Configuration

- Dataset: existing ForestPersons train400 subset.
- Evaluation: existing val600 subset.
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Initialization:
  - DINOv3 LoRA weights loaded from the best existing ViT-S LoRA + FCOS train5000 checkpoint.
  - YOLO-style detection head freshly initialized.
  - FCOS detection-head weights were intentionally not reused.
- Resolution: short-side 672.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Optimizer: AdamW.
- Learning rates: LoRA `1e-4`, head `3e-4`.
- Seed: 20261013.
- Device: Apple MPS.
- Head: single stride-16 DINOv3 patch grid, DFL bins `reg_max=32`,
  task-aligned assignment with `topk=10`, `alpha=1.0`, `beta=6.0`.

## Training Diagnostics

| Epoch | Loss | Classification | IoU | DFL | Positives/image | Optimizer steps | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.8272 | 36.6765 | 0.6439 | 2.2143 | 11.70 | 100 | 499.5 |
| 2 | 9.0459 | 3.3684 | 0.4191 | 1.6896 | 11.71 | 200 | 447.4 |

Assignment and stability:

- Non-finite losses: none.
- Center fallbacks: 0.
- Clipped target distances: 0.
- Multi-GT overwrites: 32 in epoch 1, 28 in epoch 2.
- Trainable tensors with nonzero gradients in epoch 2:
  - LoRA: 48/48.
  - YOLO head: 18/18.
- Parameter update check:
  - First LoRA trainable tensor max absolute change: 0.0069.
  - First head trainable tensor max absolute change: 0.0395.
- Total runtime: 947.1 s, about 15.8 min.
- Peak MPS driver memory: 11,645,124,608 bytes, about 10.85 GiB.

## Val600 Results

Evaluation used the canonical ForestPersons framework, IoU 0.50 for custom
recall/precision matching, NMS IoU 0.50, and pycocotools COCOeval where
available.

COCO metrics:

| Metric | Value |
| --- | ---: |
| AP@[0.50:0.95] | 0.1339 |
| AP50 | 0.5094 |
| AP75 | 0.0171 |
| AR@100 | 0.3198 |

Fixed operating point, confidence 0.60:

| Metric | Value |
| --- | ---: |
| Recall | 0.3992 |
| Precision | 0.6308 |
| FP/image | 0.2917 |
| Recall @20% visibility | 0.3161 |
| Recall @40% visibility | 0.3706 |
| Recall @70% visibility | 0.4974 |
| Recall @100% visibility | 0.4009 |

Exploratory matched-FP result:

The sweep threshold was selected on val600, so this is exploratory and should
not be treated as an unbiased tuned operating point.

| FP/image budget | Selected threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.425 | 0.60 | 0.3992 | 0.6308 | 0.2917 | 0.3161 | 0.3706 | 0.4974 | 0.4009 |

Maximum custom recall over the evaluated confidence thresholds:

- Threshold: 0.001.
- Recall: 0.8905.
- Precision: 0.0337.
- FP/image: 31.8717.

This shows the model can produce overlapping candidates for many people, but
the confidence ranking is currently poor and produces excessive false positives
at low thresholds.

## Comparison With Existing Results

All rows are val600 pilot results. The protocols are not fully matched:

- FCOS train5000 used an FCOS head initialized from a trained frozen-DINOv3
  FCOS checkpoint and trained LoRA + FCOS on train5000.
- FCOS train400 used the same FCOS formulation and trained on train400.
- This YOLO-style pilot reused only the trained LoRA backbone weights from the
  FCOS train5000 checkpoint, then trained a fresh YOLO-style head on train400.

| Model | AP | AP50 | AP75 | AR@100 | Recall at FP<=0.425 | Precision | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned train5000 | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.9039 | 0.7650 | 0.3467 | 0.7701 |
| DINOv3 LoRA + FCOS train5000 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8465 | 0.7172 | 0.4167 | 0.6667 |
| DINOv3 LoRA + FCOS train400 | 0.3334 | 0.7893 | 0.2211 | 0.4622 | 0.8011 | 0.7643 | 0.3083 | 0.5747 |
| DINOv3 LoRA + YOLO-style train400 | 0.1339 | 0.5094 | 0.0171 | 0.3198 | 0.3992 | 0.6308 | 0.2917 | 0.3161 |

## Interpretation

The YOLO-style implementation is technically functional: assignment is
producing positives, losses are finite, gradients flow through both LoRA and
the new head, and val600 evaluation works.

Scientifically, this first pilot does not justify scaling the current
YOLO-style head directly to train5000. AP75 is extremely low, and recall at the
accepted FP/image budget is far behind the existing FCOS detector.

The most likely issues are:

- confidence calibration/ranking is poor after only 2 epochs from a fresh head,
- the single stride-16 feature map is too coarse for localization,
- the BCE classification term is very large early in training and may need
  YOLO-style normalization or a focal/classification weighting adjustment,
- the head lacks FCOS's centerness signal or an equivalent localization-aware
  scoring term.

## Recommendation

Do not scale this exact configuration to train5000 yet.

Recommended next correction before scaling:

1. Run a small diagnostic on saved train400/val600 predictions to inspect
   whether missed detections are mainly localization errors or confidence
   ranking errors.
2. Adjust the YOLO-style loss/scoring in one controlled way, most likely by
   adding a localization-quality score or changing classification normalization,
   then rerun train400.

Only scale to train5000 if a corrected train400 pilot approaches the FCOS
train400 result at FP/image <= 0.425 or clearly improves AP75.

## Artifacts

- Training script:
  `scripts/train_dinov3_lora_yolo_head.py`
- Checkpoint:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt`
- Training summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/summary.json`
- Evaluation summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/summary.json`
- COCO predictions:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/coco_results.json`
- Confidence sweep:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/confidence_sweep.csv`
