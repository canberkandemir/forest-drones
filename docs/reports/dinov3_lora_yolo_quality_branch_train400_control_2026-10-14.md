# DINOv3 LoRA YOLO-Style Quality Branch Train400 Control

Date: 2026-10-14

## Scope

This experiment tests whether an optional localization-quality branch improves
the DINOv3 ViT-S + LoRA + YOLO-style DFL/IoU detector. It uses the existing
ForestPersons train400 subset and evaluates on the existing selected val600
subset. No dataset downloads, train5000 training, or architecture changes
outside the quality branch were performed.

## Initialization

Both continuations started from the same original YOLO-style train400
checkpoint:

`outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt`

Run A continued the class-only YOLO head with no quality supervision. Run B
loaded the same LoRA backbone and shared YOLO stem/classification/regression
weights, then freshly initialized only the quality branch.

Training settings were matched:

- Dataset: train400 seed20261011.
- Evaluation: val600 seed20261010.
- Resolution: short side 672.
- Epochs: 2 additional epochs.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Device: Apple MPS.
- Optimizer/loss settings: same YOLO-style DFL/IoU trainer defaults.

## Quality Loss Balance Check

The original grid-mean BCE quality loss would be dominated by background cells.
On a deterministic 50-image train400 diagnostic:

| Quantity | Value |
| --- | ---: |
| Positive cells/image | 9.28 |
| Negative cells/image | 3164.24 |
| Mean positive BCE | 0.7237 |
| Mean negative BCE | 0.9351 |
| Raw full-grid negative contribution | 99.77% |
| Mean positive quality target | 0.5024 |
| Max positive quality target | 0.9203 |

Because the negative contribution dominated the raw grid mean, the quality
branch was trained with explicit positive/negative normalization:

`quality_loss = positive_mean_bce + quality_negative_weight * negative_mean_bce`

with `quality_negative_weight = 1.0`.

## Training Stability

| Run | Trainable params | Runtime | Optimizer steps | Peak MPS driver memory | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| A: class-only continuation | 619,141 | 1185.4 s | 200 | 11,645,206,528 bytes | Finite losses; LoRA/head gradients nonzero |
| B: quality branch | 850,502 | 912.2 s | 200 | 11,657,707,520 bytes | Finite losses; LoRA/head/quality gradients nonzero |

Run B epoch quality loss decreased from 0.7503 to 0.6880. Positive quality loss
remained visible after balancing, while negative loss fell from 0.0562 to
0.0215.

## Val600 Results

All results use the canonical ForestPersons evaluation framework. COCO metrics
come from pycocotools on the selected val600 pilot subset. The matched
FP/image point is selected from the predefined confidence sweep and should be
treated as exploratory because the threshold is chosen on val600.

| Model | Scoring | AP | AP50 | AP75 | AR@100 | Recall @ FP<=0.425 | Precision | FP/img | R20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original YOLO train400 e2 | class | 0.1339 | 0.5094 | 0.0171 | 0.3198 | 0.3992 | 0.6308 | 0.2917 | 0.3161 |
| Run A continue e2 | class | 0.1643 | 0.6128 | 0.0234 | 0.3446 | 0.5901 | 0.7129 | 0.2967 | 0.3851 |
| Run B quality e2 | class | 0.1835 | 0.6597 | 0.0260 | 0.3366 | 0.6088 | 0.7427 | 0.2633 | 0.4368 |
| Run B quality e2 | class * quality | 0.1870 | 0.6644 | 0.0287 | 0.3332 | 0.6542 | 0.7020 | 0.3467 | 0.4598 |

Visibility recall at the best FP/image <= 0.425 point for Run B with
quality-aware scoring:

| Visibility | Recall |
| --- | ---: |
| 20% | 0.4598 |
| 40% | 0.6235 |
| 70% | 0.7098 |
| 100% | 0.7877 |

## NMS Diagnostic

The same first 30 val600 images were rechecked at confidence 0.001, top-k 300,
and NMS IoU 0.50.

| Scoring | GT | Pre-NMS IoU>=0.75 | Lost threshold | Lost top-k | Lost NMS | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original class-only checkpoint | 33 | 12 | 0 | 0 | 12 | 0 |
| Run B class | 33 | 15 | 4 | 0 | 10 | 1 |
| Run B class * quality | 33 | 15 | 4 | 0 | 10 | 1 |

The quality branch improves global ranking slightly, but it does not solve the
main failure: most high-IoU candidates are still either below the low diagnostic
threshold or suppressed during NMS.

Predicted quality has a positive but imperfect relationship with actual
localization IoU:

- Top-k candidates, class-aware diagnostic: Pearson 0.4500.
- NMS survivors, class-aware diagnostic: Pearson 0.6168.
- Top-k candidates, quality-aware diagnostic: Pearson 0.2979.
- NMS survivors, quality-aware diagnostic: Pearson 0.5303.

## Conclusion

The quality branch gives a real but modest improvement over the original
YOLO-style head and the class-only continuation. It improves AP, AP50, AP75,
and recall at the FP/image <= 0.425 budget, including visibility-20 recall.

However, the absolute AP75 remains extremely weak at 0.0287, far below the
single-scale FCOS train400 pilot AP75 of 0.2211 and the best ViT-S LoRA FCOS
train5000 AP75 of 0.2379. This means the YOLO-style detector still has a
localization/ranking problem and should not be scaled to train5000 yet.

Recommended next action: do not scale this YOLO-style head as-is. Return to the
stronger FCOS line or implement a more structural localization fix, such as
quality-focal/varifocal classification targets or a duplicate-aware ranking
loss, before any larger YOLO-style training run.

## Artifacts

- Comparison CSV:
  `outputs/dinov3_lora_yolo_head/quality_branch_control_train400_continuation/val600_comparison.csv`
- Quality balance diagnostic:
  `outputs/dinov3_lora_yolo_head/quality_branch_control_train400_continuation/quality_balance_train50.json`
- Run A:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_continue_e2_classonly_from_yolo400/`
- Run B:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_continue_e2_quality_from_yolo400/`
- NMS diagnostics:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_continue_e2_quality_from_yolo400/error_analysis/`
