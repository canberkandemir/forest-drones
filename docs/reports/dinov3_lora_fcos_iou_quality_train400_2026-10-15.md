# DINOv3 LoRA FCOS IoU-Quality Head Train400 Pilot

Date: 2026-10-15

## Question

Can a 161-parameter IoU-quality predictor correct the measured mismatch between
FCOS confidence and localization quality, recover valid candidates lost by
top-300/NMS, and improve recall at no more than 0.425 false positives per image?

This is a controlled train400 continuation and a pilot-subset result. It is not
an official ForestPersons benchmark and thresholds selected on val600 are
reported only as exploratory diagnostics.

## Implementation

`DINOv3FCOSQualityHead` preserves every module and tensor in the canonical
stride-16 FCOS head and adds one `Conv2d(160, 1, kernel_size=1)` predictor (161
parameters). The branch is zero initialized, corresponding to quality 0.5.
All 18 trained FCOS tensors and the rank-4 q/v LoRA state were loaded from the
original train5000 checkpoint.

For positive FCOS locations only, the target is the IoU between the currently
decoded predicted L/T/R/B box and its assigned GT box. Targets are detached.
The loss is positive-only BCE summed and divided by the positive count. The
experimental objective adds this loss with weight 1.0. Background cells do not
supervise quality.

Inference uses:

`score = class_probability^0.5 * centerness_probability^0.5 * quality_probability^q`

The legacy score is exactly preserved at `q=0`. The same score is used for the
0.001 candidate floor, top-300, NMS ordering, final ranking, and thresholding.
Top-k, NMS IoU 0.50, stride 16, input resolution 672, and all original losses
remain unchanged.

## Controlled training

Both arms started from the same original LoRA+FCOS train5000 checkpoint and
used corrected image-ID lookup, the same train400 images, seed 20261015,
identical per-epoch order hashes, AdamW, LoRA LR 1e-4, head LR 3e-4, weight
decay 1e-4, batch 1, accumulation 4, positive-count focal normalization,
Smooth L1 regression, and global gradient clipping at 5.0.

| Arm | Trainable params | Steps | Epoch losses | Runtime | Peak MPS driver memory |
| --- | ---: | ---: | --- | ---: | ---: |
| Control | 598,374 | 200 | 0.8676, 0.7884 | 1,188.7 s | 10.53 GiB |
| IoU quality | 598,535 | 200 | 1.4373, 1.2529 | 1,142.8 s | 10.62 GiB |

The quality BCE decreased from 0.5691 to 0.4622. Mean quality targets were
0.8338 and 0.8368, with epoch-2 range 0.2270-0.9873. Quality gradients were
finite and nonzero; epoch-2 quality-branch gradient L2 was 0.1904. Both arms
used 4,128 positive cells per epoch and remained numerically stable.

## Calibration

A deterministic 100-image holdout was sampled from train5000 excluding all
train400 image IDs. It has 122 annotations (v20 36, v40 30, v70 20, v100 36),
zero missing files, and zero train400 overlap. Selection seed is 20261015 and
the sample-key SHA-256 is
`3eed49c24e113482d42c01ad90132aa4e13fa6ccae2bd9d9d6b45ff281521f93`.
No calibration image was used for gradient updates.

| Quality power | Threshold at FP/image <=0.425 | Recall | R20 | AP |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.515 | 0.9590 | 0.9167 | 0.6588 |
| 0.25 | 0.480 | 0.9508 | 0.8889 | 0.6519 |
| 0.5 | 0.450 | 0.9426 | 0.8611 | 0.6472 |
| 1.0 | 0.400 | 0.9344 | 0.8333 | 0.6310 |

The original score (`q=0`) won calibration. To test the requested quality-aware
path without using val600 for selection, `q=0.25` was locked as the best
nonzero formula. Its locked final threshold is 0.480. The control threshold was
independently calibrated as 0.515.

## Val600 results

COCO metrics use pycocotools over scores down to 0.001. The operating rows
below are exploratory val600 sweeps constrained to FP/image <=0.425; these
thresholds are not independently validated.

| Model/scoring | AP | AP50 | AP75 | AR@100 | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original train5000 reference | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.550 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Matched control | 0.3510 | 0.8110 | 0.2260 | 0.4692 | 0.555 | 0.8478 | 0.7200 | 0.4117 | 0.6897 | 0.8118 | 0.8964 | 0.9623 |
| Quality head, legacy score (`q=0`) | 0.3526 | 0.8083 | 0.2312 | 0.4677 | 0.555 | 0.8478 | 0.7200 | 0.4117 | 0.6839 | 0.8176 | 0.8964 | 0.9623 |
| Quality head, quality-aware (`q=0.25`) | 0.3485 | 0.7998 | 0.2234 | 0.4682 | 0.520 | 0.8451 | 0.7144 | 0.4217 | 0.6782 | 0.8059 | 0.9016 | 0.9623 |

At calibration-locked thresholds, control recall is 0.8652 at 0.5733 FP/image;
quality-aware recall is 0.8638 at 0.5683 FP/image. The calibration FP budget
does not transfer to val600, demonstrating score-distribution shift. These
locked rows therefore do not satisfy the operational budget on val600.

Maximum IoU-0.50 recall at the 0.001 candidate floor is 0.9386 for control and
0.9292 for both quality-head scoring modes, at roughly 76-79 FP/image. This is
not COCO AR and is not an operational result.

## Ranking diagnostics

Predicted quality has a moderate association with actual candidate overlap:

- Overlapping raw candidates: Pearson 0.496, Spearman 0.499 (98,923 rows).
- Canonical top-300 candidates, including zero-overlap boxes: Pearson 0.425,
  Spearman 0.300 (180,000 rows).

Of the previously identified 40 localization cases, the newly trained model
still has an IoU>=0.50 raw candidate for 35. Both legacy and quality-aware
ranking retain 33 in top-300, 15 after NMS, and 6 at their exploratory
matched-budget thresholds. Quality-aware scoring therefore recovers none of
these cases beyond the experimental model's legacy score.

Across all GTs at the exploratory matched-budget points, quality-aware scoring
recovers one GT detected neither by the control operating point nor its match,
but loses three control GTs. The net change is -2 detections. Better rank
correlation did not translate into better operational recall.

## Decision

1. **Localization-aware ranking:** the branch learns a real but only moderate
   IoU signal; the selected nonzero score worsens AP and AP75.
2. **Top-k/NMS recovery:** no measured gain for the known 40 cases.
3. **Confidence calibration:** no gain; calibration chose `q=0`, and locked
   thresholds exceeded the FP budget on val600.
4. **Recall/R20:** quality-aware scoring lowers both at matched FP/image.
5. **Scale-up:** do not run train5000 for this formulation. The 161-parameter
   branch is technically sound, but train400 evidence rejects the hypothesis
   that multiplicative IoU quality fixes the operational bottleneck.

The more defensible next direction is not another rank-only head. The oracle
gap requires jointly improving proposal localization/selection and calibrated
confidence, or adopting an assignment/loss formulation that learns quality in
the classification target itself. That should be proposed as a separate,
controlled experiment rather than inferred from this negative result.

## Artifacts

- Training checkpoints/logs: `outputs/dinov3_lora_fcos_quality/train400_*`
- Calibration and val600 summaries: `outputs/dinov3_lora_fcos_quality/calibration100_*`
  and `outputs/dinov3_lora_fcos_quality/val600_*`
- Comparison CSV/plots: `outputs/dinov3_lora_fcos_quality/comparison/`
- Candidate diagnostics: `outputs/dinov3_lora_fcos_quality/analysis/`
- Calibration manifest:
  `outputs/subsets/forestpersons_train_quality_calibration_100_seed20261015_manifest.json`
