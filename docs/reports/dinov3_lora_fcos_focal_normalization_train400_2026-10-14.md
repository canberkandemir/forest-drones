# DINOv3 LoRA FCOS Focal-Normalization Train400 Control

Date: 2026-10-14

## Question

Does replacing the existing mean focal classification loss with the FCOS-style
sum divided by the number of positive locations improve ForestPersons recall,
especially at 20% visibility, without increasing false positives or damaging
COCO AP/AR?

This is a controlled two-epoch continuation on train400. Both arms start from
the same DINOv3 ViT-S/16 + rank-4 q/v LoRA + stride-16 FCOS train5000
checkpoint. Thresholds selected on val600 are exploratory and are not an
independent validation result.

## Data correction

The trainer previously resolved manifest entries by `file_name`. ForestPersons
contains duplicated filenames attached to different COCO image IDs, so this
could attach the wrong annotation record. The corrected loader resolves by
`image_id` and then verifies the filename.

| Subset | Images | Correct annotations | v20 | v40 | v70 | v100 | Filename-resolved annotations | Mismatched manifest rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train400 | 400 | 524 | 114 | 119 | 131 | 160 | 525 | 3 |
| train5000 | 5,000 | 6,415 | 1,423 | 1,440 | 1,595 | 1,957 | 6,431 | 18 |

The correction was applied identically to both train400 arms. Val600 has no
duplicate-filename mismatch and its 749 annotations are unchanged. Historical
checkpoints are preserved and therefore retain their original training-data
lookup behavior.

## Loss change

Only focal-loss normalization differs:

- **Mean arm:** mean focal loss across all grid locations, preserving the old
  behavior.
- **Positive-count arm:** summed focal loss divided by
  `max(number_of_positive_locations, 1)` per image. This is applied once before
  gradient accumulation.

Both use focal alpha 0.75 and gamma 2.0. Smooth L1 regression, regression weight
2.0, positive-only centerness BCE, assignment, model architecture, inference,
AdamW parameter groups, seed, input order, short side 672, batch size 1,
accumulation 4, and global gradient clipping at 5.0 are unchanged.

## Safety checks

On the exact same initialized one-image batch, regression and centerness losses
were identical. Mean normalization produced classification loss 0.000166 and
classification-gradient L2 norm 0.000143; positive-count normalization produced
0.130370 and 0.112272. The 787.5x loss ratio exactly equals grid locations per
positive for that image. All losses and gradients were finite.

A 10-image MPS smoke test completed for both modes. Neither run clipped a
gradient, all LoRA and head groups received finite gradients, and the optimizer
performed three steps in each arm.

## Training

Both full runs saw the same corrected 400 images in the same order, recorded
the same per-epoch image/sample SHA-256 digests, produced 4,128 positive cells
per epoch, and completed 200 optimizer steps.

| Arm | Epoch | Total loss | Classification | Smooth L1 | Centerness | Clipped steps | Peak pre-clip norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean | 1 | 0.6245 | 0.00072 | 0.01563 | 0.5925 | 0/100 | 1.80 |
| Mean | 2 | 0.6134 | 0.00068 | 0.01119 | 0.5903 | 0/100 | 0.94 |
| Positive count | 1 | 0.8642 | 0.23899 | 0.01627 | 0.5927 | 3/100 | 6.75 |
| Positive count | 2 | 0.7902 | 0.17100 | 0.01389 | 0.5915 | 4/100 | 7.55 |

At epoch 2, classification represented 21.64% of the positive-count objective,
versus 0.11% of the mean objective. The positive-count classification gradient
L2 norm was 0.253 versus 0.0021 for the mean arm. Seven of 200 positive-count
optimizer steps were clipped, but training remained finite and stable; this was
not persistent destructive clipping.

| Arm | Runtime | Peak observed MPS driver memory |
| --- | ---: | ---: |
| Mean | 936.4 s (15.6 min) | 10.62 GiB |
| Positive count | 1,071.3 s (17.9 min) | 10.53 GiB |

## Val600 results

COCO metrics use pycocotools with IoU 0.50:0.95 and maxDets 100. Custom recall
uses one-to-one matching at IoU 0.50. The matched operating point is the highest
recall among the evaluated thresholds with FP/image no greater than 0.425.

| Model | AP | AP50 | AP75 | AR@100 | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original train5000 reference | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.550 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Corrected train400 + mean | 0.3465 | 0.8075 | 0.2247 | 0.4685 | 0.570 | 0.8411 | 0.7135 | 0.4217 | 0.6782 | 0.7824 | 0.9067 | 0.9623 |
| Corrected train400 + positive count | 0.3520 | 0.8157 | 0.2327 | 0.4697 | 0.560 | 0.8465 | 0.7172 | 0.4167 | 0.6839 | 0.7941 | 0.9119 | 0.9623 |

Relative to the corrected mean arm, positive-count normalization changes:

- AP +0.0055, AP50 +0.0083, AP75 +0.0080, and AR@100 +0.0012.
- Recall +0.0053 (four net GT) at the matched FP budget.
- R20 +0.0057 (one net visibility-20 GT).
- FP/image decreases from 0.4217 to 0.4167 at the selected matched-budget point.

At these operating points, the positive-count model detects 13 GT not detected
by the mean model and loses 9 mean-model detections. The 13 gains comprise four
v20, three v40, four v70, and two v100 instances; the nine losses comprise
three v20, one v40, three v70, and two v100 instances.

At the same fixed threshold 0.55, the mean arm obtains recall 0.8505,
precision 0.6842, and 0.4900 FP/image. The positive-count arm obtains recall
0.8518, precision 0.7034, and 0.4483 FP/image. This supports a modest confidence
ranking/calibration benefit rather than a gain obtained simply by admitting
more false positives.

The unrestricted maximum recall over saved predictions down to confidence
0.001 rises from 0.9386 to 0.9426, but requires about 77 FP/image and is not an
operational result or COCO AR.

## Score behavior

Across all pre-NMS grid locations, positive-count training lowers mean
classification probability from 0.0240 to 0.0181 and mean combined score from
0.0796 to 0.0641, while mean centerness changes from 0.4074 to 0.4095. Combined
with the small AP gains and lower fixed-threshold FP count, this is consistent
with stronger background suppression. These global histograms do not separate
true and false candidates, so that interpretation remains a hypothesis rather
than a direct measurement of calibration.

## Conclusions

1. **Overall recall:** yes, positive-count normalization improves matched-budget
   recall over the corrected mean continuation by four GT, but only modestly.
2. **20% visibility:** yes, by one GT versus the corrected mean arm and three GT
   versus the original reference at the selected operating points.
3. **Confidence ranking:** AP50/AP/AP75 and fixed-threshold precision all improve,
   providing evidence of a small ranking benefit. The effect is not large.
4. **False positives:** no increase at the matched budget; at confidence 0.55,
   positive-count normalization produces 25 fewer false positives.
5. **AR/AP:** both improve relative to the corrected mean arm, so the result is
   not confined to one threshold. The positive-count arm still trails the
   original reference in AP (-0.0067), AP75 (-0.0052), and AR@100 (-0.0007).
6. **Train5000 scaling:** cautiously justified as a controlled follow-up because
   every principal COCO metric and R20 moved in the intended direction without
   an FP penalty. Expected gains are small. A scientifically clean scale-up
   should retain the image-ID correction and compare matched mean and
   positive-count arms; scaling only the experimental arm would confound loss
   normalization with corrected annotations and additional optimization.

## Artifacts

- Mean checkpoint and history:
  `outputs/dinov3_lora_focal_norm/train400_mean_seed20261014_from_lora5k_e2/`
- Positive-count checkpoint and history:
  `outputs/dinov3_lora_focal_norm/train400_positive_seed20261014_from_lora5k_e2/`
- Evaluation summaries and COCO predictions: each run's `eval_val600/` folder.
- Comparison CSV, JSON, and plots:
  `outputs/dinov3_lora_focal_norm/train400_mean_vs_positive_seed20261014_comparison/`
- Reproducible comparison script:
  `scripts/compare_fcos_focal_normalization.py`

