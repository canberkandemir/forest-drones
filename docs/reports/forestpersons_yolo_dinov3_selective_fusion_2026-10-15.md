# ForestPersons YOLO11s + DINOv3 FCOS Selective Fusion

Date: 2026-10-15

## Scope

This experiment tested prediction-level fusion of the existing ForestPersons
train5000 detectors without changing or retraining either model:

- fine-tuned YOLO11s at image size 640;
- DINOv3 ViT-S/16 + rank-4 q/v LoRA + stride-16 FCOS at short side 672.

The detector outputs on val600 were reused. Only a new, disjoint calibration200
subset required local inference. No data or model weights were downloaded.

## Audit

- YOLO val600 artifact: 2,622 post-NMS boxes, scores 0.0010-0.9237; 1,737
  candidates are below 0.05. The low-confidence YOLO rescue study therefore did
  not require new val600 inference.
- DINOv3 val600 artifact: 46,586 post-NMS boxes, scores 0.0957-0.9258. Its actual
  saved floor censors lower-scoring DINO candidates.
- The evaluator exactly reproduced the canonical fixed points:
  YOLO at 0.05 has recall 0.9039, precision 0.7650 and 0.3467 FP/image; DINOv3
  at 0.55 has recall 0.8465, precision 0.7172 and 0.4167 FP/image.
- Earlier raw union/NMS and WBF results were reviewed. Their best recall under
  0.425 FP/image was 0.8838 and 0.8652 respectively, below YOLO alone. Those
  score-scale-mixing methods were not repeated as candidate solutions.

## Calibration Protocol

Calibration200 was deterministically selected from the official validation
split after excluding all val600 filenames. It contains 200 locally available
images, 50 targeted per visibility level, and 252 GT people (61/58/65/68 at
20/40/70/100 visibility). It has zero image overlap with val600 and belongs to
the official validation split, which is separate from detector train5000.

Both detector checkpoints were run once on calibration200 with their canonical
preprocessing and score floor 0.001. Fusion gates and locked score thresholds
were selected only on calibration200 under 0.425 FP/image. The 75 method-A and
180 method-B configurations were fixed before val600 evaluation.

The tested methods were:

- **A, selective addition:** retain YOLO >= 0.075; add DINO >= 0.55 only when
  IoU with retained YOLO boxes is below 0.30; one-class NMS remains 0.50.
- **B, supported YOLO rescue:** retain YOLO >= 0.10; rescue YOLO scores >= 0.01
  only when a DINO box >= 0.50 agrees at IoU >= 0.30 and box-area consistency
  is at least 0.25.
- **C, calibrated support ranking:** map each source score through a ten-bin
  monotonic empirical precision calibrator. For each candidate, compute
  `1 - (1 - p_base) * (1 - IoU * area_consistency * p_support)`, then rank and
  NMS using that score. Ground truth is never used by these inference rules.

Calibration results were:

| Method | Locked threshold | Recall | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: |
| YOLO | 0.040 | 0.9206 | 0.385 | 0.8197 |
| DINOv3 | 0.535 | 0.8730 | 0.425 | 0.7377 |
| Selective A | gates only | 0.9365 | 0.410 | 0.8689 |
| Selective B | gates only | 0.9365 | 0.380 | 0.8361 |
| Calibrated C | 0.410 | 0.9008 | 0.405 | 0.7869 |

## Val600 Results

### Calibration-Locked Operating Points

These parameters were transferred unchanged. This is the least biased result
available in the experiment.

| Method | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO | 0.9092 | 0.7418 | 0.3950 | 0.7701 | 0.8941 | 0.9793 | 0.9717 |
| DINOv3 | 0.8518 | 0.6868 | 0.4850 | 0.6782 | 0.8118 | 0.9223 | 0.9623 |
| Selective A | 0.9226 | 0.7131 | 0.4633 | 0.8161 | 0.9000 | 0.9845 | 0.9717 |
| Selective B | 0.9212 | 0.7240 | 0.4383 | 0.7989 | 0.9059 | 0.9793 | 0.9811 |
| Calibrated C | 0.8945 | 0.7030 | 0.4717 | 0.7471 | 0.8647 | 0.9689 | 0.9717 |

Only calibration-locked YOLO remained below 0.425 FP/image on val600. The
fusion methods improved recall but failed the locked false-positive constraint,
so they do not establish an operational improvement.

### Exploratory Val600-Matched Budget

The following thresholds were selected on val600 and are therefore exploratory,
not independent confirmation.

| Method | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 | AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO | 0.035 | 0.9119 | 0.7352 | 0.4100 | 0.7759 | 0.9000 | 0.9793 | 0.9717 | 0.6339 |
| DINOv3 | 0.550 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 | 0.4704 |
| Selective A | 0.110 | 0.9186 | 0.7304 | 0.4233 | 0.8046 | 0.9000 | 0.9793 | 0.9717 | 0.6160 |
| Selective B | 0.045 | **0.9212** | 0.7302 | 0.4250 | 0.7989 | 0.9059 | 0.9793 | 0.9811 | 0.6170 |
| Calibrated C | 0.430 | 0.8945 | 0.7314 | 0.4100 | 0.7471 | 0.8647 | 0.9689 | 0.9717 | **0.6409** |

At the exploratory matched budget, method B gained ten GT and lost three
relative to matched-budget YOLO: net +7/749 people, with nine additional false
positives. It recovered 11 of the 28 reference DINO-only people. Method A gained
14 and lost nine: net +5 people with eight additional false positives, while
recovering 18/28 DINO-only people. Method A produced the larger R20 increase
(+0.0287), while method B produced the larger overall net gain (+0.0093 recall).

Threshold-free ranking quality did not improve. YOLO AP/AP75/AR@100 were
0.5541/0.5687/0.6339. Selective A scored 0.5295/0.5431/0.6160 and selective B
0.5298/0.5434/0.6170. Method C increased AR@100 to 0.6409 but reduced
operational recall and did not recover any of the 28 reference DINO-only GTs.

## Oracle Bound

The non-deployable GT-assisted union of YOLO matches at 0.05 and DINO matches
at 0.55 covers 705/749 GT: recall 0.9413, with R20/R40/R70/R100 of
0.8448/0.9294/0.9845/0.9906. It is an opportunity bound, not an inference
method and not evidence that arbitrary prediction union will attain it.

## Decision

Selective fusion has a measurable *exploratory* opportunity: supported
low-confidence YOLO rescue added seven net GT at the val600 matched budget, and
selective DINO addition improved R20. It has not yet demonstrated a reliable
operational benefit because the calibration-locked fusion settings exceeded
the target FP rate after transfer, and AP/AR did not improve for methods A/B.

Do not start a more complex learned fusion model from this result. The smallest
valid confirmation is to freeze method A/B parameters and evaluate them once on
a new held-out sample from the remaining official validation images, suggested
size 1,200 (300 target images per visibility level), excluding calibration200
and val600. Report both the locked operating point and matched-budget curves.

## Artifacts

- `src/forestpersons/selective_fusion.py`
- `scripts/create_fusion_calibration_subset.py`
- `scripts/evaluate_selective_fusion.py`
- `tests/test_selective_fusion.py`
- `outputs/subsets/forestpersons_val_fusion_calibration_200_seed20261015_manifest.json`
- `outputs/fusion/yolo_dino_selective_20261015/calibration200_yolo/`
- `outputs/fusion/yolo_dino_selective_20261015/calibration200_dino/`
- `outputs/fusion/yolo_dino_selective_20261015/selective_eval_verified/`

