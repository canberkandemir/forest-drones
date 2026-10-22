# DINOv3-Only Detections and Localization Failures on ForestPersons Val600

Date: 2026-10-15

## Scope and evidence

This is a saved-artifact analysis. No model inference, training, dataset
download, or checkpoint modification was performed.

Inputs:

- ForestPersons val600: 600 images and 749 visibility-labeled GT persons.
- Fine-tuned YOLO11s COCO predictions: confidence floor 0.0010.
- DINOv3 ViT-S/16 + LoRA + FCOS COCO predictions: post-NMS/top-300 output,
  observed confidence floor 0.0957.
- Preserved DINOv3 pre-NMS dump: 1,917,636 decoded grid candidates with
  separate class and centerness probabilities; combined-score floor 0.00143.
- Operating points: YOLO confidence 0.05, DINO confidence 0.55, IoU 0.50,
  one-to-one matching.

The pre-NMS FCOS artifact removes most censoring for DINO candidate analysis.
No equivalent YOLO pre-NMS artifact exists, so YOLO failure causes below are
limited to its saved post-NMS candidates.

## DINO-only detections

At the fixed operating points, 606 GTs are detected by both models, 71 by YOLO
only, 28 by DINO only, and 44 by neither model.

Among the 28 DINO-only GTs:

- 22/28 (78.6%) have visibility 20 or 40.
- 21/28 are sitting; 3 are lying; 4 are standing.
- 19/28 are medium-size by the project's maximum-side bins, 5 are small, and
  4 are large.
- Median GT size is 103 x 124 pixels and median area is 11,877 pixels.
- 22 use See3CAM, 4 GoPro, and 2 Sony a57; 22 are clear and 6 rain; all are in
  forest scenes. These camera/weather proportions do not reveal a unique
  DINO-only scene domain.
- YOLO has a geometrically valid IoU >= 0.50 box below confidence 0.05 for
  20/28, a saved localization failure for 5/28, and no useful saved candidate
  for 3/28.

Thus most complementarity is a confidence-ranking difference, not proof that
YOLO generated no person-like candidate. Compared with the 71 YOLO-only GTs,
DINO-only GTs are larger (median area 11,877 versus 8,216 pixels), much more
often medium-size (67.9% versus 38.0%), and less often small (17.9% versus
38.0%). Low visibility is common in both groups (78.6% versus 73.2%). DINO may
therefore complement YOLO on some medium-size, seated, occluded people, but
this is a descriptive val600 observation, not a causal pretraining result.

The complete metadata and scores are in `dino_only_diagnostics.csv`. Compact
values for all 28 GTs are shown below.

| Image | Ann | Vis | GT size | DINO score / IoU | YOLO score / best IoU | YOLO saved-candidate failure |
|---:|---:|---:|---:|---:|---:|---|
| 8244 | 4984 | 20 | 190x73 | 0.691 / 0.563 | 0.422 / 0.389 | localization |
| 340 | 3109 | 20 | 80x131 | 0.742 / 0.760 | 0.001 / 0.773 | confidence |
| 14534 | 37209 | 20 | 51x45 | 0.563 / 0.785 | 0.018 / 0.703 | confidence |
| 17795 | 28086 | 20 | 185x118 | 0.657 / 0.823 | 0.005 / 0.785 | confidence |
| 12143 | 36097 | 20 | 67x168 | 0.733 / 0.834 | 0.113 / 0.000 | no useful saved candidate |
| 8678 | 15743 | 20 | 94x96 | 0.611 / 0.736 | 0.001 / 0.578 | confidence |
| 17907 | 28538 | 20 | 70x60 | 0.641 / 0.551 | 0.568 / 0.027 | no useful saved candidate |
| 2493 | 11767 | 20 | 204x262 | 0.641 / 0.537 | 0.003 / 0.730 | confidence |
| 884 | 4387 | 20 | 101x122 | 0.652 / 0.531 | 0.020 / 0.652 | confidence |
| 11460 | 33441 | 20 | 45x71 | 0.674 / 0.546 | 0.002 / 0.521 | confidence |
| 7968 | 36353 | 20 | 96x124 | 0.792 / 0.760 | 0.014 / 0.838 | confidence |
| 16698 | 14251 | 20 | 188x401 | 0.608 / 0.518 | 0.550 / 0.441 | localization |
| 11376 | 13182 | 20 | 122x124 | 0.683 / 0.727 | 0.012 / 0.642 | confidence |
| 2497 | 11771 | 40 | 75x74 | 0.690 / 0.722 | 0.038 / 0.716 | confidence |
| 11328 | 15962 | 40 | 115x149 | 0.685 / 0.629 | 0.043 / 0.699 | confidence |
| 11337 | 16204 | 40 | 115x155 | 0.715 / 0.747 | 0.044 / 0.804 | confidence |
| 11489 | 33470 | 40 | 61x124 | 0.845 / 0.590 | 0.721 / 0.464 | localization |
| 11354 | 1252 | 40 | 123x179 | 0.656 / 0.582 | 0.002 / 0.599 | confidence |
| 17597 | 27151 | 40 | 39x48 | 0.567 / 0.574 | 0.720 / 0.367 | localization |
| 17477 | 26895 | 40 | 42x105 | 0.698 / 0.732 | 0.002 / 0.692 | confidence |
| 11620 | 34840 | 40 | 52x126 | 0.672 / 0.734 | 0.252 / 0.000 | no useful saved candidate |
| 16667 | 14210 | 40 | 117x145 | 0.783 / 0.813 | 0.002 / 0.555 | confidence |
| 11146 | 28291 | 70 | 85x122 | 0.550 / 0.645 | 0.007 / 0.813 | confidence |
| 2941 | 12320 | 70 | 158x75 | 0.703 / 0.675 | 0.042 / 0.980 | confidence |
| 3858 | 16539 | 100 | 105x83 | 0.631 / 0.587 | 0.032 / 0.813 | confidence |
| 3909 | 16680 | 100 | 180x184 | 0.774 / 0.805 | 0.008 / 0.847 | confidence |
| 7906 | 34787 | 100 | 255x270 | 0.804 / 0.857 | 0.739 / 0.495 | localization |
| 17596 | 27145 | 100 | 160x283 | 0.818 / 0.607 | 0.022 / 0.694 | confidence |

## DINO localization failures

The 59 failures are the GTs whose best saved post-NMS DINO box has meaningful
overlap but IoU below 0.50. They contain 32 visibility-20, 19 visibility-40,
4 visibility-70, and 4 visibility-100 people. Low visibility therefore
accounts for 51/59 (86.4%), compared with 344/749 (45.9%) of all val600 GTs.

### Reproducible geometry rules

For a GT box with width `w`, height `h`, center error `(dx, dy)`, and predicted
width/height ratios `rw`, `rh`:

- Normalized center error: `sqrt((dx/w)^2 + (dy/h)^2)`.
- Center displacement: normalized center error greater than 0.25.
- Scale mismatch: `rw` or `rh` outside `[0.75, 1.33]`.
- Wrong-object/ambiguous matching: the candidate overlaps another annotated GT
  at IoU >= 0.50 and by at least 0.10 more than the target GT.
- Uncertain: IoU below 0.50 while none of these diagnostic rules applies.

These thresholds are fixed descriptive rules, not fitted classifiers.

| Localization group | Count | Fraction |
|---|---:|---:|
| A. Center displacement only | 1 | 1.7% |
| B. Width/height or scale mismatch only | 35 | 59.3% |
| C. Mixed center and scale error | 23 | 39.0% |
| D. Wrong-object/ambiguous annotated-GT match | 0 | 0.0% |
| E. Uncertain by geometry rules | 0 | 0.0% |

Scale is not biased in one direction: 28 selected boxes are too small, 28 too
large, and 3 have similar area or compensating dimension errors. Visual review
shows boxes clipped to visible body fragments, expanded around vegetation, and
shifted around occluding trunks. It does not support claiming that every
unmatched region is background or an unannotated person.

### Pre-NMS versus post-NMS

| Diagnostic | Post-NMS selected box | Best raw candidate |
|---|---:|---:|
| Median best IoU | 0.4380 | 0.5577 |
| Mean best IoU | 0.4147 | 0.5450 |
| IoU >= 0.50 among the 59 | 0 | 40 |
| IoU >= 0.75 among the 59 | 0 | 2 |
| Median combined score | 0.4682 | 0.2068 |
| Median class probability | 0.3676 | 0.1017 |
| Median centerness probability | 0.5722 | 0.3257 |

All 59 best-localized raw candidates disappear before final output. Actual
overlapping higher-scored suppressors confirm NMS removal in 35 cases; the raw
candidate ranks below the top-300 cutoff in 24. Among the 40 raw candidates
with IoU >= 0.50, 24 are removed by NMS and 16 by top-300 selection. The score
ordering is therefore poorly aligned with localization quality for this failure
set. This is a confirmed post-processing mechanism, not an inference from the
final boxes alone.

The finding does not mean all 59 can be recovered at useful precision. The
better raw candidates have substantially lower class and centerness scores,
and only two reach IoU 0.75. A naive score reduction or larger proposal budget
would likely increase false positives.

## Comparative localization

The following uses each model's best saved post-NMS candidate for each of the
same 749 GTs, irrespective of operating threshold.

| Group | Model | IoU >= .50 | IoU >= .75 | Median center error | Median abs log width error | Median abs log height error |
|---|---|---:|---:|---:|---:|---:|
| All | YOLO | 95.2% | 66.9% | 0.052 | 0.067 | 0.057 |
| All | DINO | 92.1% | 41.3% | 0.090 | 0.121 | 0.101 |
| Vis 20 | YOLO | 88.5% | 47.1% | 0.088 | 0.091 | 0.088 |
| Vis 20 | DINO | 81.6% | 25.3% | 0.127 | 0.151 | 0.135 |
| Vis 40 | YOLO | 93.5% | 52.4% | 0.070 | 0.088 | 0.075 |
| Vis 40 | DINO | 88.8% | 26.5% | 0.107 | 0.122 | 0.136 |
| Vis 70 | YOLO | 99.0% | 78.8% | 0.043 | 0.057 | 0.050 |
| Vis 70 | DINO | 97.9% | 46.1% | 0.079 | 0.125 | 0.099 |
| Vis 100 | YOLO | 98.6% | 84.0% | 0.040 | 0.056 | 0.040 |
| Vis 100 | DINO | 98.1% | 61.8% | 0.066 | 0.106 | 0.070 |
| Small | YOLO | 94.2% | 45.8% | 0.086 | 0.087 | 0.070 |
| Small | DINO | 88.3% | 25.8% | 0.111 | 0.121 | 0.129 |
| Medium | YOLO | 94.4% | 67.3% | 0.054 | 0.075 | 0.062 |
| Medium | DINO | 92.9% | 44.0% | 0.086 | 0.107 | 0.101 |
| Large | YOLO | 96.6% | 75.2% | 0.040 | 0.050 | 0.042 |
| Large | DINO | 92.8% | 44.5% | 0.089 | 0.133 | 0.092 |

DINO's localization disadvantage is strongest in low visibility and is present
at every object-size bin. Small GTs have the lowest DINO IoU>=0.50 and IoU>=0.75
rates, but the IoU>=0.75 gap relative to YOLO is also large for medium and large
GTs. The issue is therefore not confined to tiny people.

## Manual visual review

The generated gallery uses green for GT, magenta for YOLO, and cyan for DINO.
It contains representative DINO-only and YOLO-only true positives, DINO
localization failures, and low-IoU cases requiring human review. The gallery is
diagnostic only: apparent people outside annotations are not labeled as real
without ground-truth evidence.

## Decision

### Confirmed bottleneck

The dominant measured bottleneck is score/localization misalignment before
top-k and NMS. For 40/59 localization failures, FCOS already produced an
IoU>=0.50 raw box, but its class/centerness score was too low to survive ranking.
Selected final boxes then show scale or mixed scale/center errors. Pure box
regression is not the whole problem.

### One recommended experiment

Add a lightweight FCOS localization-quality scalar trained from detached IoU
targets, and use it **only to rank overlapping candidates for top-k/NMS**. Keep
the existing combined FCOS confidence for final thresholding and reporting.
This restricted use avoids turning a quality estimate into another global
occlusion confidence penalty.

The controlled experiment should initialize from the current train5000
checkpoint, train the new ranker on train400 first, select any mixing exponent
on a training-only calibration subset, and compare unchanged versus
quality-ranked NMS at identical candidate count, NMS IoU, and FP/image budget.
Primary endpoints are recovery of the 40 valid raw boxes, AP75, AR@100, overall
recall, and R20 at FP/image <= 0.425.

This is preferred over repeating prior changes:

- GIoU did not improve AP75.
- FPN and stride-8 reduced performance.
- Box-only RoI refinement improved AP75 by 0.034 but improved matched-budget
  recall by only 0.008 and R20 by one GT.
- RoI person/quality rescoring suppressed low-visibility detections.
- Post-hoc class/centerness exponent changes did not improve the target FP
  budget.

The proposed ranker is still a hypothesis. Its success must be measured by
final detection metrics, not merely by retaining more raw boxes.

## Artifacts

- Analysis code: `scripts/analyze_dinov3_only_localization_failures.py`
- Machine-readable summary: `outputs/error_analysis/dinov3_only_localization_val600_20261015/analysis_summary.json`
- DINO-only details: `outputs/error_analysis/dinov3_only_localization_val600_20261015/dino_only_diagnostics.csv`
- All 59 localization failures: `outputs/error_analysis/dinov3_only_localization_val600_20261015/dino_localization_failures.csv`
- Error breakdown: `outputs/error_analysis/dinov3_only_localization_val600_20261015/localization_error_breakdown.csv`
- Comparative per-GT geometry: `outputs/error_analysis/dinov3_only_localization_val600_20261015/comparative_per_gt_localization.csv`
- Group tables: `outputs/error_analysis/dinov3_only_localization_val600_20261015/comparative_localization_by_group.csv`
- Gallery: `outputs/error_analysis/dinov3_only_localization_val600_20261015/gallery/`

