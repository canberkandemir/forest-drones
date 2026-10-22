# DINOv3 FCOS Ranking Oracle and Candidate-Recovery Study

Date: 2026-10-15

## Scope

This analysis uses only the saved ForestPersons val600 ground truth, the
preserved DINOv3 FCOS pre-NMS candidate dump, and the canonical saved COCO
predictions. No model inference, training, dataset download, or checkpoint
change was performed.

Reference detector: DINOv3 ViT-S/16 + LoRA + stride-16 FCOS train5000.

## Canonical reproduction

The offline reconstruction exactly reproduces the canonical post-processing:

1. Apply the 0.001 `sqrt(class_probability * centerness)` candidate floor.
2. Select the top 300 candidates per image.
3. Apply class-agnostic NMS at IoU 0.50, ordered by the combined score.
4. Save surviving predictions; apply operating confidence thresholds later in
   evaluation.

| Check | Result |
|---|---:|
| Raw candidates after floor | 1,917,636 |
| Generated final predictions | 46,586 |
| Canonical final predictions | 46,586 |
| Image/category/order mismatches | 0 |
| Maximum coordinate difference | 0.0 px |
| Maximum score difference | 4.82e-8 |

This is below the declared 1e-5 box and 1e-6 score tolerances. The oracle
analysis therefore proceeds from a verified reconstruction rather than an
approximation.

## Ranking definitions

The deployable score alternatives are:

- **Current:** `sqrt(class * centerness)`.
- **Class:** classification probability alone.
- **Centerness:** centerness probability alone.

Each alternative is tested as a top-300-only change, an NMS-order-only change,
and a combined change. For stage isolation, the original combined score remains
the final confidence used for AP and threshold sweeps.

### Strict IoU oracle

For each image, candidate-GT pairs are sorted by descending IoU and assigned
greedily, with at most one candidate per GT and one GT per candidate. Unassigned
candidates receive oracle quality zero. This prevents a background candidate
from receiving GT-derived credit and prevents one candidate from claiming
multiple people.

Two oracle ranking scores are tested:

- Oracle IoU.
- Oracle IoU multiplied by the current combined confidence.

For stage-isolation experiments, oracle quality changes only top-k and/or NMS
ordering; final confidence remains the current model confidence. A separate
strict ceiling retains only one-to-one assigned candidates and uses GT-derived
quality as the final score. That ceiling is deliberately non-deployable and is
only a candidate-geometry upper bound.

## Deployable score alternatives

Val600 thresholds selected to satisfy FP/image <= 0.425 are exploratory.

| Variant | AP | AP75 | AR@100 | Maximum recall | Recall <=0.425 FP/img | R20 <=0.425 | FP/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current canonical | 0.3587 | 0.2379 | 0.4704 | 0.9212 | 0.8465 | 0.6667 | 0.4167 |
| Class top-k only | 0.3588 | 0.2379 | 0.4730 | 0.9306 | 0.8465 | 0.6667 | 0.4167 |
| Class NMS only | 0.3484 | 0.2401 | 0.4684 | 0.9252 | 0.8238 | 0.6552 | 0.4017 |
| Class combined | 0.3484 | 0.2401 | 0.4688 | 0.9279 | 0.8238 | 0.6552 | 0.4017 |
| Centerness top-k only | 0.3369 | 0.2288 | 0.4207 | 0.8211 | 0.7971 | 0.5862 | 0.4133 |
| Centerness NMS only | 0.3570 | 0.2409 | 0.4698 | 0.9266 | 0.8371 | 0.6609 | 0.3983 |
| Centerness combined | 0.3350 | 0.2338 | 0.4196 | 0.8158 | 0.7904 | 0.5747 | 0.4017 |

Classification-only top-k modestly raises maximum recall and AR, but produces
no gain at the target FP budget. Classification-only NMS ordering and all
centerness-dominant variants reduce matched-budget recall. Existing scalar
scores therefore do not solve the ranking problem.

## Oracle stage isolation

| Oracle stage change | AP | AP75 | AR@100 | Maximum recall | Maximum R20 | Recall <=0.425 FP/img | R20 <=0.425 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-k only | 0.3588 | 0.2379 | 0.4742 | 0.9279 | 0.8333 | 0.8465 | 0.6667 |
| NMS order only | 0.2277 | 0.2355 | 0.6374 | 0.9640 | 0.9023 | 0.3818 | 0.2471 |
| Top-k + NMS order | 0.2207 | 0.2272 | 0.6415 | 0.9746 | 0.9310 | 0.3858 | 0.2529 |

The large AR and maximum-recall gains show that NMS ordering has considerable
geometric opportunity. The poor AP and matched-budget recall are equally
important: oracle-selected well-localized boxes retain their original low
class/centerness confidence, while many background candidates retain higher
confidence. Improving selection alone therefore cannot deliver the geometric
gain at a useful false-positive budget.

## Recovery of the 40 known candidates

These 40 GTs are the prior post-NMS localization failures with a raw candidate
at IoU >= 0.50.

| Variant | Best raw in top-300 | Best raw survives NMS | Any final IoU>=.50 at floor | Detected <=0.425 FP/img |
|---|---:|---:|---:|---:|
| Current canonical | 24/40 | 0/40 | 0/40 | 0/40 |
| Class top-k only | 33/40 | 2/40 | 7/40 | 0/40 |
| Class NMS only | 24/40 | 2/40 | 7/40 | 0/40 |
| Class combined | 33/40 | 3/40 | 9/40 | 0/40 |
| Centerness combined | 2/40 | 0/40 | 2/40 | 0/40 |
| Oracle top-k only | 40/40 | 5/40 | 5/40 | 0/40 |
| Oracle NMS only | 24/40 | 24/40 | 32/40 | 3/40 |
| Oracle top-k + NMS | 40/40 | 40/40 | 40/40 | 4/40 |

The combined oracle retains every known valid candidate, including all 20
visibility-20 and all 12 visibility-40 cases. With unchanged final confidence,
however, only one visibility-20, zero visibility-40, one visibility-70, and two
visibility-100 cases are detected at the target FP budget.

## Strict non-deployable ceiling

| GT-informed ceiling | AP | AP50 | AP75 | AR@100 | Recall | R20 | R40 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle IoU final score | 0.6436 | 0.9703 | 0.7228 | 0.6441 | 0.9746 | 0.9310 | 0.9588 | 0.0000 at IoU-score >=0.50 |
| Oracle IoU x current | 0.5930 | 0.9680 | 0.6481 | 0.6441 | 0.9746 | 0.9310 | 0.9588 | 0.0317 at score >=0.001 |

These numbers use validation GT both to select candidates and to remove
unassigned background. They are not deployable metrics and must not be compared
as if they were a trained detector. They establish that the raw candidate pool
contains sufficient box geometry for a much stronger detector if quality and
background can be predicted reliably.

## Decision

**Proceed with one controlled train400 quality-head pilot, but reject a
rank-only implementation.**

The pilot should predict detached IoU quality for positive FCOS locations and
use a predefined quality-aware confidence, such as
`sqrt(current_confidence * predicted_quality)`, consistently for top-k, NMS,
and final ranking. Keep an unchanged-current-score evaluation arm from the same
checkpoint. Choose any mixing exponent and operating threshold only on a
training-only calibration subset, then evaluate val600 once.

Why this is justified:

- Perfect top-k/NMS ordering recovers all 40 known valid candidates.
- The strict candidate ceiling raises AR@100 from 0.4704 to 0.6441 and R20 from
  0.6667 at the reference operating point to a geometric ceiling of 0.9310.
- Existing class or centerness scores cannot realize the gain.
- Rank-only oracle selection recovers just 4/40 at the target FP budget,
  proving that final confidence calibration must change too.

This remains a risky pilot rather than an expected improvement. Previous RoI
quality rescoring suppressed occluded people, and an imperfect quality target
may do the same. Success must require improved AP/AR and recall/R20 at
FP/image <= 0.425, not merely more candidates surviving NMS.

## Artifacts

- Script: `scripts/analyze_fcos_ranking_oracle.py`
- Summary: `outputs/error_analysis/dinov3_fcos_ranking_oracle_val600_20261015/analysis_summary.json`
- Stage table: `outputs/error_analysis/dinov3_fcos_ranking_oracle_val600_20261015/ranking_stage_comparison.csv`
- Candidate recovery: `outputs/error_analysis/dinov3_fcos_ranking_oracle_val600_20261015/candidate_recovery_per_gt.csv`
- Recovery summaries: `candidate_recovery_summary.csv` and
  `candidate_recovery_by_visibility.csv` in the same output directory.
- Strict ceiling: `strict_oracle_ceiling.csv`.
- Curves: `recall_vs_fp_per_image.png` and `r20_vs_fp_per_image.png`.

