# DINOv3 ViT-S LoRA + FCOS Val600 Candidate Failure Analysis

Date: 2026-10-13

## Scope

This is a read-only diagnostic analysis of the best existing DINOv3 ViT-S +
LoRA + single-scale FCOS train5000 checkpoint on the selected ForestPersons
val600 subset.

No training, dataset download, checkpoint modification, or new model inference
was performed. The analysis uses saved COCO-format predictions.

Primary DINOv3 operating point:

- Checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`
- Saved predictions:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/coco_results.json`
- Confidence threshold: 0.55.
- IoU threshold: 0.50.
- FP/image at this point: 0.4167.

YOLO11s comparison point:

- Saved predictions:
  `outputs/evaluation_framework/val600_yolo11s_finetuned/coco_results.json`
- Confidence threshold: 0.05.
- IoU threshold: 0.50.
- FP/image at this point: 0.3467.

## Important Limitation

The saved predictions are post-NMS COCO result files. They do not contain:

- Pre-NMS candidate boxes.
- Raw class logits.
- Raw centerness logits.
- Separate classification and centerness scores.

Therefore, this report can diagnose best surviving post-NMS candidates and
combined confidence scores. It cannot prove that a missed ground-truth instance
was specifically suppressed by NMS, and it cannot separate classification-score
suppression from centerness-score suppression without rerunning inference with
extra logging.

## Overall Candidate Quality

The canonical val600 evaluation contains 749 visibility-labeled GT persons.

| Model | Saved post-NMS predictions | Detected at operating point | Best candidate IoU >= 0.50 | Best candidate IoU >= 0.75 | Mean best IoU | Median best IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3 LoRA + FCOS | 46,586 | 634 | 690 | 309 | 0.705 | 0.727 |
| YOLO11s fine-tuned | 2,622 | 677 | 713 | 501 | 0.781 | 0.824 |

Interpretation:

- DINOv3 usually produces a surviving candidate near each person: every GT has
  a best DINO candidate with IoU >= 0.20, and 690/749 have IoU >= 0.50.
- DINOv3 localization is much weaker than YOLO at stricter overlap: only
  309/749 DINO best candidates reach IoU >= 0.75, compared with 501/749 for
  YOLO.
- This agrees with the earlier AP50/AP75 gap: DINOv3 finds many people roughly
  but boxes are not tight enough.

## DINOv3 Miss Categories

For each GT person, the highest-IoU saved DINO candidate was inspected before
applying the operating confidence threshold. Misses are categorized using
surviving post-NMS candidates.

| Category | Count |
| --- | ---: |
| Detected at confidence 0.55 and IoU >= 0.50 | 634 |
| Candidate overlaps, but IoU < 0.50 | 59 |
| Candidate IoU >= 0.50, but confidence < 0.55 | 56 |
| No meaningful surviving candidate, IoU < 0.10 | 0 |
| Candidate eligible but unmatched by one-to-one matching | 0 |

So the 115 DINOv3 misses at this operating point are almost evenly split:

- 59 localization misses.
- 56 confidence/ranking misses.

## Visibility Breakdown

| Visibility | GT | Detected | Missed | Localization miss | Confidence miss | Best IoU >= 0.50 | Best IoU >= 0.75 | Mean best confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 174 | 116 | 58 | 32 | 26 | 142 | 44 | 0.583 |
| 40 | 170 | 137 | 33 | 19 | 14 | 151 | 45 | 0.656 |
| 70 | 193 | 177 | 16 | 4 | 12 | 189 | 89 | 0.718 |
| 100 | 212 | 204 | 8 | 4 | 4 | 208 | 131 | 0.770 |

Findings:

- 20%-visibility people are the main remaining weakness: they account for
  58/115 DINO misses.
- For 20% visibility, the failure split is balanced: 32 localization misses and
  26 confidence misses.
- Confidence rises monotonically with visibility, from mean best confidence
  0.583 at 20% visibility to 0.770 at 100% visibility.
- Best-candidate IoU also improves with visibility.

## Size, Camera, Pose, and Season

By size:

| Size bin | GT | Detected | Missed | Best IoU >= 0.50 | Best IoU >= 0.75 | Mean best confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, max side < 96 px | 120 | 86 | 34 | 106 | 31 | 0.609 |
| Medium, max side 96-192 px | 339 | 286 | 53 | 315 | 149 | 0.667 |
| Large, max side >= 192 px | 290 | 262 | 28 | 269 | 129 | 0.742 |

Small people are hardest: they have the lowest detection rate, lowest mean
confidence, and weakest IoU distribution.

By camera:

| Camera | GT | Detected | Missed | Best IoU >= 0.50 | Best IoU >= 0.75 | Mean best confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GoPro | 103 | 87 | 16 | 97 | 31 | 0.644 |
| See3CAM | 575 | 485 | 90 | 528 | 239 | 0.690 |
| Sony a57 (24-70) | 71 | 62 | 9 | 65 | 39 | 0.727 |

The Sony subset has the strongest localization and confidence, but it is also
much smaller than See3CAM in val600. Treat camera conclusions as descriptive,
not causal.

By pose:

- Lying: 141/170 detected.
- Sitting: 415/488 detected.
- Standing: 78/91 detected.

By season:

- Summer dominates the subset and most misses: 90/115 DINO misses.
- Winter has the strongest median best IoU in this subset.

## Confidence and Centerness

The saved score is the final combined detector score. In the DINOv3 FCOS
decoder, this score is derived from classification and centerness, but the saved
COCO result does not preserve the two components separately.

Measured facts:

- 56 GT persons have a surviving candidate with IoU >= 0.50 but score below
  the operating threshold.
- 26 of those confidence misses are at 20% visibility.
- For 20%-visibility confidence misses, mean best-candidate IoU is 0.627 and
  mean best-candidate score is 0.367.
- The highest 20%-visibility confidence miss has score 0.539, just below the
  0.55 operating point.

Hypothesis:

- Some heavily occluded people are represented by valid boxes but receive low
  combined confidence, possibly because centerness or classification confidence
  is weak when the visible evidence is off-center or sparse.

This remains a hypothesis until pre-NMS predictions with separate class and
centerness outputs are saved and analyzed.

## YOLO Comparison

At the selected operating points:

| Cross-model case | Count |
| --- | ---: |
| Both detected | 606 |
| YOLO only | 71 |
| DINOv3 only | 28 |
| Both missed | 44 |

YOLO's stronger result is not only a confidence issue. Its best-candidate IoU
distribution is substantially better:

- YOLO best IoU >= 0.75 for 501/749 GT.
- DINO best IoU >= 0.75 for 309/749 GT.

Among DINO localization misses, YOLO detects 35/59. Among DINO confidence
misses, YOLO detects 36/56. So YOLO helps in both failure modes.

## Visualizations

Representative examples were saved under:

`outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/visualizations/`

Each image shows:

- Green: ground truth.
- Cyan dashed: best saved DINO candidate.
- Magenta dotted: best saved YOLO candidate.

The selected examples include near-miss localization failures and low-confidence
DINO candidates with IoU >= 0.50.

## Artifacts

- Analysis script:
  `scripts/analyze_lora_candidate_failures.py`
- Summary:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/summary.json`
- Per-GT table:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/gt_candidate_analysis.csv`
- Visibility breakdown:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/dino_miss_categories_by_visibility.csv`
- Size breakdown:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/dino_by_size_bin.csv`
- Camera breakdown:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/dino_by_camera.csv`
- YOLO comparison:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013/dino_category_vs_yolo_category.csv`

## Recommendation

The single most promising next experiment is not another architecture change.
It is a controlled DINOv3 FCOS scoring/centerness diagnostic and calibration
experiment:

1. Rerun val600 inference once while saving pre-NMS candidates with separate
   classification score, centerness score, decoded box, and final score.
2. On a training-derived calibration split, compare fixed scoring formulas such
   as:
   - `sqrt(class_score * centerness)`
   - `class_score`
   - `class_score * centerness^0.25`
   - `class_score * centerness^0.5`
3. Lock the scoring rule before evaluating val600.
4. Keep the checkpoint unchanged and evaluate at matched FP/image <= 0.425.

Why this is the best next step:

- 56/115 current misses already have IoU >= 0.50 candidates.
- 26/58 missed 20%-visibility people are confidence misses.
- GIoU did not improve AP75, and the FPN pilot did not improve recall.
- A scoring/centerness diagnostic is cheap, scientifically controlled, and
  directly tests whether useful low-visibility candidates are being suppressed
  by the current FCOS score.

If scoring calibration does not recover recall without increasing false
positives, the next likely direction is localization-target refinement rather
than larger architecture changes.
