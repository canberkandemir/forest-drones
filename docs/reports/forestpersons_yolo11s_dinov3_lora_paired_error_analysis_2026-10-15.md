# ForestPersons YOLO11s vs DINOv3 LoRA FCOS Paired Error Analysis

Date: 2026-10-15

## Scope

This analysis compares saved predictions from the best fine-tuned YOLO11s and
the original DINOv3 ViT-S/16 + rank-4 q/v LoRA + stride-16 FCOS train5000
checkpoint on the selected ForestPersons val600 pilot subset. No inference,
training, dataset download, architecture change, or checkpoint modification was
performed.

The newer positive-count focal-normalization train400 continuation is not used
for the primary comparison because it has a different continuation protocol.
Its R20=0.6839 result remains separately documented.

## Comparability verification

| Item | YOLO11s | DINOv3 LoRA + FCOS |
| --- | --- | --- |
| Checkpoint | `outputs/yolo11_finetune/train_runs/yolo11s_forestpersons_train5000_seed20261012_img640_b8_e50/weights/best.pt` | `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt` |
| Saved predictions | `outputs/evaluation_framework/val600_yolo11s_finetuned/coco_results.json` | `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/coco_results.json` |
| Input preprocessing | Ultralytics letterbox, `imgsz=640` | DINOv3 processor normalization, short side 672, stride-16 grid |
| Saved minimum confidence | 0.001 | requested 0.001; actual minimum saved score 0.0957 |
| NMS IoU | 0.50 | 0.50 |
| Reference threshold | 0.05 | 0.55 |

Both artifacts contain predictions for the same 600 manifest image IDs and use
the same 749 visibility-labeled GT annotations. Every saved result is mapped to
the one `person` category (`category_id=1`), all boxes have positive area, and
all 49,208 saved boxes lie within original-image bounds. Custom metrics use the
project's unchanged one-to-one matching at IoU 0.50.

The DINOv3 saved results contain no pair of post-NMS boxes above IoU 0.50.
YOLO contains nine pairs slightly above 0.50 after mapping/clipping back to
original image coordinates (maximum 0.581); most involve boundary-clipped,
very-low-score boxes. This does not affect the reproduced reference counts and
is not treated as evidence of an NMS failure.

The saved predictions are post-NMS. Conclusions about absence of a pre-NMS
proposal or NMS suppression cannot be made from these artifacts.

## Verified reference metrics

COCO metrics use pycocotools, IoU 0.50:0.95 and maxDets=100. Threshold metrics
use IoU 0.50 one-to-one matching.

| Model | AP | AP50 | AP75 | AR@100 | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s train5000 | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.05 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |
| DINOv3 LoRA FCOS train5000 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |

These reproduce the saved canonical summaries. They are pilot val600 results,
not official full-split ForestPersons benchmark scores.

## Paired GT outcomes

| Outcome | Count | Percent of 749 GT |
| --- | ---: | ---: |
| Detected by both | 606 | 80.91% |
| YOLO only | 71 | 9.48% |
| DINOv3 only | 28 | 3.74% |
| Missed by both | 44 | 5.87% |

### Visibility

| Visibility | GT | Both | YOLO only | DINO only | Both miss | YOLO recall | DINO recall | Oracle recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 174 | 103 | 31 | 13 | 27 | 0.7701 | 0.6667 | 0.8448 |
| 40 | 170 | 128 | 21 | 9 | 12 | 0.8765 | 0.8059 | 0.9294 |
| 70 | 193 | 175 | 13 | 2 | 3 | 0.9741 | 0.9171 | 0.9845 |
| 100 | 212 | 200 | 6 | 4 | 2 | 0.9717 | 0.9623 | 0.9906 |

Twenty-two of the 28 DINO-only detections are at visibility 20 or 40. The
complementary detections are therefore relevant to occlusion, although YOLO
still has more unique detections at every visibility level.

### GT size

Size uses the maximum original-image box side.

| Size | GT | Both | YOLO only | DINO only | Both miss | YOLO recall | DINO recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small, <96 px | 120 | 81 | 27 | 5 | 7 | 0.9000 | 0.7167 |
| Medium, 96-192 px | 339 | 267 | 27 | 19 | 26 | 0.8673 | 0.8437 |
| Large, >=192 px | 290 | 258 | 17 | 4 | 11 | 0.9483 | 0.9034 |

The largest recall gap is on small people: 18.33 percentage points. This is
consistent with DINOv3's stride-16 patch grid being less favorable for small
boxes, but that causal interpretation remains a hypothesis because model head,
training, augmentation, and localization losses also differ.

### Other metadata

- By pose, YOLO/DINO recall is 0.912/0.829 for lying, 0.902/0.850 for
  sitting, and 0.901/0.857 for standing.
- Camera-specific YOLO recall ranges from 0.901 to 0.915; DINO ranges from
  0.843 to 0.873. Sample sizes differ, so this is descriptive rather than
  causal.
- Under clear weather, YOLO/DINO recall is 0.902/0.831. Under rain it is
  0.911/0.899, indicating a smaller gap on this selected subset.
- Both models detect all 32 valley GTs. Valley is too small a group for a broad
  environment conclusion.

The complete pose, camera, weather, place, visibility, and size tables are in
`paired_group_summary.csv`.

## Candidate and localization analysis

For every GT, the highest-IoU saved post-NMS candidate was examined before the
reference operating threshold was applied.

| Model | Mean best IoU | Median | Best IoU >=0.50 | Best IoU >=0.75 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s | 0.781 | 0.824 | 713/749 | 501/749 |
| DINOv3 LoRA FCOS | 0.705 | 0.727 | 690/749 | 309/749 |

DINOv3's 115 operating-point misses comprise 59 localization misses and 56
confidence misses. For the 71 YOLO-only instances specifically, DINOv3 has 35
localization misses and 36 candidates with IoU >=0.50 but score below 0.55.
Thus the YOLO advantage is almost evenly split between DINOv3 localization and
confidence/ranking at IoU 0.50. The much larger AP50-to-AP75 gap and only
309 best candidates above IoU 0.75 establish localization as the larger
high-quality-detection weakness.

No YOLO-only case lacks a meaningful DINOv3 saved post-NMS overlap. This does
not prove that every DINO proposal pipeline is adequate: the saved artifact is
already after NMS and its candidate volume is unusually large.

## False-positive analysis

At the reference thresholds, YOLO has 208 unmatched predictions and DINOv3 has
250.

| Geometric category | YOLO | DINOv3 |
| --- | ---: | ---: |
| Duplicate/one-to-one competition, IoU >=0.50 | 7 | 39 |
| Localization failure, IoU 0.10-0.50 | 95 | 115 |
| Weak GT overlap, IoU <0.10 | 7 | 6 |
| No GT overlap | 99 | 90 |

These are geometric categories, not semantic judgments. In particular,
`no GT overlap` does not mean `nonhuman`.

A stratified manual review of 16 representative crops per model found:

- Visible person-related localization or duplicate detections in both models.
- Clear nonhuman examples in both models: YOLO fired on concrete memorial
  blocks and bright foreground leaves; DINOv3 fired on vegetation,
  branches, and dark forest regions.
- Several high-confidence zero-overlap boxes visibly cover additional
  person-like figures not represented by the nearby GT boxes. These are marked
  `possible_missing_gt_person`, not background false positives.
- One dark YOLO crop and one person/equipment crop remain uncertain because
  image resolution is insufficient for a defensible label.

The sample was deliberately stratified by geometric category and high score,
so its semantic-label frequencies cannot estimate prevalence over all 458
unmatched predictions. The exhaustive geometric table and separate manual
review CSV preserve this distinction.

**Conclusion:** there is no measured evidence that DINOv3 is better at avoiding
nonhuman false detections. It has more unmatched predictions at its reference
point, and manual review finds genuine nonhuman examples for both detectors.
A verified person-absent benchmark is required for a stronger semantic false
alarm conclusion.

## Fair false-alarm comparison

Thresholds below are selected on val600 and are exploratory.

### Matched FP/image

| FP/image budget | YOLO recall | YOLO FP/image | DINO recall | DINO FP/image |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.8919 | 0.2450 | 0.7984 | 0.2467 |
| 0.3467 | 0.9039 | 0.3467 | 0.8331 | 0.3400 |
| 0.425 | 0.9039 | 0.3467 | 0.8465 | 0.4167 |
| 0.50 | 0.9146 | 0.4950 | 0.8545 | 0.4983 |
| 1.00 | 0.9266 | 0.8083 | 0.8745 | 0.9833 |

### Matched recall

| Target recall | YOLO actual recall | YOLO precision | YOLO FP/image | DINO actual recall | DINO precision | DINO FP/image |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.8024 | 0.9120 | 0.0967 | 0.8051 | 0.7966 | 0.2567 |
| 0.85 | 0.8505 | 0.8934 | 0.1267 | 0.8518 | 0.6868 | 0.4850 |
| 0.90 | 0.9039 | 0.7650 | 0.3467 | 0.9012 | 0.1796 | 5.1383 |

There is no meaningful tested operating region up to 0.92 recall where DINOv3
uses fewer false alarms at comparable recall. At roughly 0.85 recall, DINOv3
costs 3.8x more FP/image. At roughly 0.90 recall, it costs about 14.8x more.

For YOLO, moving from approximately 0.85 to 0.90 recall gains 40 TP and adds
132 FP across 600 images. For DINOv3, the same transition gains 37 TP and adds
about 2,792 FP. DINOv3's low-score candidate ranking therefore becomes
impractical well before YOLO's.

## Person-absent images

No locally verified person-absent evaluation set is available. Val600 contains
zero annotation-free images. The official train annotations contain only 13
annotation-free images, but absence of COCO annotations is not verified absence
of people; prior EfficientAD feasibility work explicitly rejected that
assumption. No person-absent evaluation was performed.

## Complementarity and fusion

The oracle union detects 705/749 GT, recall 0.9413. Relative to YOLO alone,
this is an upper-bound gain of 28 GT or 3.74 percentage points. Oracle R20 is
0.8448, 7.47 points above YOLO's 0.7701. This is not an achievable fusion
result because ground truth selects the correct detector per person.

Previous saved-prediction fusion should not be repeated unchanged:

- At FP/image <=0.425, union/NMS recall was 0.8838 and weighted-box-fusion
  recall was 0.8652, both below YOLO's 0.9039.
- Different confidence scales and DINOv3's many low-score false candidates
  prevent naive union from reaching the oracle benefit.

The 28 DINO-only detections establish real complementarity, especially under
occlusion, but not enough evidence to prioritize another naive fusion attempt.
A future learned proposal rescoring experiment would need a training-derived
calibration set and independent validation.

## Research conclusions

### A. Why is DINOv3 behind YOLO11s?

Measured causes are weaker localization, poorer low-score ranking, and a large
small-person recall gap. YOLO has 192 more GTs with best-candidate IoU >=0.75,
and DINOv3 requires far more false positives to approach YOLO's recall.

### B. Is the main issue confidence, localization, or missing candidates?

At IoU 0.50, localization and confidence are nearly tied: 59 versus 56 DINO
misses overall and 35 versus 36 among YOLO-only cases. There is no evidence of
a missing *saved post-NMS* overlap for the YOLO-only set. At stricter quality,
localization is the clearest weakness. Pre-NMS candidate absence remains
unknown.

### C. Is DINOv3 better at avoiding false detections of nonhuman objects?

No defensible evidence supports that claim. DINOv3 has more unmatched boxes at
the reference point and worse false-alarm cost at matched recall. Both models
produce visually confirmed nonhuman detections; possible missing annotations
make raw unmatched counts semantically imperfect.

### D. Are the models complementary enough for further fusion?

They are modestly complementary: 28 DINO-only GTs, mostly at 20/40 visibility.
However, prior naive fusion underperformed YOLO at matched FP budgets. This
supports eventual learned/calibrated rescoring, not another uncalibrated union.

### E. Single most promising next DINOv3 experiment

Run the already-justified **matched train5000 positive-count focal-normalization
experiment with image-ID-safe annotation lookup**, keeping architecture,
initialization, optimizer, resolution, assignment, and evaluation fixed.

Why this is the most defensible next step:

- 56/115 DINO misses are confidence misses; 36/71 YOLO-only GTs already have a
  valid low-confidence DINO box.
- The controlled train400 result improved AP, AP50, AP75, AR, matched recall,
  and R20 over its corrected mean-loss control without increasing FP/image.
- Post-hoc scoring calibration, FPN, stride-8, YOLO-style, and RoI rescoring
  experiments did not yield a substantial recall improvement.

Expected improvement is modest, and this experiment will not solve the separate
localization deficit. Both train5000 arms should be matched because scaling only
the positive-count arm would confound loss normalization with corrected
annotations and extra optimization.

## Outputs

- Reproducible analysis: `scripts/analyze_yolo_dinov3_paired_errors.py`
- Per-GT table: `per_gt_paired_comparison.csv`
- Metadata summaries: `paired_group_summary.csv`
- Exhaustive unmatched predictions: `false_positive_comparison.csv`
- Manual review: `manual_false_positive_review.csv`
- Matched-budget and matched-recall tables: `matched_fp_budget_comparison.csv`,
  `matched_recall_comparison.csv`
- Curves: `precision_recall.png`, `recall_vs_fp_per_image.png`,
  `r20_vs_fp_per_image.png`
- Representative crops: `false_positive_review/` and `paired_examples/`
- Machine-readable summary: `analysis_summary.json`

All generated artifacts are under
`outputs/error_analysis/yolo11s_vs_dinov3_lora_paired_val600_20261015/`.
