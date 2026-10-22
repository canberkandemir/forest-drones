# Weitefeld Annotation Verification and Tiny-Object Localization

Date: 2026-10-16  
Status: completed analysis of existing pilot200 images and saved predictions  
Scope: no inference, training, checkpoint change, or download

## Purpose

The original Weitefeld pilot produced zero IoU-0.50 matches for frozen
ForestPersons YOLO11s, DINOv3 ViT-S LoRA + FCOS, and Selective Fusion B. This
audit separates four possible causes: target size, predicted-box scale,
annotation semantics, and complete candidate failure.

All 155 official class-3 boxes were reviewed through the official police
comments and rendered into manual-review panels. Numerical diagnostics use only
the saved full-image and tiled predictions. IoU 0.50 remains the strict metric;
IoU 0.10/0.25, center distance, and per-GT best-candidate values are explicitly
diagnostic.

## Annotation review

No annotation was automatically promoted to `confirmed_person`. Only police
comments explicitly naming a non-person object, no finding, or no abnormality
were prepopulated as provisional `non_person`.

| Provisional review result | Count | Meaning |
|---|---:|---|
| Confirmed person | 0 | No ground-confirmed person statement exists in the pilot comments |
| Non-person | 134 | Police note explicitly names a non-person/no abnormality |
| Uncertain, person-like wording | 18 | 15 group-of-people and 3 cyclist labels, all marked excluded/not checked |
| Uncertain, no police comment | 3 | `NA` |

Thus, 18 annotations contain explicit person/cyclist wording, but none is
ground verified and all remain uncertain. The current 155 boxes cannot support
a defensible confirmed-person recall estimate.

Annotation multiplicity introduces another ambiguity. Twenty-five images have
more than one class-3 box, with up to three boxes per image. Among 31 within-image
box pairs, 11 have IoU >=0.50 and two exceed 0.90. These may be repeated or
overlapping findings, but `data.txt` exposes no physical finding ID. A single
prediction can therefore geometrically cover multiple annotations while
one-to-one matching credits only one.

The review package contains:

- `annotation_review.csv`, with empty `reviewer_status` and `reviewer_notes`
  fields for human correction;
- 155 individual panels showing full context, a larger context crop, a
  magnified target crop, the official box, and the best saved YOLO/DINO tiled
  candidates;
- 39 four-panel contact-sheet pages and a local `gallery.html` index;
- `REVIEW_INSTRUCTIONS.md`, which defines the allowed reviewer statuses.

## Target scale

The original-resolution class-3 boxes comprise 58 COCO-small, 90 medium, and
seven large boxes when the standard 32/96-pixel area boundaries are applied to
the source image. Their median size is 41x37 pixels in an 8416x6032 image.

| Input path | Median target size at model input | Both dimensions <4 px | <8 px | <16 px |
|---|---:|---:|---:|---:|
| YOLO full image, 640 | 3.12x2.81 | 102/155 | 148/155 | 155/155 |
| DINO full image, short side 672 | 4.57x4.12 | 55/155 | 127/155 | 148/155 |
| YOLO 2048 tile to 640 | 12.81x11.56 | 0/155 | 19/155 | 96/155 |
| DINO 2048 tile, short side 672 | 13.45x12.14 | 0/155 | 19/155 | 95/155 |

Full-image resizing therefore removes most usable target detail. Existing
equal-coverage tiling materially improves the input representation, but more
than half of the targets remain below 16 pixels in both dimensions.

## Frozen operating-point results

These figures use the unchanged ForestPersons thresholds and one-to-one
matching. Lower IoU columns are diagnostics, not replacement detection metrics.

| Coverage | System | Recall@0.10 | Recall@0.25 | Recall@0.50 |
|---|---|---:|---:|---:|
| Full | YOLO11s | 0/155 | 0/155 | 0/155 |
| Full | DINOv3 FCOS | 0/155 | 0/155 | 0/155 |
| Full | Fusion B | 0/155 | 0/155 | 0/155 |
| Tiled | YOLO11s | 3/155 (1.94%) | 0/155 | 0/155 |
| Tiled | DINOv3 FCOS | 1/155 (0.65%) | 1/155 (0.65%) | 0/155 |
| Tiled | Fusion B | 0/155 | 0/155 | 0/155 |

The frozen detector outputs do not produce valid IoU-0.50 class-3 detections.
This remains true even after tiling, so the result is not caused solely by
confidence thresholding or full-image resizing.

## Saved-candidate opportunity

Predictions were saved down to score 0.001. The table below asks whether each GT
has any overlapping saved candidate. It is an annotation-assisted,
non-deployable upper-bound diagnostic: candidates are not one-to-one and often
fall below the frozen threshold.

| Saved predictions | GT best IoU >=0.10 | >=0.25 | >=0.50 | Maximum IoU |
|---|---:|---:|---:|---:|
| YOLO full | 1 | 0 | 0 | 0.158 |
| DINO full | 1 | 0 | 0 | 0.130 |
| YOLO tiled | 10 | 2 | 0 | 0.254 |
| DINO tiled | 24 | 7 | 0 | 0.451 |

DINO tiled has the stronger low-score localization opportunity: its best IoU
exceeds YOLO's for 113/155 annotations, versus 24/155 in the opposite direction
(18 ties). This advantage is not yet deployable performance.

By the fixed source-pixel size bins:

| Size bin | Annotations | YOLO >=0.10 / >=0.25 | DINO >=0.10 / >=0.25 |
|---|---:|---:|---:|
| Small (<32²) | 58 | 1 / 0 | 0 / 0 |
| Medium (32²-96²) | 90 | 9 / 2 | 17 / 5 |
| Large (>=96²) | 7 | 0 / 0 | 7 / 2 |

The DINO advantage is concentrated in medium and large annotation boxes. It
does not demonstrate improved localization of the 58 smallest boxes. At the
provisional track level, YOLO has a >=0.10 candidate in 6/48 track proxies and
DINO in 11/48; at >=0.25 the counts are 2/48 and 4/48. Full strip and track
tables are saved in `localization_group_summary.csv`.

## Center and scale errors

The diagnostic uses the per-GT best-overlap candidate when overlap exists;
otherwise it uses the nearest-center saved candidate. A candidate is
predeclared as nearby when its center is within two GT diagonals. Area ratios
outside `[0.25, 4.0]` are a severe scale mismatch.

| Mode/model | Median center error | Any-overlap GT | Median area ratio among overlaps | Main diagnosis |
|---|---:|---:|---:|---|
| YOLO full | 1,166 px / 23.12 diagonals | 14/155 | 382x | 140 no-nearby candidates |
| DINO full | 496 px / 8.15 diagonals | 89/155 | 373x | 89 overlap but wrong scale |
| YOLO tiled | 424 px / 7.06 diagonals | 32/155 | 14.1x | 113 no-nearby candidates |
| DINO tiled | 104 px / 1.86 diagonals | 134/155 | 44.4x | 125 overlap but wrong scale |

For tiled YOLO, missing the relevant region is the dominant failure. For tiled
DINO, regional activation is common, but its person boxes are much larger than
the tiny official annotations. DINO's median nearest-center distance is 84
pixels (1.51 GT diagonals), versus YOLO's 394 pixels (7.06 diagonals), confirming
a regional-localization advantage alongside a serious box-scale mismatch.

The strongest DINO overlaps also illustrate the semantic problem. The maximum
IoU 0.451 and second-highest 0.438 correspond to 154x154 boxes whose police
comment says uprooted tree/no abnormality. The highest uncertain examples are
group-of-people boxes marked excluded/not checked. A high overlap therefore
does not establish successful person detection.

## Findings

### Confirmed

1. **Annotation semantics are the primary evaluation-validity blocker.** There
   are zero ground-confirmed persons, 134 explicit non-person findings, and 21
   uncertain cases.
2. **Full-image resizing is unsuitable for these targets.** Nearly every target
   becomes less than 8 pixels in both dimensions for YOLO; 127/155 do so for
   DINO.
3. **Tiling helps but does not solve strict localization.** Neither detector has
   a saved candidate at IoU >=0.50.
4. **YOLO primarily misses candidate regions.** It has no nearby tiled
   prediction for 113/155 annotations.
5. **DINO primarily has box-scale mismatch.** It overlaps 134/155 regions, but
   125 are severe scale mismatches. Its candidate-level localization advantage
   is measurable but not an operational person-detection gain.
6. **The most difficult tiny objects remain unsolved.** DINO has no IoU >=0.10
   candidate among the 58 small boxes.

### Hypotheses

- ForestPersons-trained person heads may regress the spatial extent of a human,
  whereas Weitefeld boxes often tightly enclose a tiny visible anomaly fragment.
  This would explain correct regional activation with inflated boxes, but the
  current metadata cannot prove it.
- Some annotation multiplicity may represent repeated labels of the same
  physical finding. The absent finding ID prevents confirmation.

## Decision

- **Current suitability:** Weitefeld pilot200 is not suitable for a defensible
  confirmed-person recall benchmark. It remains useful for qualitative tiny
  anomaly localization and cross-domain failure analysis.
- **DINO versus YOLO:** DINO has a clear low-score, tiled regional-localization
  advantage, especially for the 21 uncertain annotations (11 versus two at
  IoU >=0.10), but no strict IoU-0.50 advantage and no advantage on the smallest
  boxes. Its boxes are substantially oversized.
- **Further crop/tiling work:** A bounded finer-tiling study is scientifically
  reasonable only after manual adjudication of the 21 uncertain annotations
  and with a predeclared point/center metric alongside IoU. GT-centered crops
  would be annotation-assisted and must not be reported as deployable detector
  performance. Repeating the experiment on all 155 semantically mixed boxes is
  not justified.
- **Dataset direction:** Use a dedicated, exhaustively labeled person-detection
  dataset such as HERIDAL or SARD for the next quantitative cross-dataset person
  benchmark, after auditing its split, license, and annotation quality. Keep
  Weitefeld as a supplementary anomaly/occlusion case study. Do not download the
  remaining 800 Weitefeld images now.

## Artifacts

Canonical output directory:
`outputs/weitefeld/annotation_review_verified_20261016/`

- `annotation_review.csv`
- `REVIEW_INSTRUCTIONS.md`
- `gallery.html`
- `review_panels/` (155 panels)
- `contact_sheets/` (39 pages)
- `per_annotation_localization_diagnostics.csv`
- `localization_group_summary.csv`
- `overlapping_annotation_pairs.csv`
- `best_iou_histogram.png`
- `tiled_candidate_area_ratio.png`
- `summary.json`

Reproducible entry point: `scripts/review_weitefeld_annotations.py`.

