# Independent YOLO11s + DINOv3 FCOS Selective-Fusion Evaluation

Date: 2026-10-15

## Objective

This experiment independently evaluated the previously frozen YOLO11s +
DINOv3 FCOS selective-fusion rules. No model weights, fusion rules, score
calibrators, thresholds, or NMS parameters were changed after inspecting the
new holdout. The primary results use only the frozen operating points.

## Holdout Audit

The holdout was sampled from locally available images in the official
ForestPersons validation split after directly excluding val600, fusion
calibration200, and the earlier smoke30/visualization manifests. A subsequent
SHA-256 audit verified zero exact-image overlap with train400, train5000, and
the train-derived calibration100 manifest.

| Property | Verified result |
| --- | ---: |
| Images | 1,200 |
| GT people | 1,461 |
| Target images per visibility | 300 / 300 / 300 / 300 |
| GT at visibility 20/40/70/100 | 341 / 330 / 343 / 447 |
| Unique image IDs / filenames / SHA-256 hashes | 1,200 / 1,200 / 1,200 |
| Prior-subset filename or image-ID overlap | 0 |
| SHA-256 overlap with protected train/calibration/validation images | 0 |
| Unreadable images / dimension mismatches | 0 / 0 |

The subset is independently disjoint by exact image identity and original-file
SHA-256. It is **not scene-disjoint**: all 50 selected sequence folders also
occur in protected manifests. ForestPersons' sequential acquisition therefore
limits claims about independence across scene context or adjacent frames.

The deterministic manifest is
`outputs/subsets/forestpersons_val_fusion_independent_1200_seed20261015_manifest.json`.

## Frozen Protocol

The pre-holdout configuration is stored in
`experiments/fusion/configs/yolo_dino_selective_frozen_20261015.json`. Hashes of
the source calibration summary and both checkpoints were verified before
evaluation.

- **YOLO:** confidence threshold 0.04, input 640, model NMS IoU 0.50.
- **Selective A:** YOLO keep >= 0.075; DINO add >= 0.55 only below 0.30 IoU
  with retained YOLO; fusion NMS IoU 0.50; final fused-score threshold 0.
- **Selective B:** YOLO keep >= 0.10; rescue YOLO >= 0.01 when supported by
  DINO >= 0.50 at IoU >= 0.30 and area consistency >= 0.25; fusion NMS IoU
  0.50; final fused-score threshold 0.
- Existing ten-bin monotonic score calibrators from calibration200 were loaded
  verbatim. They were not refit on this holdout.
- Evaluation uses one-to-one matching at IoU 0.50 and standard COCO AR@100.

Inference used local CPU for YOLO and Apple MPS for DINOv3. YOLO took 59.9 s
(0.050 s/image); DINOv3 took 548.0 s (0.457 s/image).

## Primary Independent Results

| Method | Recall | Precision | FP/image | TP | FP | FN | R20 | R40 | R70 | R100 | AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO frozen | 0.9062 | 0.7662 | 0.3367 | 1,324 | 404 | 137 | 0.7859 | 0.8818 | 0.9475 | 0.9843 | **0.6373** |
| Selective A frozen | 0.9172 | 0.7334 | 0.4058 | 1,340 | 487 | 121 | **0.8035** | **0.9061** | 0.9534 | 0.9843 | 0.6238 |
| Selective B frozen | **0.9185** | 0.7599 | 0.3533 | **1,342** | 424 | **119** | **0.8035** | 0.9030 | **0.9534** | **0.9911** | 0.6260 |

All three frozen methods satisfy the strict `FP/image <= 0.425` criterion.
Unlike val600, no holdout threshold adjustment was needed.

Selective A rescued 24 GT detected by fusion but not frozen YOLO, lost eight
YOLO true detections, and produced a net gain of 16 people. It added 83 false
positives. Selective B rescued 23, lost five, and produced a net gain of 18
people while adding only 20 false positives.

## Paired Bootstrap

Ten thousand paired image-level bootstrap replicates were used. Each replicate
resampled the same images for YOLO and fusion, then recomputed aggregate recall.

| Comparison | Observed recall difference | Bootstrap mean | 95% percentile CI | P(diff > 0) |
| --- | ---: | ---: | ---: | ---: |
| Selective A - YOLO, overall | +0.0110 | +0.0110 | [+0.0035, +0.0184] | 0.9982 |
| Selective B - YOLO, overall | +0.0123 | +0.0123 | [+0.0049, +0.0202] | 0.9996 |
| Selective A - YOLO, R20 | +0.0176 | +0.0175 | [-0.0061, +0.0425] | 0.9023 |
| Selective B - YOLO, R20 | +0.0176 | +0.0175 | [-0.0059, +0.0402] | 0.9163 |

The overall recall improvements persist and their intervals exclude zero. The
observed R20 improvement is positive for both methods, but its confidence
interval includes zero; this holdout does not establish a statistically clear
heavy-occlusion gain by itself.

## AP/AR Tradeoff

| Method | AP | AP50 | AP75 | AR@100 |
| --- | ---: | ---: | ---: | ---: |
| YOLO | **0.5710** | **0.9027** | **0.6247** | **0.6373** |
| Selective A | 0.5506 | 0.8756 | 0.6083 | 0.6238 |
| Selective B | 0.5510 | 0.8762 | 0.6088 | 0.6260 |

Fusion improves threshold-specific recall but still weakens global score
ranking/localization metrics. It should therefore be described as a
recall-oriented operating mode, not as a detector that dominates YOLO.

## Optional Matched-Budget Diagnostic

This diagnostic is secondary because its thresholds inspect holdout labels.
YOLO reaches recall 0.9117 at threshold 0.03 and 0.3933 FP/image. The frozen A/B
points already fall below 0.425, so their diagnostic points equal their primary
frozen results. No primary conclusion depends on this diagnostic.

## Conclusions

1. **Does A improve R20?** Observed R20 increases from 0.7859 to 0.8035, but
   the paired 95% CI includes zero. The direction persists; statistical
   certainty for 20%-visible people is limited.
2. **Does B improve overall recall?** Yes. Recall rises from 0.9062 to 0.9185,
   a net 18 GT, with a paired 95% CI of [+0.0049, +0.0202].
3. **Do improvements persist independently?** Overall recall gains persist on
   this exact-image-disjoint holdout. Scene-level independence is not proven.
4. **Do methods satisfy the FP limit?** Yes: A is 0.4058 and B is 0.3533.
5. **Practical decision:** keep Selective B as an optional high-recall mode. It
   gives the larger net recall gain, retains precision close to YOLO, and adds
   only 20 FP over 1,200 images. Deprioritize Selective A because it gives less
   net gain with 83 extra FP. Do not start complex fusion training; the current
   lightweight B rule is sufficient for reporting and future full-split checks.

## Artifacts

- `scripts/create_fusion_independent_holdout.py`
- `scripts/evaluate_selective_fusion_holdout.py`
- `experiments/fusion/configs/yolo_dino_selective_frozen_20261015.json`
- `outputs/subsets/forestpersons_val_fusion_independent_1200_seed20261015_manifest.json`
- `outputs/fusion/yolo_dino_selective_independent1200_20261015/`
- `tests/test_selective_fusion_holdout.py`

