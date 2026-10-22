# SARD Above-Canopy RGB Cross-Dataset Pilot

Date: 2026-10-16

## Scope

This experiment measures zero-shot transfer of the frozen ForestPersons
detectors to above-canopy and oblique RGB search-and-rescue imagery. No model,
checkpoint, score threshold, fusion parameter, or evaluation rule was trained
or selected on SARD.

The primary benchmark uses all 200 images in the official SARD v1.0
`mountains` scenario. A separate, post-hoc tiling diagnostic uses frames
91-120 only and is not mixed with the primary result.

## Dataset audit and selection

### HERIDAL

- Primary paper: Bozic-Stulic et al., *Deep Learning Approach in Aerial
  Imagery for Supporting Land Search and Rescue Missions*, IJCV (2019),
  DOI [10.1007/s11263-019-01177-1](https://doi.org/10.1007/s11263-019-01177-1).
- The paper and later literature describe high-resolution wilderness aerial
  imagery and a held-out test set, making HERIDAL the stronger thematic match.
- The original host at
  `ipsar.fesb.unist.hr/HERIDAL database.html` refused connections during this
  audit. Exact current files, official split counts, terms of use, and license
  could therefore not be verified.
- Secondary sources disagree on the released split totals. Those values are
  not treated as verified and no third-party copy was downloaded.

### SARD v1.0

- Official record: [Zenodo DOI 10.5281/zenodo.3924925](https://zenodo.org/records/3924925).
- Official code/data source:
  [gvessio/uav-search-and-rescue v1.0](https://github.com/gvessio/uav-search-and-rescue/tree/v1.0),
  commit `8961f735580b9127e24f1a8bad3ad8bbbe1a3d03`.
- The release has 310 RGB frames: 200 `mountains` and 110 `beaches`. It is
  explicitly intended for testing and evaluation, with no official
  train/validation/test split.
- Labels are normalized YOLO text files. Category 1 is person. Each released
  frame has a label file, and empty files denote released negative frames.
- The Zenodo archive is 145,644,728 bytes (138.9 MiB; published MD5
  `f46be920acb9a00fc46a0cc6529315be`). A sparse checkout transferred only the
  mountain directory: 100,492,194 source bytes (95.8 MiB). The working tree is
  97 MiB; Git's partial object cache raises local disk use to 216 MiB.
- Zenodo reports open access with license identifier `other-open`, but the
  record does not name a concrete license and the GitHub repository has no
  license file. The data should remain research-only pending written license
  clarification; redistribution and commercial rights are not established.

SARD was selected because its official source was directly accessible, the
entire mountain scenario fits the requested 200-image pilot, and it provides
person boxes plus released negative frames. HERIDAL remains preferable for a
future wilderness benchmark if official access and licensing can be resolved.

## Integrity and sequence structure

All 200 image/label pairs opened successfully. Every image is 3840x2160.
There are 305 valid person boxes in 101 positive images and 99 empty-label
images. This differs by one frame from the README's 100/100 description; the
files themselves are the source of truth for this experiment.

The images are strongly sequence-correlated. A deterministic 16x16 dHash audit
found seven clip transitions at frames 31, 61, 91, 121, 151, and 190. Median
adjacent-frame Hamming distance is only 7/256 bits. Results therefore describe
seven correlated source-video segments, not 200 independent scenes.

The released one-label-file-per-frame structure supports apparent FP/image
reporting. However, annotation exhaustiveness has not been independently
audited, so FP values should not be interpreted as stronger than the release's
annotation assumptions.

## Target scale

| Quantity | Width | Height |
| --- | ---: | ---: |
| Source min | 21.0 px | 55.0 px |
| Source median | 55.0 px | 108.0 px |
| Source max | 185.0 px | 421.0 px |
| Median at YOLO 640 letterbox scale | 9.17 px | 18.00 px |
| Median at DINO short-side 672 | 17.11 px | 33.60 px |

Frames 91-120 are the hardest clip: 150 people, median 40x95 source pixels,
or approximately 6.7x15.8 pixels at YOLO input scale. No primary pipeline
detects any of these 150 people at IoU 0.50.

## Frozen protocol

- YOLO11s: ForestPersons train5000 checkpoint, `imgsz=640`, CPU, prediction
  floor 0.001, model NMS 0.50, frozen operating threshold 0.04.
- DINOv3: ViT-S/16 LoRA plus FCOS train5000 checkpoint, short side 672, MPS,
  prediction floor 0.001, model NMS 0.50, frozen operating threshold 0.55.
- Selective Fusion B: exact frozen ForestPersons calibrators and parameters;
  reliable YOLO threshold 0.10, rescue floor 0.01, DINO support 0.50, support
  IoU 0.30, area consistency 0.25, final fusion threshold 0.0.
- Matching: one-to-one person matching at IoU 0.50. COCO AP/AR uses
  pycocotools with maxDets `[1, 10, 100]`.

SARD has no visibility, occlusion, pose, weather, or viewpoint attributes, so
ForestPersons-style R20/R40/R70/R100 metrics are unavailable.

## Primary results: mountain200

| Method | TP | FP | FN | Recall | Precision | Apparent FP/image | AP | AP50 | AP75 | AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 24 | 1715 | 281 | 0.0787 | 0.0138 | 8.575 | 0.0052 | 0.0161 | 0.0002 | 0.0482 |
| DINOv3 LoRA FCOS | 40 | 160 | 265 | 0.1311 | 0.2000 | 0.800 | 0.0160 | 0.0367 | 0.0090 | 0.1000 |
| Selective Fusion B | 22 | 1150 | 283 | 0.0721 | 0.0188 | 5.750 | 0.0048 | 0.0128 | 0.0001 | 0.0249 |

At the fixed detector operating points, 17 GT people are detected by both
YOLO and DINOv3, 7 by YOLO only, 23 by DINOv3 only, and 258 by neither. Fusion
B rescues 1 person missed by YOLO but loses 3 YOLO true detections, for a net
loss of 2.

All 207 COCO-medium source boxes are missed at the fixed operating points.
Among 98 COCO-large boxes, recall is 0.2449 for YOLO, 0.4082 for DINOv3, and
0.2245 for Fusion B.

The low-score saved-candidate upper opportunity is also weak: YOLO has an
IoU-0.50 candidate for 43/305 GT at 15.15 FP/image, while DINOv3 has one for
67/305 at 78.74 FP/image. These are diagnostic candidate floors, not useful
operating points.

Inference including image loading took 0.094 seconds/image for YOLO on CPU and
0.521 seconds/image for DINOv3 on MPS.

## Post-hoc tiling diagnostic

Frames 91-120 were tiled into eight 1280x1280 crops per image with 160-pixel
overlap, half-overlap center ownership, and cross-tile NMS. This subset and
configuration were chosen after inspecting the primary scale failure and are
not an independent benchmark.

| Method | TP / 150 | Recall | Apparent FP/image | AR@100 | Time/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11s tiled | 3 | 0.0200 | 44.1 | 0.0167 | 0.605 s |
| DINOv3 tiled | 0 | 0.0000 | 2.8 | 0.0187 | 1.462 s |
| Fusion B tiled | 1 | 0.0067 | 37.2 | 0.0027 | derived |

Tiling increases target pixels but does not recover useful detection at the
frozen thresholds. It sharply increases YOLO and fusion false alarms and does
not address the dominant domain/localization mismatch.

## Conclusions

1. **Generalization is poor.** ForestPersons-trained detectors do not transfer
   reliably to SARD's oblique, rocky mountain footage, especially for small
   people.
2. **DINOv3 is more complementary and stronger here.** It detects 23 people
   missed by YOLO, versus 7 in the opposite direction, and doubles AR@100.
   Its absolute recall remains only 13.1%.
3. **Frozen Fusion B does not transfer.** Its ForestPersons calibration removes
   useful cross-domain detections and retains many false alarms. It is worse
   than both individual detectors on recall and AR.
4. **Scale matters but is not the only issue.** The larger-person clip is much
   easier, while the 150-person small-target clip has zero primary recall.
   Tiling's failure shows that extra pixels alone are insufficient.
5. **Expanding SARD is unnecessary for the immediate question.** The complete
   200-frame mountain scenario is already evaluated. The beach scenario is
   available but does not improve forest relevance.
6. **Recommended next step before RGB-thermal or multi-view work:** obtain the
   official HERIDAL release and written license clarification, then run the
   same frozen protocol on its official held-out split. If that remains
   unavailable, choose a modern exhaustively labeled wilderness UAV dataset
   with scene-level splits rather than tuning on these seven SARD clips.

## Artifacts

- `data/sard/official_v1.0/dataset/mountains/`
- `outputs/sard/pilot200_preparation_20261016/`
- `outputs/sard/pilot200_predictions/`
- `outputs/sard/pilot200_evaluation_20261016_verified/`
- `outputs/sard/pilot200_analysis_20261016_final/`
- `outputs/sard/tiling_diagnostic_segment4_preparation_20261016/`
- `outputs/sard/tiling_diagnostic_segment4_predictions/`
- `outputs/sard/tiling_diagnostic_segment4_evaluation_20261016/`

