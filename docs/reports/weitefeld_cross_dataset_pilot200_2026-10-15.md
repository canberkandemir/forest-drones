# Weitefeld Cross-Dataset Pilot200

Date: 2026-10-15  
Status: completed zero-shot pilot; no training or threshold tuning

## Objective

This pilot tests whether the existing ForestPersons-trained YOLO11s,
DINOv3 ViT-S/16 LoRA + FCOS, and frozen Selective Fusion B systems transfer to
the Weitefeld aerial imagery without adaptation. The experiment is local to the
Apple M4 Mac and preserves every existing checkpoint and ForestPersons result.

The source data are the [peer-reviewed paper](https://doi.org/10.1038/s41597-026-07101-w),
the [static Zenodo record](https://zenodo.org/records/15848419), and the
[official image viewer](https://weitefeld.cg.jku.at/). The checked-in paper is
`docs/papers/4-Weitefeld.pdf`.

## Annotation audit

The core contains 10,659 priority-zone images and 34,424 annotation rows from
405 physical findings across all four classes. The individual physical finding
identifier is not present in `data.txt`, so the number of unique class-3
physical findings cannot be recovered exactly from this release.

Each `data.txt` row contains:

1. Source strip, image number, source marker x/y, class, and volunteer comment.
2. Target strip, image number, bounding-box lower-left x/y, height, width, and
   police comment.

Categories are `0=unknown`, `1=shelter`, `2=object`, and `3=person`. The official
viewer and released benchmark notebook both convert a box to top-left image
coordinates as `(x, y-height, x+width, y)`. Seven official boxes become
degenerate after clipping, including one class-3 row; these source records are
retained but excluded from localization evaluation.

After exact duplicate removal:

| Item | Count |
|---|---:|
| All annotation rows | 33,367 |
| Class 0 / 1 / 2 / 3 rows | 13,856 / 11,216 / 6,554 / 1,741 |
| Valid class-3 boxes | 1,740 |
| Images with a class-3 row | 1,416 |
| Images with any annotation | 9,063 |
| Provisional class-3 strip-track proxies | 48 |

The 48 proxies group contiguous views with matching strip, category, and
comments. They are useful for reducing near-duplicate sampling but are **not**
verified physical identities. The paper's 405 figure applies to all classes.

Class 3 is also not clean, confirmed person ground truth. Of the 1,741 class-3
rows, text-based conservative tiers are 405 explicit person-view descriptions,
342 ambiguous candidates, and 994 rows whose police comment names a non-person
or no abnormality. The tiers preserve the official class; they do not relabel
it. Consequently, results below are class-3 candidate localization diagnostics,
not conventional person-detection benchmark scores.

The labels are not exhaustive. Predictions outside published boxes therefore
cannot be called false people with confidence. Reported unmatched outputs per
image are diagnostic counts only.

## Pilot selection and download

The deterministic manifest uses seed `20261015` and covers all 15 flight
strips. It selects up to three spaced views from each provisional class-3 track
and then fills the pilot using priority-zone images with no published anomaly
row. Those context images are explicitly marked unknown, never negative.

| Pilot item | Count |
|---|---:|
| Images | 200 |
| Class-3 track-view images | 127 |
| Unannotated-context images | 73 |
| Valid class-3 boxes | 155 |
| Provisional tracks represented | 48 |
| Explicit / ambiguous / police-nonperson boxes | 41 / 28 / 86 |

The official per-strip `index.json` endpoints expose direct `rgb` URLs, so
selective individual-image transfer is supported without downloading 144.1 GB
of strip archives. The pilot downloaded 200/200 valid JPEGs at 8416x6032. The
actual transfer was 2,808,030,688 bytes (2.615 GiB) and took 567.6 seconds.

## Model protocol

No checkpoint, fusion rule, or operating threshold was changed.

| System | Frozen operating point |
|---|---|
| YOLO11s train5000 | confidence 0.04, NMS IoU 0.50, 640 input |
| DINOv3 LoRA FCOS train5000 | confidence 0.55, NMS IoU 0.50, short side 672 |
| Selective Fusion B | YOLO keep 0.10; rescue floor 0.01; DINO support 0.50; support IoU 0.30; area consistency 0.25; final NMS 0.50 |

Two source-coverage modes were compared:

- **Full image:** the complete 8416x6032 image is resized once.
- **Tiled:** 2048x2048 tiles with 256-pixel overlap, 20 tiles per image. Both
  models receive identical source coverage. Half-overlap center ownership and
  cross-tile NMS prevent duplicate edge predictions, and boxes are mapped back
  to original coordinates before fusion/evaluation.

The median official class-3 box is only 41x37 source pixels. It becomes roughly
3.1x2.8 pixels at YOLO full-image scale and 4.6x4.1 pixels at DINO full-image
scale. Tiling raises it to approximately 12.8x11.6 and 13.5x12.1 pixels,
respectively.

## Results

At IoU 0.50, all three frozen systems localized 0/155 official class-3 boxes in
both coverage modes. COCO AP, AP50, AP75, and AR@100 are therefore all zero.
No YOLO-only, DINO-only, or fusion-rescued class-3 detection exists at the
frozen operating points.

| Mode | System | Class-3 TP/FN | Recall@0.50 | Unmatched outputs/image* | Time/image |
|---|---|---:|---:|---:|---:|
| Full | YOLO11s | 0 / 155 | 0.000 | 1.145 | 0.407 s |
| Full | DINOv3 FCOS | 0 / 155 | 0.000 | 0.595 | 0.769 s |
| Full | Fusion B | 0 / 155 | 0.000 | 0.650 | n/a |
| Tiled | YOLO11s | 0 / 155 | 0.000 | 9.520 | 1.909 s |
| Tiled | DINOv3 FCOS | 0 / 155 | 0.000 | 4.885 | 4.134 s |
| Tiled | Fusion B | 0 / 155 | 0.000 | 5.890 | n/a |

\*Not definitive false positives because annotation coverage is incomplete and
class 3 is not verified person ground truth.

The result was audited against low-score predictions saved down to confidence
0.001. The official coordinate conversion was also checked against both the
viewer and benchmark notebook. This excludes an annotation-axis bug as the
cause of zero recall.

| Saved candidates | Best IoU max | GT >=0.10 IoU | GT >=0.25 IoU | GT >=0.50 IoU |
|---|---:|---:|---:|---:|
| YOLO full | 0.158 | 1 | 0 | 0 |
| YOLO tiled | 0.254 | 10 | 2 | 0 |
| DINO full | 0.130 | 1 | 0 | 0 |
| DINO tiled | 0.451 | 24 | 7 | 0 |

Tiling measurably improves proximity, especially for DINOv3, but is not enough
for box detection. Among annotations with any overlap, the median best-candidate
area ratio is 382x (YOLO full), 373x (DINO full), 14x (YOLO tiled), and 44x
(DINO tiled). The dominant mismatch is therefore target scale and box extent,
not merely confidence thresholding. At the frozen thresholds, only 4 YOLO and
10 DINO tiled candidates even contain the center of an official class-3 box.

## Interpretation

1. **Available labels:** there are 1,741 class-3 rows (1,740 valid) in 1,416
   images. The exact number of unique class-3 physical findings is unavailable;
   48 is only a reproducible strip-track proxy.
2. **Dataset suitability:** Weitefeld is valuable for tiny, occluded anomaly
   localization and qualitative transfer testing. Its current metadata is not
   a clean, exhaustive person-detection benchmark.
3. **Recall validity:** IoU-based class-3 box recall is reproducible, but it
   should not be described as confirmed-person recall. Definitive precision or
   false-positive rates are not supported.
4. **Model complementarity:** neither DINOv3 nor Selective Fusion B adds an
   IoU-0.50 class-3 detection beyond YOLO in this pilot.
5. **Effect of resizing:** full-image resizing destroys most usable target
   detail. Equal-coverage tiling improves low-IoU localization but does not
   solve the domain and scale mismatch.
6. **Scaling decision:** downloading the remaining 800 images is not justified
   for this unchanged IoU-0.50 detector protocol. It would add roughly 10.5 GiB
   and about 81 minutes of tiled inference while likely measuring the same
   mismatch more precisely.

The most useful next Weitefeld step is a manually adjudicated subset of
explicit class-3 views, ideally with physical finding IDs supplied by the
authors, followed by a predeclared metric appropriate to tiny targets (for
example, center-in-box or point localization alongside IoU). That should be
approved before any additional download. For the present project, the pilot is
a valid negative cross-dataset transfer result and does not motivate automatic
expansion to 1,000 images.

## Reproducibility artifacts

- Manifest: `outputs/weitefeld/pilot200_manifest.json`
- Download verification: `data/weitefeld/images/pilot200/download_verification.json`
- Raw predictions: `outputs/weitefeld/pilot200_predictions/`
- Fixed-threshold evaluation: `outputs/weitefeld/pilot200_evaluation/`
- Per-annotation diagnostics and gallery: `outputs/weitefeld/pilot200_analysis/`
- Loader/utilities: `src/forestpersons/weitefeld.py`
- Scripts: `scripts/prepare_weitefeld_pilot.py`,
  `scripts/download_weitefeld_pilot.py`, `scripts/predict_weitefeld_pilot.py`,
  `scripts/evaluate_weitefeld_pilot.py`, and
  `scripts/analyze_weitefeld_pilot.py`

