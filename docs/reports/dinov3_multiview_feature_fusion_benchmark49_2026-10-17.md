# DINOv3 Multi-View Feature Fusion Benchmark49

Date: 2026-10-17

## Objective

Test whether geometry-aligned averaging of DINOv3 patch features improves the
existing ForestPersons-trained DINOv3 ViT-S + LoRA + FCOS detector beyond the
initial eight-group, GT-qualified pilot. No training, checkpoint change,
threshold tuning, or data download was performed.

## Benchmark construction

Groups were selected from the official ForestPersons validation split using
only file/sequence metadata and image pixels. Annotation boxes, visibility,
and model predictions were not read during selection.

For every candidate sequence, the selector searched for one reference frame,
one earlier source frame, and one later source frame at fixed offsets. ORB
ratio-test correspondences and a RANSAC homography had to satisfy all of the
following image-only checks:

- at least 40 ratio-test matches and 30 RANSAC inliers;
- inlier ratio at least 0.50;
- median registration-scale reprojection error at most 2 pixels;
- at least 70% projected reference-image overlap;
- at least 5% inlier spatial spread in both images; and
- projected source-image area between 0.50 and 1.50 of reference area.

The requested target was 50 groups. Forty-nine groups passed without relaxing
the criteria: 147 unique images from 49 unique sequences. The process rejected
640 reference candidates and evaluated 4,690 source-frame attempts. The main
source rejection reasons were too few ratio-test matches (3,619), missing
offset frames (388), too few RANSAC inliers (124), low inlier ratio (85), and
low image overlap (59).

Train/validation filename and sequence overlap are both zero. Exact overlap
with the previous eight-group pilot is also zero. Six of the 49 sequence names
occur in the earlier pilot, so this is not fully scene-independent from that
exploratory work. Numeric COCO image IDs overlap across train/validation JSONs
and were not treated as global identities.

Ground truth was joined only after the manifest was frozen. The 49 unique
reference frames contain 85 unique person annotations: 12 at visibility 20,
15 at 40, 22 at 70, and 36 at 100. Twenty-three reference images contain more
than one person. Each reference frame and annotation is evaluated once with
one-to-one matching; the same reference person is therefore not counted twice.
ForestPersons provides no track or physical-person identity across sequences.

## Frozen evaluation protocol

- Model: `facebook/dinov3-vits16-pretrain-lvd1689m` with trained rank-4 q/v
  LoRA adapters and the trained stride-16 FCOS train5000 head.
- Input: short side 672 with the model's existing preprocessing.
- Single view: reference-frame patch features and unchanged FCOS inference.
- Multi-view: warp both source patch grids to reference patch centers, average
  the reference and valid source features equally, L2-normalize, then apply the
  unchanged FCOS head.
- Confidence: fixed at 0.55; NMS IoU 0.50; evaluation IoU 0.50.
- COCO metrics use predictions retained from confidence 0.001 and maxDets 100.
- No threshold or fusion parameter was selected on these 49 groups.

## Results

| Metric | Single reference | Aligned three-view mean | Change |
|---|---:|---:|---:|
| TP / FP / FN at confidence 0.55 | 74 / 12 / 11 | 50 / 21 / 35 | -24 TP, +9 FP |
| Recall | 0.8706 | 0.5882 | -0.2824 |
| Precision | 0.8605 | 0.7042 | -0.1562 |
| FP/image | 0.2449 | 0.4286 | +0.1837 |
| R20 | 0.6667 | 0.6667 | 0.0000 |
| R40 | 0.8000 | 0.6000 | -0.2000 |
| R70 | 0.9091 | 0.5909 | -0.3182 |
| R100 | 0.9444 | 0.5556 | -0.3889 |
| AP | 0.4142 | 0.2291 | -0.1851 |
| AP50 | 0.8778 | 0.6124 | -0.2654 |
| AP75 | 0.3099 | 0.0628 | -0.2471 |
| AR@100 | 0.5094 | 0.3600 | -0.1494 |

Fusion rescued two annotations, one visibility-20 and one visibility-70, but
lost 26 existing single-view detections: one at visibility 20, three at 40,
eight at 70, and fourteen at 100. The paired net change is therefore -24.

The paired sequence-cluster bootstrap recall difference is -0.2824 with a 95%
percentile interval of [-0.4142, -0.1591] over 10,000 replicates. R20's
difference is 0.0000 with a wide [-0.25, 0.25] interval because it contains
only 12 people. Results from the 43 sequences not used by the previous pilot
show the same direction: single recall 0.8861 versus fused recall 0.5949.

Runtime was 0.461 seconds per reference for single-view inference and 1.513
seconds per group for the complete three-view pipeline, approximately 3.28
times slower. Peak observed MPS driver allocation was 2,424,307,712 bytes
(about 2.26 GiB).

## Why fusion failed

Passing a global background homography does not guarantee target-level
correspondence. A post-selection diagnostic projected each source frame's
annotated boxes through the already frozen image homographies and matched them
spatially to reference annotations. This diagnostic never changed membership
or metrics and cannot prove identity because ForestPersons has no track IDs.

- Only 23/85 reference people have a projected GT IoU >=0.50 in both sources.
- 51/85 have such correspondence in at least one source.
- 34/85 have no source correspondence at IoU 0.50.
- Median minimum source-GT IoU is 0.225; median maximum is 0.569.

For the 34 people with no source correspondence, single-view recall is 0.882
and fused recall is 0.235, a net loss of 22 detections. In contrast, the net
loss is only one detection among the 23 people aligned in both sources. Visual
inspection confirms that people move, change pose, or occupy different depths
while trees and terrain remain well registered. Equal feature averaging thus
mixes incompatible person locations and sometimes introduces ghosted target
features. This is the primary measured bottleneck.

## Conclusion

The initial eight-group gain does not generalize to registration-only group
selection. On this larger benchmark, naive global-homography feature averaging
substantially reduces recall, AP, AP75, and AR while increasing false positives.
It should not be trained or scaled in its current form.

DINOv3 multi-view research remains plausible, but it requires geometry that is
valid at the target depth or a fusion method that can learn/estimate local
correspondence and reject incompatible views. ForestPersons does not provide
camera poses, depth, calibration, or track identities needed to establish that
reliably.

## Exact next data requirement

The next defensible experiment is an audit of the official AOS release before
any new fusion implementation:

1. `RAW_lowresRGB.zip` (4,516,263,778 bytes): rectified/downscaled RGB flight
   frames plus the RAW package's KML flight files and OpenCV pinhole
   rectification routine.
2. `YOLO.zip` (7,955,819,284 bytes): prepared train/validation/test labels,
   single images, and AOS integral images.
3. A verified mapping among raw frame IDs, KML camera positions, rectified
   images, and labeled person boxes. If that mapping is absent, request it from
   the authors rather than inferring correspondences.

The dataset is CC BY 4.0. No AOS archive was downloaded in this task. Official
record: <https://zenodo.org/records/4024677>.

## Artifacts

- Preparation manifest: `outputs/multiview/forestpersons_val_benchmark50_seed20261016/preparation/manifest.json`
- Accepted/rejected registration tables: `outputs/multiview/forestpersons_val_benchmark50_seed20261016/preparation/*.csv`
- Evaluation summary and predictions: `outputs/multiview/forestpersons_val_benchmark49_seed20261016/evaluation/`
- Correspondence audit: `outputs/multiview/forestpersons_val_benchmark49_seed20261016/correspondence_analysis/`
- Comparison figures: `outputs/multiview/forestpersons_val_benchmark49_seed20261016/figures/`
- Visual examples: `outputs/multiview/forestpersons_val_benchmark49_seed20261016/evaluation/visual_examples/`

## Validation

- 143 maintained tests passed in 4.65 seconds.
- Eight focused multi-view geometry/benchmark tests passed.
- All new preparation, evaluation, correspondence-analysis, and reporting
  scripts compiled in their intended DINOv3 or OpenCV environments.
- The 49-row experiment registry parsed with nine fields per row.
- Artifact checks confirmed 49 groups, 147 unique views, 85 per-GT diagnostic
  rows, 12 combined visual examples, 2 rescues, and 26 losses.

