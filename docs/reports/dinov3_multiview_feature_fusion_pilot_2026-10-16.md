# DINOv3 Multi-View Feature Fusion Pilot

Date: 2026-10-16

## Scope

This work cancels the unfinished custom DETR-head experiment and evaluates a
minimal multi-view alternative: align DINOv3 patch features from multiple UAV
frames, average valid aligned features, and run the unchanged trained FCOS head
on the fused reference grid. No training, checkpoint modification, or dataset
download was performed.

## DETR cleanup

The project directory is not a Git repository, so `git status` and Git diffs
were unavailable. The filesystem was inspected before cleanup. Exactly three
new unfinished files were removed:

- `src/forestpersons/dinov3_detr.py`
- `scripts/train_dinov3_lora_detr.py`
- `scripts/evaluate_dinov3_lora_detr.py`

No DETR process, checkpoint, result directory, or training run existed. The
historical MMDetection DETR baseline preparation and all validated FCOS, LoRA,
YOLO, fusion, ForestPersons, SARD, and Weitefeld artifacts were preserved.

## Dataset feasibility

### Local datasets

| Dataset | Multi-view evidence | Person ground truth | Decision |
|---|---|---|---|
| ForestPersons | Sequence folders and adjacent UAV frames; no camera poses, calibration, homographies, or track IDs | Confirmed per-frame person boxes and visibility metadata | Suitable only for a small registration-qualified exploratory pilot |
| Weitefeld pilot200 | Flight-strip grouping, but no local camera pose/calibration metadata | Class 3 is not confirmed-person ground truth: 0 confirmed, 134 explicit non-person/no-abnormality, 21 uncertain | Unsuitable for person-recall multi-view evaluation |
| SARD mountain200 | Consecutive sequence frames, but no local camera pose/calibration metadata | 305 person boxes | Potential temporal study, but not calibrated multi-view geometry |

The local ForestPersons copies contain 262 train sequence folders and 57
validation sequence folders. Adjacent frames frequently depict the same staged
person and ground region, but temporal proximity alone was not accepted as
alignment evidence.

### Official AOS data

The relevant official AOS person-search release is the data associated with
*Search and rescue with airborne optical sectioning*, not the unrelated local
bird/nesting AOS PDF. The official release documents 18 flights at six sites
over ten days and provides KML flight information plus rectification code. The
Zenodo record contains approximately 47.7 GB high-resolution RGB, 4.5 GB
low-resolution RGB, and an 8.0 GB prepared YOLO package. Data are CC BY 4.0;
the linked code repository states MIT licensing.

Sources:

- [Official AOS dataset on Zenodo](https://zenodo.org/records/4024677)
- [Dataset README](https://zenodo.org/records/4024677/preview/README_v2.md?include_deleted=0)
- [Nature Machine Intelligence paper](https://www.nature.com/articles/s42256-020-00261-3)
- [Official JKU AOS code](https://github.com/JKU-ICG/AOS)

No AOS files were downloaded because even the smallest useful packages are
multi-gigabyte transfers and require approval.

## Pilot construction

Eight three-view groups were selected from the official ForestPersons
validation split. Each group uses one reference frame and two source frames
from the same sequence and pose. The reference images are disjoint from the
2,000 image IDs already used by the val600, calibration200, and holdout1200
manifests.

Image-only alignment uses ORB descriptors, a 0.75 ratio test, and RANSAC
homography estimation. A source/reference pair was retained only when it had:

- at least 30 RANSAC inliers;
- inlier ratio at least 0.50;
- median registration-scale reprojection error at most 2 pixels; and
- projected source-GT/reference-GT IoU at least 0.50.

Ground truth is used only to qualify that a proposed pair corresponds to the
same annotated target region. It is not used to estimate the homography or at
inference. This qualification makes the pilot selection-biased and unsuitable
as an independent benchmark.

Pilot composition: 8 unique sequences, 24 total views, 8 reference people,
three at visibility 20 and five at visibility 40. In group 08, the 40%-visible
reference has source views at visibility 100 and 70, providing complementary
visual evidence.

## Fusion method

The existing `facebook/dinov3-vits16-pretrain-lvd1689m` encoder, trained rank-4
q/v LoRA adapters, and trained stride-16 FCOS head are shared across every
view. For each group:

1. Extract the reference and source patch grids at short side 672.
2. Convert the original-image homography to independently scaled resized-image
   coordinates.
3. Warp source patch features to reference patch centers with bilinear
   sampling.
4. Average the reference feature and all valid aligned source features with
   equal weights.
5. L2-normalize the fused channels and run the unchanged FCOS decoder and NMS.

This is feature-level fusion, not prediction union. Evaluation used the same
fixed confidence 0.55, NMS IoU 0.50, and detection IoU 0.50 for both arms. No
threshold tuning was performed.

## Results

These numbers describe only eight GT-qualified examples.

| Metric | Single reference | Aligned 3-view fusion |
|---|---:|---:|
| TP / FP / FN at confidence 0.55 | 7 / 3 / 1 | 8 / 0 / 0 |
| Recall | 0.875 | 1.000 |
| Precision | 0.700 | 1.000 |
| FP/image | 0.375 | 0.000 |
| Recall, visibility 20 (3 GT) | 1.000 | 1.000 |
| Recall, visibility 40 (5 GT) | 0.800 | 1.000 |
| COCO AP | 0.3703 | 0.4215 |
| AP50 | 0.8500 | 1.0000 |
| AP75 | 0.2145 | 0.2145 |
| AR@100 | 0.4375 | 0.5000 |

Fusion rescued reference image ID 17354 (group 08, visibility 40) and lost no
fixed-threshold detections. Its best saved-candidate IoU increased from 0.5525
to 0.7464. Best-candidate IoU improved in five of eight groups and decreased in
three; unchanged AP75 confirms that alignment/fusion did not consistently
improve high-IoU localization.

Runtime on Apple M4 MPS was 0.471 seconds per group for single-reference
inference and 1.524 seconds per group for the complete three-view pipeline,
approximately 3.24 times slower. Peak observed MPS driver allocation was
1,748,877,312 bytes (about 1.63 GiB).

## Interpretation

Confirmed:

- The shared DINOv3 encoder, homography-aware feature warp, valid-region
  averaging, and unchanged FCOS head execute correctly on MPS.
- A complementary source view recovered one fixed-threshold miss in this
  selected pilot.
- Coordinate handling, feature-grid warping, and output evaluation are covered
  by focused tests.

Not established:

- General recall improvement, because the sample has only eight people and was
  selected with GT-based correspondence qualification.
- Robustness to parallax, vegetation motion, large viewpoint changes, or bad
  registration.
- A benefit at visibility 20, because the single-view model already detected
  all three selected examples.
- Transfer to true multi-view AOS flights with known camera geometry.

## Decision

The feature-fusion path is technically viable and shows genuinely
complementary information in group 08, but the current ForestPersons pilot is
too small and selection-biased to justify training. The next defensible action
is to obtain approval for the official AOS low-resolution RGB/flight metadata
and corresponding labeled package, verify frame/pose/label mapping, and then
run a larger inference-only benchmark with geometry supplied by the dataset.
Do not train a fusion module until that audit establishes trustworthy aligned
views and confirmed-person ground truth.

## Reproducibility

- Preparation manifest: `outputs/multiview/forestpersons_val_pilot8_seed20261016/preparation/manifest.json`
- Alignment gallery: `outputs/multiview/forestpersons_val_pilot8_seed20261016/preparation/alignment_gallery/`
- Evaluation summary: `outputs/multiview/forestpersons_val_pilot8_seed20261016/evaluation/summary.json`
- COCO predictions: `outputs/multiview/forestpersons_val_pilot8_seed20261016/evaluation/*_coco_results.json`
- Comparison gallery: `outputs/multiview/forestpersons_val_pilot8_seed20261016/evaluation/visualizations/comparisons/`
- Persistent log: `outputs/multiview/forestpersons_val_pilot8_seed20261016/evaluation_run.log`

## Validation

- `PYTHONPATH=src .venv-dinov3/bin/python -m pytest -q tests`: 140 passed.
- Focused multi-view suite: 5 passed.
- New modules and scripts compiled successfully in their DINOv3/OpenCV
  environments.
- The experiment-registry CSV parsed successfully with 48 rows and nine fields
  per row.
- An unrestricted repository-root pytest invocation also collected vendored
  DEIMv2 tests and legacy root smoke scripts; collection stopped because the
  DINOv3 environment intentionally lacks their optional `tensorboard` and
  `ultralytics` dependencies. This does not affect the maintained `tests/`
  suite.

