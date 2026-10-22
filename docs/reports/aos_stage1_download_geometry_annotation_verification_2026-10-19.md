# AOS Stage 1 Download, Geometry, and Annotation Verification

## Scope

Stage 1 of the official AOS release was downloaded and inspected locally on
2026-10-18/19. No model was trained or evaluated. No checkpoint was changed.
Only `RAW_lowresRGB.zip` and `Integral Images.zip` were transferred; neither
the high-resolution RGB archive nor the legacy `YOLO.zip` was downloaded.

Official sources:

- Zenodo record 4024677: <https://zenodo.org/records/4024677>
- Zenodo record 6382373: <https://zenodo.org/records/6382373>
- JKU-ICG/AOS: <https://github.com/JKU-ICG/AOS>

## 1. Download and integrity

At least 131 GiB was free before transfer, above the 30 GiB gate. Downloads
used resumable `curl`; one stalled raw transfer resumed from byte
1,065,807,872 rather than restarting. The original archives remain present.

| Archive | Exact bytes | Official MD5 | Local MD5 | ZIP test |
|---|---:|---|---|---|
| `RAW_lowresRGB.zip` | 4,516,263,778 | `f66ebeb35c324fa51bf8fed58a8f5f77` | match | pass |
| `Integral Images.zip` | 5,034,548,706 | `bbd996f24bb3fbdcff36e5b39a9add0a` | match | pass |

Disk use after extraction is 8.9 GiB for retained archives, 9.6 GiB for raw
low-resolution data, and 4.8 GiB for integral images. Approximately 107 GiB
remains free.

## 2. Verified dataset structure

The raw extraction contains 18,947 files in 18 scenes (`F0`-`F11`, `O1`-`O6`):

- 6,271 rectified RGB JPEGs;
- 6,271 original thermal TIFFs and 6,271 rectified/mean-adjusted TIFFs;
- 36 pose JSON files, one RGB and one thermal file per scene;
- 18 COLMAP reconstructions and camera files;
- 18 scene KML tracks plus one aggregate KML;
- Sony A6000 and FLIR calibration/rectification material.

Every RGB filename maps exactly to one RGB pose. Every thermal filename maps
exactly to one thermal pose, and each scene has equal RGB/thermal pose counts.
The per-frame RGB-to-thermal relative pose is fixed: the largest deviation over
all scenes is `5.995e-15`, confirming the released synchronized rig ordering.
Sampled first/middle/last files from every image directory opened successfully.

The integral extraction contains 38,448 files:

- 9,612 RGB PNGs and 9,612 RGB YOLO label files;
- 9,612 thermal PNGs and 9,612 thermal YOLO label files;
- official `TRAIN`, `VALID`, and `TEST` scene folders.

All RGB image/label stems match their thermal counterparts. Corresponding RGB
and thermal labels are byte-identical. Every row has category 0 and normalized
coordinates in `[0, 1]`. Across augmented variants there are 53,298 boxes per
modality. The unaugmented canonical product
`rot000_fpH0.00_rx0.0_ry0.0` reproduces the paper's scene counts for all 18
scenes (37 train, 5 validation, 57 test people).

Machine-readable inventories:

- `outputs/aos/stage1_verification/aos_stage1_verification.json`
- `outputs/aos/stage1_verification/raw_scene_inventory.csv`
- `outputs/aos/stage1_verification/integral_scene_inventory.csv`

## 3. Camera geometry validation

The released 3x4 pose matrices are COLMAP world-to-camera `[R|t]` matrices:
the sampled F0 JSON matrix agrees with the corresponding COLMAP quaternion and
translation. Camera centers are therefore `C = -R^T t`. In F0, 411 RGB and
411 thermal poses are available. The sampled COLMAP RGB camera is
`SIMPLE_RADIAL`, 2048x2048, focal length 2159.77 px, principal point
(1024, 1024), and radial coefficient -0.0073489. Released rectified RGB frames
are 512x512, so intrinsics require a 0.25 scale after accounting for the
rectified-image convention.

A reproducible F0 example used reference frame `DSC00756.JPG` and nearby
source frames `DSC00728.JPG` and `DSC00788.JPG`. Candidates were selected from
pose neighborhoods and ORB/RANSAC checks only; no labels or detections were
used.

| Source | Baseline (COLMAP units) | Axis difference | RANSAC inliers | Inlier ratio | Median error | Warped area ratio |
|---|---:|---:|---:|---:|---:|---:|
| `DSC00728.JPG` | 1.267 | 0.685 deg | 11 | 0.367 | 0.83 px | 0.795 |
| `DSC00788.JPG` | 0.624 | 2.519 deg | 53 | 0.373 | 1.41 px | 0.926 |

Both pairs pass the declared image-plane registration checks. This validates
overlapping visual coverage and demonstrates that the pose/frame association
is usable. It does **not** validate person alignment: ORB features lie mainly
on vegetation/canopy, and a background homography cannot account for varying
terrain, person height, or motion.

Visuals:

- `outputs/aos/stage1_verification/F0_pose_selected_triplet_and_official_integral.png`
- `outputs/aos/stage1_verification/F0_camera_track_triplet.png`
- `outputs/aos/stage1_verification/F0_orb_matches_DSC00728_to_DSC00756.png`
- `outputs/aos/stage1_verification/F0_orb_matches_DSC00788_to_DSC00756.png`
- matching overlay images in the same directory.

The released KML tracks give flight context and altitude, while COLMAP poses
provide the relative camera geometry. Stage 1 does not include the exact DEM,
virtual-camera matrix, focal surface, or original 3D person contours used to
produce the official integral images. Consequently, an exact independent
reconstruction of an official integral image was not possible and was not
approximated as if it were authoritative.

## 4. Annotation linkage audit

| Representation | Annotation status | Valid person recall now? |
|---|---|---|
| Raw rectified RGB frames | No direct raw-RGB labels in Stage 1 | No |
| Custom geometry-aligned features | No verified mapping to official virtual coordinates | No |
| Official RGB integral images | Direct, validated YOLO boxes | Yes |
| Official thermal integral images | Direct, validated YOLO boxes | Yes |

The paper describes original person contours as 3D/projectable annotations,
but the downloaded archives expose derived 2D integral-image boxes rather than
those original contours. Projecting an integral-image rectangle back into a
raw frame would be an unsupported approximation. Raw-frame or custom-fusion
recall must therefore remain blocked until the missing coordinate linkage is
obtained.

## 5. Fair A/B/C experiment design

The scientifically fair comparison is conditional on obtaining the exact
virtual camera/focal surface (or DEM) and original 3D contours from the AOS
authors.

1. Use the official scene split: training scenes only for any later fitting,
   O6 for validation, and forest test scenes for final evaluation. Treat a
   scene/person, not an augmented integral variant, as the independent unit.
2. Use one canonical output frame per scene. Evaluate official test scenes,
   retaining empty F9 and F11 for false-positive measurement.
3. Select source views from poses/coverage only. Begin with nine sequentially
   encoded views per scene so MPS memory remains bounded.
4. Method A: warp one raw RGB view's frozen DINOv3 ViT-S/16 + existing LoRA
   features into the verified virtual frame, then use the unchanged FCOS head.
5. Method B: warp the same features from all selected views, aggregate by
   geometry/coverage with a reference fallback, then use the unchanged head.
6. Method C: run the same frozen encoder/head on the official canonical RGB
   integral image. This is an official-image baseline, not a recreation.
7. Use identical 512x512 output coordinates, confidence/NMS settings, and
   direct official labels. Report IoU 0.50 for project consistency and IoU
   0.25 separately for relation to the AOS paper.
8. Report scene-level paired recall, AP/AR, FP/scene, coverage, latency, and
   peak memory. Label all results zero-shot because the detector was trained
   on ForestPersons perspective RGB.

The existing DINOv3/LoRA/FCOS checkpoint can be reused without modification.
Views should be encoded sequentially on MPS and fused on CPU or MPS; 16 GB is
adequate for a small nine-view pilot. The geometry implementation, not model
memory, is the present blocker.

## 6. Licensing and decision

The Zenodo datasets are CC BY 4.0. The current JKU-ICG/AOS code has a separate
non-commercial research/evaluation license. Because this work occurs in a
company internship context, supervisor/legal confirmation is required before
copying or executing repository code; the present audit uses no repository
implementation.

**Decision:** Stage 1 download and archive/data verification are complete.
Official integral-image evaluation is technically available, but a fair
single-view versus geometry-fusion comparison is not yet defensible. The next
single action is to request from the AOS authors the original 3D person
contours and the exact integral-generation virtual-camera/DEM configuration.
Do not begin DINOv3 multi-view recall evaluation until that coordinate mapping
passes a reprojection test.

