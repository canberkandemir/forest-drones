# Official AOS Dataset Audit for Geometry-Aware DINOv3 Fusion

## Scope and decision

This is a read-only feasibility audit. No AOS archive, checkpoint, or training
data was downloaded, and no model was run. Archive inventories were inspected
through Zenodo metadata and ZIP central-directory byte ranges. A few individual
files (under 1 MiB total) were read to verify dimensions and metadata schemas;
they were kept only in `/tmp`.

**Decision:** the official AOS release is the correct next source for calibrated
multi-view work, but it is not yet a plug-and-play paired RGB detection
benchmark. The geometry is strong; the public annotation linkage is the main
verification gate. Download only the low-resolution raw package and the RGB /
thermal integral-image supplement first. Before implementing fusion, verify
that a released integral-image coordinate frame can be reproduced from the raw
RGB poses and that person labels can be mapped to a raw RGB reference view. If
that mapping cannot be established, request the original 3D contours and the
integral-generation configuration from the authors.

Primary sources:

- [Search and Rescue with Airborne Optical Sectioning paper](https://arxiv.org/html/2009.08835)
- [Official raw and prepared data release, DOI 10.5281/zenodo.4024677](https://zenodo.org/records/4024677)
- [Official RGB and thermal integral-image supplement, DOI 10.5281/zenodo.6382373](https://zenodo.org/records/6382373)
- [Official JKU-ICG/AOS repository](https://github.com/JKU-ICG/AOS)
- [Current JKU repository license](https://github.com/JKU-ICG/AOS/blob/stable_release/LICENSE.txt)

## 1. Official data inventory

### 1.1 Zenodo record 4024677

| File | Exact download | MD5 | Verified role |
|---|---:|---|---|
| `RAW_lowresRGB.zip` | 4,516,263,778 B (4.206 GiB) | `f66ebeb35c324fa51bf8fed58a8f5f77` | Rectified/downscaled RGB, thermal frames, poses, COLMAP metadata, KML, calibration and rectification files |
| `RAW_highresRGB.zip` | 47,661,450,106 B (44.388 GiB) | `2a66b011665e7fbf0dd8f25f1068eaed` | Original high-resolution RGB; needed only to rerun high-resolution pose estimation |
| `YOLO.zip` | 7,955,819,284 B (7.409 GiB) | `f803a9af420f54944e18ca6a4ef5917c` | Paper-era prepared thermal single-image and AOS data, transferred labels, Darknet weights/results/scripts |

The low-resolution archive contains **19,063 entries** and expands to
10,292,771,399 B (9.586 GiB). Its directly verified inventory is:

- 18 flights/scenes: `F0`-`F11` and `O1`-`O6`;
- 6,271 rectified RGB JPEGs and 12,542 thermal TIFFs (original and
  rectified/mean-adjusted variants);
- `RGB_poses.json` and `thermal_poses.json` for every flight, with filename-keyed
  3x4 matrices;
- COLMAP `cameras.txt`, `images.txt`, `points3D.txt`, and `project.ini` per
  flight;
- 18 per-flight KML files plus one aggregate KML;
- OpenCV pinhole calibration files for the Sony A6000 and FLIR cameras and the
  rectification source/routine.

A sampled released RGB image is 512x512, as are sampled thermal frames. A
sampled F0 COLMAP camera is `SIMPLE_RADIAL`, 2048x2048, with focal length
2159.77 px. The 512-pixel released image therefore requires explicitly verified
intrinsic scaling; the 2048-pixel COLMAP calibration must not be used unchanged.

The archive has 6,271 RGB frames, but per-scene counts do not always equal the
nominal counts in the paper (for example, F1 contains 207 released RGB frames,
whereas the paper describes the aborted flight as 153 images). This may reflect
capture filtering or preparation differences and must be reconciled after
extraction rather than assumed away.

### 1.2 Zenodo record 6382373

`Integral Images.zip` is 5,034,548,706 B (4.689 GiB), MD5
`bbd996f24bb3fbdcff36e5b39a9add0a`, and expands to 5,063,677,938 B
(4.716 GiB). The archive contains **38,496 entries**:

- 9,612 RGB AOS integral PNGs with matching YOLO text labels;
- 9,612 corresponding thermal AOS integral PNGs with matching labels;
- all official `TRAIN`, `VALID`, and `TEST` scene folders;
- images and labels at 512x512.

The same label file sampled from corresponding RGB and thermal integral images
was byte-equivalent. The filename encodes scene, rotation, focal-plane height,
x/y focal-plane tilt, and optional AHE. These 9,612 images per modality are
augmented products, not independent flights or independent people. Most scenes
have 540 products; F0 and F8 have 486.

### 1.3 Paper split and scene ground truth

The paper reports 18 flights at six sites over ten days. Twelve are forest
flights and six are open meadow flights. Cameras were synchronously triggered
at roughly 30-35 m AGL: thermal at 640x512 and RGB at 6000x4000, with about 360
pairs during a typical ten-minute flight. RGB was primarily used for COLMAP
pose estimation; the published detection result used thermal AOS integral
images.

| Split | Scenes | Scene-level people |
|---|---|---:|
| Train | O1=10, O2=10, O3=6, O4=6, O5=5, F8=0, F10=0 | 37 |
| Validation | O6=5 | 5 |
| Test | F0=3, F1=10, F2=10, F3=6, F4=6, F5=10, F6=10, F7=2, F9=0, F11=0 | 57 |

The test split is the scientifically relevant forest evaluation split. F9 and
F11 are explicit empty forest scenes and are important for false-positive
analysis. The published AOS result (AP 92.2%, precision 96.4%, recall 93%,
53/57 TP and 2 FP) used an IoU threshold of **0.25**, not our normal IoU 0.50
protocol, and must remain separately labeled.

## 2. Geometry, calibration and rectification

The geometry release is much stronger than ForestPersons:

1. High-resolution RGB frames were processed with COLMAP to estimate camera
   poses.
2. Thermal poses were derived using the fixed RGB/thermal rig relationship and
   calibration.
3. The low-resolution release supplies per-frame pose matrices for both
   modalities, COLMAP reconstruction files, camera calibration, rectified
   frames and KML tracks.
4. KML supplies flight trajectory context, but KML alone is not accurate enough
   for feature/pixel warping; the COLMAP/pose matrices and calibrated intrinsics
   are the required inputs.

The matrix convention still needs a numerical verification. The current JKU
`LFR_SIMPLE` loader transposes each released 3x4 matrix after extending it to
4x4 before adding a view. A reprojection unit test must establish world-to-camera
versus camera-to-world convention, handedness, pixel-center convention and the
intrinsic scale at 512x512.

Rectification uses OpenCV's pinhole model. Both Sony and FLIR calibration YAML
files are present. The release does not expose an obvious per-frame RGB/thermal
pair table; filenames use different naming schemes (`DSC...JPG` versus thermal
timestamps). Synchronization and any dropped frames therefore have to be
verified using pose order/timestamps, not array index alone.

## 3. Person annotations and coordinate association

The paper states that people were manually outlined by polygonal contours in
integral images. Because camera poses and focus were known, those contours were
treated as 3D and projected to other focus settings and single images.

What the public archives expose is more limited:

- the integral supplement exposes derived 2D YOLO boxes for RGB and thermal
  integral images;
- `YOLO.zip` exposes prepared thermal single-image PNGs and transferred YOLO
  labels under `data/SARAOS/img/...`;
- `RAW_lowresRGB.zip` exposes raw/rectified RGB and thermal frames plus poses,
  but no standalone original 3D polygon file was found in its archive index;
- no prepared, directly labeled single-view RGB set was found in the public
  archive inventories.

Therefore, confirmed scene-level person counts and integral-image boxes are
available, and thermal single-frame labels are available in the optional
legacy package. Accurate raw **RGB** frame boxes are not yet directly verified.
Deriving them from a 2D integral box would require the exact virtual camera,
focal plane/terrain surface and pose convention; a box-corner projection is
only an approximation to the unreleased 3D contour. It must not be silently
treated as ground truth.

## 4. How official integral images are generated

The paper describes each scene as an unstructured 4D light field. Camera rays
are projected through a synthetic aperture onto a chosen ground/focal plane;
rays arriving at the same output location are averaged. Correct pose and focus
align ground targets while foreground vegetation is spread/blurred across
views. The paper reports roughly 94 million thermal rays per typical scene.

The 2020 Zenodo README explicitly says the software used to compute the released
integral images is not included in that package and that further processing
code is available from the corresponding author on reasonable request. The
current JKU repository now provides `LFR_SIMPLE`, a C++/Cython/OpenGL renderer,
but it is not demonstrated to be the exact 2020 release pipeline. Its sample:

- loads an OBJ digital elevation model;
- loads filename-keyed pose matrices and images;
- renders a 512x512 virtual view (sample FOV 50.815 degrees);
- returns a four-channel accumulator whose last channel is view coverage.

`LFR_SIMPLE` targets Python 3.7-era packages, Cython 0.29, OpenCV 4.5, GLFW,
Assimp and OpenGL. The provided Unix build references Linux-style Assimp
libraries. It may be portable to Apple Silicon, but macOS compilation and
headless OpenGL are **unverified**. It does not use CUDA or MPS.

## 5. Can the three methods be compared fairly?

### What is feasible

The same official test scenes can support:

- a raw single-view RGB DINOv3 pass;
- a geometry-warped multi-view RGB DINOv3 feature fusion pass;
- DINOv3/FCOS applied to official precomputed RGB AOS integral images.

The shared DINOv3 ViT-S/16 + rank-4 q/v LoRA encoder and trained FCOS head can
remain frozen. All methods can use 512x512 imagery and the same confidence/NMS
settings. Feature extraction can run sequentially on MPS and feature alignment
on CPU, fitting within 16 GB.

### What prevents an immediate fair detector benchmark

The three outputs do not automatically share a coordinate frame:

- raw frames use each physical camera pose;
- geometry-aware fusion needs a declared virtual/reference camera and focal
  surface;
- official AOS labels are in the released 512x512 integral-image frame.

A fair IoU comparison requires either (a) the original 3D contours projected
into every method's frame, or (b) reproduction of the official virtual frame so
single- and multi-view predictions can be mapped to the released integral
labels. Neither mapping has yet been verified from public metadata alone.

The official RGB integral images make method C possible, but they do not by
themselves provide raw-frame RGB boxes. Repeated augmented integral images also
must not be counted as independent test samples. The correct statistical unit
is scene/person, with ten official test scenes and only eight non-empty scenes.

## 6. Proposed DINOv3 implementation

### Data model

Create one `AOSScene` record per flight containing:

- split, scene ID and forest/open condition;
- ordered rectified RGB filenames;
- 4x4 poses with explicit convention;
- 512x512 camera intrinsics after verified scaling;
- KML path and optional COLMAP reconstruction references;
- official RGB integral image/label variants;
- an optional mapping to thermal frames and prepared single-image labels.

Every scene loader should validate filename identity, pose count, image size,
matrix finiteness, rotation orthogonality and projected coverage.

### Geometry-aware feature fusion

Use a shared, frozen DINOv3 ViT-S/16 + existing LoRA encoder for all views.
For a small pilot, choose a fixed, metadata-only set of views distributed over
the synthetic aperture. Encode views sequentially and cache patch features.

For each source patch, cast a ray using the verified intrinsic and pose, intersect
it with the declared ground plane or DEM, and project that 3D point into the
common 512x512 virtual output. Bilinearly splat/sample features and retain a
coverage/validity map. Start with deterministic coverage-normalized averaging
and a reference residual/fallback. Do not use GT or detector success in view
selection or fusion weights.

Feed the resulting feature map to the unchanged stride-16 FCOS head. Because
the existing head was trained on ForestPersons perspective RGB images, both
raw AOS and RGB integral results are zero-shot/domain-shift results. No claim
about fusion should be made until geometry-only reprojection tests pass.

### Predeclared pilot scenes and views

After the mapping gate passes, use four official test scenes for the first
resource-bounded pilot:

- F0: conifer, 3 people;
- F3: mixed forest, 6 people;
- F5: conifer, 10 people;
- F9: broadleaf, 0 people.

Use nine views per scene selected from camera positions to cover the aperture,
not selected by annotations or detections. Exclude F7 initially because its
31-frame circular 5 m aperture is a different acquisition regime. Report scene
and person counts rather than treating augmented integral variants as
independent samples.

Compare:

1. **A: single view**, the aperture-central raw RGB frame;
2. **B: geometry-aware feature fusion**, nine raw RGB views in the common
   virtual frame;
3. **C: official RGB AOS integral**, one predeclared unrotated, untilted,
   no-AHE variant per scene.

Use IoU 0.50 for our detector metrics only after frame/label validity is proven.
Report the paper's IoU 0.25 result separately. Also report per-person scene
recall, FP per scene, coverage, reprojection error, latency and peak memory.

## 7. Exact staged download plan

No official scene-selective download is provided; the useful files are
monolithic ZIPs. HTTP range extraction is technically possible but is brittle
and should not be the reproducible project path.

### Stage 1: recommended verification package

Download only:

1. `RAW_lowresRGB.zip` from Zenodo record 4024677;
2. `Integral Images.zip` from Zenodo record 6382373.

Exact totals:

- transfer: **9,550,812,484 B = 8.895 GiB**;
- extracted content: **15,356,449,337 B = 14.302 GiB**;
- archive plus extracted peak: **23.197 GiB**;
- recommended free space before starting: **at least 30 GiB**.

The current disk check found approximately **131 GiB available**, so Stage 1
fits comfortably. Use resumable downloads, then verify the official MD5 hashes
before extraction. Proposed paths:

```text
data/aos/downloads/RAW_lowresRGB.zip
data/aos/downloads/Integral Images.zip
data/aos/raw_lowres/
data/aos/integral_images/
```

After extraction, run a metadata-only gate before model inference:

1. confirm all 18 scene directories and expected pose/image counts;
2. verify 512x512 intrinsic scaling and matrix convention by reprojection;
3. match raw RGB, thermal and prepared filenames/timestamps for F0;
4. reproduce the coordinate frame of one F0 RGB integral variant;
5. verify that its three labels map to the intended people;
6. repeat on empty F9 to test false correspondence.

### Stage 2: optional legacy prepared package

Only if Stage 1 proves that `YOLO.zip` is needed for thermal single-frame
pairing or paper-result reproduction, add it:

- additional transfer: 7.409 GiB;
- additional extracted content: 7.469 GiB;
- all three archives: 16.304 GiB transferred, 21.770 GiB extracted, 38.075
  GiB archive-plus-extracted peak;
- recommended free space: at least 45 GiB.

`YOLO.zip` does contain transferred labels for prepared **thermal** single
images, but it is not currently expected to solve raw RGB label association by
itself. Do not download it merely as a precaution.

### Explicitly defer

- `RAW_highresRGB.zip` (44.388 GiB): unnecessary for the first pilot because
  poses and rectified low-resolution imagery already exist.
- Any training or fine-tuning.
- Full JKU repository build until the license and macOS/OpenGL feasibility are
  resolved.

## 8. Licensing and limitations

The two Zenodo records declare the **data** as CC BY 4.0. Their packaged README
describes the release code as MIT. The current `JKU-ICG/AOS` repository uses a
different license: free for evaluation, education and research, but not for
commercial use without a separate written agreement. Because this work occurs
inside a company internship, use of current repository code requires supervisor
and company confirmation and may require contacting JKU. This report is not a
legal opinion.

Main scientific limitations:

- only 18 flights and ten official test scenes;
- people lie on the ground and mostly remain still;
- the published detector benchmark is thermal, while our proposed DINOv3 study
  is RGB;
- original 3D contour files and exact 2020 integration configuration are not
  visibly exposed as standalone public files;
- augmented integral images are strongly correlated;
- terrain/focal-plane mismatch can dominate feature fusion;
- current JKU renderer compatibility with Apple Silicon is unverified.

## Recommendation

Approve **Stage 1 only**: 8.895 GiB of downloads and a metadata/reprojection
verification pass. Do not train, and do not claim a paired single-view versus
multi-view RGB benchmark until raw-frame annotation association and the common
virtual coordinate frame are demonstrated. If that gate fails, the exact next
action is to request the 3D person contours, RGB/thermal synchronization map,
virtual-camera/focal-plane parameters and integral-generation configuration
from the corresponding author.
