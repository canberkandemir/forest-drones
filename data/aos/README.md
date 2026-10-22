# AOS dataset

This directory contains Stage 1 of the official Airborne Optical Sectioning
(AOS) release. The files are dataset inputs, not project-generated results.

## Downloaded archives

| Archive | Zenodo record | Bytes | MD5 |
|---|---|---:|---|
| `downloads/RAW_lowresRGB.zip` | 4024677 | 4,516,263,778 | `f66ebeb35c324fa51bf8fed58a8f5f77` |
| `downloads/Integral Images.zip` | 6382373 | 5,034,548,706 | `bbd996f24bb3fbdcff36e5b39a9add0a` |

Both archives passed `unzip -tq` and are intentionally retained after verified
extraction. `RAW_highresRGB.zip` and `YOLO.zip` were not downloaded.

## Layout

- `raw_lowres/RAW_lowresRGB/`: rectified 512x512 RGB frames, thermal frames,
  per-frame poses, COLMAP reconstructions, KML tracks, and rectification files.
- `integral_images/Integral Images/data/`: official RGB and thermal integral
  images and their YOLO-format labels.
- `downloads/`: immutable original archives.

The Zenodo data are CC BY 4.0. The current JKU-ICG/AOS source repository has a
separate non-commercial research/evaluation license; repository code must not
be assumed to share the dataset license.

Raw RGB frames do not contain direct person labels in these Stage 1 archives.
Do not evaluate raw-RGB person recall until the original 3D contours or the
exact virtual-camera/projection metadata have been obtained and verified.

Verification artifacts are in `outputs/aos/stage1_verification/`; the full
technical record is `docs/reports/aos_stage1_download_geometry_annotation_verification_2026-10-19.md`.

