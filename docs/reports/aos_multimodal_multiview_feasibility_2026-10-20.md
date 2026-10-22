# AOS Multimodal and Multi-View Feasibility

Date: 2026-10-20

## Scope

This bounded audit reuses the downloaded official AOS Stage 1 data and prior
geometry code. It performs no detection, training, feature fusion, or model
changes. Three scenes were selected before inspection of model outputs:

| Scene | Split | Setting | Raw RGB/thermal pose pairs | Canonical integral GT people |
|---|---|---|---:|---:|
| F0 | Test | Forest | 411 | 3 |
| F8 | Train | Forest | 358 | 0 |
| O6 | Validation | Open/meadow | 387 | 5 |

## RGB-thermal correspondence

Pairing was established from official pose metadata, not filenames. In each
scene, the RGB and thermal pose arrays have equal length. For same-index pairs,
`T_thermal @ inverse(T_RGB)` is constant to at most `6.0e-15`; pairing thermal
slot `i+1` with RGB slot `i` increases transform deviation to 2.03-4.94. The
unrelated filename schemes are therefore only labels attached after the pose
slot association.

Across F0, F8, and O6, the released model places paired camera centers together
to numerical precision and applies a constant 0.8896-degree relative rotation.
The released rectifier maps both cameras to 512 x 512 pinhole images with
`fx=fy=538.947368`, `cx=cy=256`. Consequently, the **released pose model**
supports a depth-independent rotation homography between modalities.

This does not prove exact observed-pixel registration. A physical sensor
baseline, exposure timing differences, rolling motion, and modality-dependent
edges are not quantified by the released co-located pose model. The generated
edge overlays show broad structural correspondence but also residual red/green
edge disagreement; they are diagnostics, not registration error measurements.

## Multi-view overlap and altitude

All six source/reference RGB pairs passed the existing ORB/RANSAC reliability
rule without using annotations:

| Scene | Source -> reference | Baseline (COLMAP units) | Center-z difference | Inliers | Median error |
|---|---|---:|---:|---:|---:|
| F0 | 180 -> 205 | 1.267 | +0.230 | 11 | 0.83 px |
| F0 | 235 -> 205 | 0.624 | +0.194 | 53 | 1.41 px |
| F8 | 59 -> 60 | 0.286 | -0.042 | 124 | 1.37 px |
| F8 | 61 -> 60 | 0.272 | +0.058 | 156 | 1.09 px |
| O6 | 189 -> 190 | 0.277 | -0.008 | 2,362 | 0.80 px |
| O6 | 191 -> 190 | 0.241 | +0.023 | 2,451 | 0.80 px |

KML uses `relativeToGround` altitude in meters. Its scene-wide median/range is
25.32/60.77 m for F0, 27.48/44.33 m for F8, and 25.99/43.99 m for O6. These
ranges include takeoff/landing. Pose JSON contains no timestamps, so KML
samples cannot be assigned safely to the selected image frames; selected-view
height differences are therefore reported only in COLMAP coordinates.

## Annotation validity

- Raw rectified RGB and thermal frames have no direct person boxes in Stage 1.
- Official RGB and thermal integral products have identical stems and
  byte-identical YOLO labels. F0, F8, and O6 canonical products contain 3, 0,
  and 5 people, matching the official scene counts.
- Raw-view or custom multi-view person recall remains invalid without the
  missing 3D contours or exact raw-to-virtual annotation projection.
- Paired canonical integral products can support a valid RGB-versus-thermal or
  same-frame multimodal experiment because both modalities share the official
  labeled virtual frame.

## Minimal next experiment

Run an inference-only, frozen-feature pilot on the canonical official integral
pairs for F0, F8, and O6: extract DINOv3 ViT-S/16 features independently from
RGB and channel-replicated thermal, measure patch cosine correspondence in the
shared virtual frame, and compare correct RGB/thermal pairs against deliberately
shifted scene/frame controls. Only if correspondence is measurable should a
predeclared coverage-preserving feature average be evaluated with the unchanged
FCOS head and official integral boxes. This tests multimodal feature utility
without claiming raw-view geometry-aware recall or requiring training.

## Artifacts

- `scripts/audit_aos_multimodal_feasibility.py`
- `outputs/aos/multimodal_feasibility_20261020_rerun1/summary.json`
- Six paired-view/integral and rotation-homography diagnostic figures in that
  output directory.

The earlier output directory without `_rerun1` has an obsolete figure caption
about translation/depth parallax. Its numerical data are unchanged; `rerun1`
is canonical.
