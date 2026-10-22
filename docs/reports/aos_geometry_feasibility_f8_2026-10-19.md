# AOS F8 Geometry Feasibility Check

Date: 2026-10-19  
Scope: one official forest training scene (`F8`), no test-set use

## Verified geometry

- Rectified RGB images are `512 x 512`. The bundled rectifier uses
  `fx = fy = 512 / 0.95 = 538.947368`, `cx = cy = 256`, equivalent to a
  `50.815436` degree field of view.
- `RGB_poses.json` contains COLMAP world-to-camera transforms `[R|t]`.
  For frames `DSC03455.JPG`, `DSC03456.JPG`, and `DSC03457.JPG`, these agree
  with `images.txt` to within `3.14e-9`. Camera centers are `C = -R^T t`.
- The released 3x4 matrices must be extended to 4x4 and transposed when passed
  to the official GLM/OpenGL renderer, as done by `pyaos.pyx`.

## Three-view validation

The neighboring views 59/60/61 were aligned to reference frame 60 using
image-only ORB/RANSAC. Frame 59 produced 124 inliers with 1.37 px median
reprojection error; frame 61 produced 156 inliers with 1.09 px error. This
confirms substantial static-scene overlap and consistency of the released
poses, but it is not the official terrain-aware virtual-camera projection.

Visualization:
`outputs/aos/geometry_feasibility_f8_20261019/f8_three_view_alignment.png`

## Exact reproduction blocker

The official AOS renderer requires a scene-specific DEM/OBJ focal surface via
`loadDEM`, a virtual-camera pose, and input camera poses. Stage 1 contains no
F8 DEM/OBJ/PLY terrain surface. Exact reproduction also lacks:

1. The F8 terrain mesh and its scale/transform into the COLMAP frame.
2. The exact virtual-camera pose used for each released integral image.
3. The focal-surface height/tilt interpretation encoded by `fpH`, `rx`, and
   `ry`, including any terrain offset.
4. The exact source-view subset/order and integral normalization settings.

Intrinsics, rectification, and per-frame extrinsics are available; they are
not blockers.

## Decision

Exact official virtual-camera reprojection is not currently possible. The
simplest defensible local alternative is reference-view ORB/RANSAC alignment
for exploratory feature fusion, explicitly labeled image-plane rather than
terrain-aware AOS projection. A faithful experiment requires obtaining the F8
DEM/OBJ, its COLMAP transform, and the official virtual-camera configuration
from the dataset authors.

Machine-readable measurements:
`outputs/aos/geometry_feasibility_f8_20261019/summary.json`
