# AOS RGB Integral Zero-Shot Failure Diagnostic

The saved 830 post-NMS predictions were reproduced exactly with the unchanged
ForestPersons DINOv3 ViT-S/16 + LoRA + FCOS checkpoint. No threshold was tuned.

## Verified pipeline

- All ten images are canonical products from the official AOS `TEST` folders.
- Every 512x512 PNG has its same-stem YOLO label; 57 boxes reproduce the
  published test-scene counts.
- Normalized `cx cy w h` labels decode directly to valid 512x512 `xyxy` boxes.
- Images resize uniformly to 672x672 (`scale_x = scale_y = 1.3125`), and FCOS
  boxes are divided by those same scales. No padding or coordinate mismatch was
  found.
- Preprocessing is code-identical to ForestPersons: RGB conversion, bicubic
  patch-grid resize, and the same DINOv3 normalization. The difference is the
  input domain: synthetic-aperture integral composites rather than perspective
  UAV frames.

## Where the 57 GT people are lost

| Last useful stage | GT count |
|---|---:|
| No raw candidate reaches IoU 0.50 (localization failure) | 31 |
| Valid raw candidate removed by top-300 score ranking | 8 |
| Valid top-300 candidate suppressed by NMS | 7 |
| Valid post-NMS candidate below frozen confidence 0.55 | 9 |
| Available at confidence 0.55 | 2 |

All 17,640 raw grid candidates exceed the nominal 0.001 floor, so that floor
removes nothing. Candidate coverage at IoU 0.50 falls from 26/57 raw to 18/57
after top-300 and 11/57 after NMS. Median best IoU falls from 0.483 raw to 0.302
after top-300 and 0.240 after NMS. At IoU 0.25, raw candidates cover 50/57 GT,
showing that many responses are near people but boxes/ranking are unsuitable
for strict localization. Median GT size is only 32.8x32.8 px (minimum 14.4 px).

## Conclusion and next step

The 2/57 result is primarily a domain-shift/localization failure, compounded by
ForestPersons-trained score ranking and NMS; it is not caused by label scaling
or different preprocessing. Lowering confidence alone cannot fix the 31 GT
without a valid IoU-0.50 candidate and would greatly increase false positives.

Next, obtain/verify the exact AOS virtual camera, focal surface/DEM, and 3D
annotation projection. Validate geometry on official train/validation scenes,
then compare a frozen single warped view against geometry-aligned multi-view
DINOv3 feature aggregation in the same integral-image coordinate frame. Keep
the official test scenes untouched until that mapping is fixed.

Artifacts:

- `scripts/diagnose_aos_rgb_integral_zero_shot.py`
- `outputs/aos/rgb_integral_zero_shot_canonical_test_20261019/diagnostics/summary.json`
- `outputs/aos/rgb_integral_zero_shot_canonical_test_20261019/diagnostics/per_gt_diagnostics.csv`

