# DINOv3 MVR Hybrid ANN2SNN Pilot Audit

## Finding

The original 100-image pilot used the first 100 entries of the val600 manifest.
The manifest is ordered in 150-image target-visibility blocks, so all 100 images
were selected from the visibility-20 block. Direct annotation inspection found:

- 100 images with visibility-20 presence;
- 116 visibility-20 GT annotations;
- zero visibility-40, visibility-70, or visibility-100 annotations.

Therefore overall recall and R20 are necessarily identical within every model
mode. The modes themselves did not have identical recall: ANN 0.6034, T4
0.4569, T8 0.5862, and T16 0.6207. This is a sampling bias, not a
visibility-metric bug.

## AP50 discrepancy

The pilot ANN AP50 was 0.6151, compared with MVR AP50 0.8162 on all val600.
The 100-image subset contains only the hardest visibility-20 stratum, whereas
val600 contains 174/170/193/212 GT at visibility 20/40/70/100. The pilot ANN
prediction JSON exactly matches all 9,213 corresponding predictions in the
previous full-val600 MVR prediction dump. Thus the AP50 difference is caused by
the biased subset and changed evaluation population, not changed inference,
coordinate scaling, or checkpoint behavior.

## Correction

The ANN2SNN evaluator now supports `--eval-sampling stratified`. A 100-image
pilot deterministically selects 25 target images from each visibility block;
`--eval-images 600` selects the complete manifest. Prefix sampling remains the
default so the original result is reproducible. The evaluator also reports
R20/R40/R70/R100 and can reuse the exact saved train-only calibration thresholds
through `--reuse-calibration-metrics`, avoiding calibration recomputation.

No checkpoint, cache, or original pilot output was modified.
