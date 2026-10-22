# AOS Thermal vs Mean-Fusion Detection Analysis

Date: 2026-10-20  
Scope: offline analysis of saved canonical TEST predictions; no inference or training

## Result

At the validation-locked score 0.70, thermal detected 26/57 people with two
false positives (recall 45.61%, precision 92.86%, 0.20 FP/image). Mean fusion
detected 5/57 with two false positives (recall 8.77%, precision 71.43%, 0.20
FP/image). Fusion lost 21 thermal true positives and gained 0.
AP50 fell from 0.7390 to 0.3603;
AR@100 fell from 0.3632 to 0.2491.

| Scene | GT | Thermal TP | Fusion TP | Lost | Gained | Thermal FP | Fusion FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 3 | 1 | 0 | 1 | 0 | 0 | 0 |
| F1 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| F2 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| F3 | 6 | 5 | 0 | 5 | 0 | 1 | 0 |
| F4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| F5 | 10 | 9 | 2 | 7 | 0 | 0 | 0 |
| F6 | 10 | 9 | 3 | 6 | 0 | 1 | 2 |
| F7 | 2 | 2 | 0 | 2 | 0 | 0 | 0 |
| F9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Lost detections

The 21 thermal-only GT instances divide into: no nearby post nms candidate: 1, post nms localization below iou50: 2, valid box below confidence: 18.
For each GT, `per_gt_diagnostics.csv` records the best available saved candidate
IoU, score, normalized center error, width/height ratios, and area ratio. Across
thermal-detected GT, the mean fusion-minus-thermal best-candidate score change
was -0.282. The mean best
fusion IoU for thermal-only GT was 0.620.

The dominant mechanism is interpreted from the saved post-NMS candidates only:
valid fusion boxes below 0.70 indicate confidence suppression; IoU 0.25-0.50
indicates localization failure; IoU below 0.25 indicates no useful nearby saved
candidate. Box-scale and center diagnostics distinguish displaced from badly
sized candidates but cannot recover candidates removed before serialization.

## Limitations

Each prediction file contains exactly 100 post-NMS detections per image with a
0.001 score floor. Therefore this analysis cannot attribute losses to pre-NMS
thresholding, top-k ordering, or NMS suppression. Thresholds were selected on
VALID/O6, not TEST. The ten canonical test products contain 57 people across
eight positive scenes and two empty scenes. The detector heads were trained on
augmented renderings of only 37 unique training people, so conclusions remain
pilot-scale.

## Artifacts

- `per_scene_comparison.csv`
- `per_gt_diagnostics.csv`
- `summary.json`
- `overlays/`: all ten canonical scenes; magenta marks thermal TP lost by fusion
