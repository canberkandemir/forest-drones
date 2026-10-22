# DINOv3 MVR Hybrid ANN2SNN — ForestPersons val600

## Protocol

- Checkpoint: trained frozen-DINOv3 MVR + FCOS train5000 model.
- Evaluation: all 600 images and 749 GT persons in the fixed val600 manifest.
- Visibility GT: 174/170/193/212 at 20/40/70/100 percent.
- Calibration: unchanged 99.9th-percentile thresholds from 100 train5000
  feature-cache samples; no validation calibration.
- Fixed confidence 0.63, IoU 0.50, NMS 0.50, top-300 candidates.
- Hybrid conversion: only four MVR projection ReLUs and two context ReLUs use
  deterministic soft-reset IF rate coding. The DINOv3 encoder and FCOS detector
  remain analog.

## Results

| Mode | AP50 | Recall | Precision | FP/image | R20 | R40 | R70 | R100 | ms/image | Activity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN | 0.8162 | 0.8077 | 0.7601 | 0.3183 | 0.5862 | 0.7765 | 0.9016 | 0.9292 | 13.0 | n/a |
| SNN T=4 | 0.7622 | 0.7156 | 0.7679 | 0.2700 | 0.4425 | 0.6765 | 0.8135 | 0.8821 | 23.2 | 0.0246 |
| SNN T=8 | 0.8002 | 0.7891 | 0.7685 | 0.2967 | 0.5632 | 0.7412 | 0.8756 | 0.9340 | 31.1 | 0.0320 |
| SNN T=16 | 0.8083 | 0.7997 | 0.7729 | 0.2933 | 0.5862 | 0.7529 | 0.8808 | 0.9387 | 49.1 | 0.0356 |

T=16 recovered the ANN's visibility-20 recall and retained 99.0% of ANN AP50,
but remained slightly below ANN overall recall and was 3.77x slower in the MPS
software simulation. Lower FP/image partly reflects fewer emitted detections,
not an independently calibrated operating advantage.

## Interpretation

Increasing timesteps improves the rate approximation. The result demonstrates
conversion feasibility for the MVR ReLU boundary, not end-to-end SNN deployment.
Dense MPS operations remain, and activity ratios alone cannot support energy or
hardware-efficiency claims.

