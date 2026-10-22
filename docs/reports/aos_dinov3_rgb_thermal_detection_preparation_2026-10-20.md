# AOS DINOv3 RGB/Thermal Detection Comparison: Preparation

Date: 2026-10-20  
Status: protocol and code prepared; no feature extraction, training, or evaluation run

## Validity audit

The downloaded official integral-image release supports a limited supervised
feasibility comparison. RGB and thermal products have matching scene/stem,
512 x 512 dimensions, and byte-identical one-class YOLO label files. The
official scene split is disjoint:

| Split | Scenes | Unique people in canonical products |
|---|---:|---:|
| TRAIN | F8, F10, O1-O5 (7) | 37 |
| VALID | O6 (1) | 5 |
| TEST | F0-F7, F9, F11 (10) | 57 |

The proposed deterministic subset contains 32 evenly spaced products per
TRAIN scene (224 pairs), 32 products from VALID/O6, and one predeclared
canonical no-AHE product per TEST scene. It contains 1,184, 160, and 57 box
instances respectively. These are augmented renderings of only 37 training
people and five validation people, not independent annotations. Consequently,
the experiment can test optimization and cross-modal utility, but it cannot
support a strong claim about generalization. TEST is never used for training,
threshold selection, or fusion design.

## Matched experiment

All arms use the existing frozen ForestPersons DINOv3 ViT-S/16 rank-4 LoRA
encoder and a fresh FCOS head with identical random initialization:

1. RGB integral-image patch features.
2. Thermal integral images converted to grayscale, replicated to three
   channels, and encoded with the same frozen DINOv3 preprocessing.
3. Fixed mean of L2-normalized RGB and thermal patch features, followed by
   L2 normalization. This is deliberately non-trainable and precedes any
   attention-based fusion.

The official paired products share virtual coordinates and labels. No fitted
warp is used. Residual modality/rendering misalignment remains a known
limitation, as documented by the correspondence protocol. The detector head
alone is trained for 10 fixed epochs with AdamW, LR 3e-4, weight decay 1e-4,
batch size 1 and gradient accumulation 4. It retains the established FCOS
assignment, positive-normalized focal loss, Smooth L1 distance regression,
centerness loss, NMS 0.5, and top-300 filtering. The threshold satisfying
FP/image <= 0.425 is selected on VALID/O6 only. This calibration is fragile
because O6 is one scene and repeats five people.

The TEST evaluation reports AP/AP50/AP75, AR@1/10/100, and threshold-specific
precision, recall, TP/FP/FN, and FP/image. AOS does not provide ForestPersons
visibility levels, so visibility-specific recall is unavailable and must not
be fabricated.

## Commands

Run from the repository root. Each output directory is protected
against overwrite.

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/run_aos_dinov3_multimodal_detection.py prepare \
  --device mps

PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/run_aos_dinov3_multimodal_detection.py train \
  --device mps

PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/run_aos_dinov3_multimodal_detection.py evaluate \
  --device mps
```

Preparation caches each frozen modality feature once. Training reads only
TRAIN and VALID cache records. Evaluation accepts exactly one canonical TEST
product per official test scene and uses each arm's O6-locked threshold.

## Validation

- Actual local discovery: 224/32/10 paired products and 1,184/160/57 labels.
- RGB/thermal label bytes, image dimensions, paired files, and split scenes are
  checked before feature extraction.
- Four focused tests pass for deterministic selection, scene leakage,
  normalized fusion, coordinate conversion, losses, and finite gradients.
- The implementation passes Python compilation.

No experiment metrics exist yet because training was explicitly prohibited.

