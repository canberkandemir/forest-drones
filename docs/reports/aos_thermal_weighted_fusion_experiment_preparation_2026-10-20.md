# AOS Thermal-Weighted Fusion Experiment Preparation

Date: 2026-10-20  
Status: implemented and tested; training/evaluation not executed

## Controlled comparison

The existing scene-disjoint AOS feature cache and manifest are reused without
feature extraction. Three fresh FCOS heads start from one identical random
state and receive the same images, order, optimizer, epochs, gradient
accumulation, losses, assignment, NMS, and candidate limits:

| Arm | Frozen feature supplied to FCOS |
|---|---|
| `thermal` | thermal DINOv3 feature |
| `thermal_weighted_075` | normalize(0.25 RGB + 0.75 thermal) |
| `thermal_weighted_090` | normalize(0.10 RGB + 0.90 thermal) |

The feature tensors come from the same frozen ForestPersons DINOv3 ViT-S/16 +
LoRA encoder. Only each fresh FCOS head is trained. The established budget is
10 epochs, AdamW LR 3e-4, weight decay 1e-4, batch size 1, accumulation 4,
seed 20261020, short side 672, and FP/image calibration budget 0.425.

TRAIN uses 224 paired products from seven scenes. Operating thresholds and the
preferred arm are chosen only on 32 VALID/O6 products. Selection first prefers
arms satisfying FP/image <= 0.425, then maximizes validation recall and
precision, with predeclared arm order resolving exact ties. This decision is
persisted in `matched_training_summary.json` before TEST evaluation.

TEST evaluation requires that selection record and uses the ten untouched
canonical test scenes. It evaluates all three locked arms for transparent
comparison while marking the validation-preferred arm as primary. The TEST
labels never influence arm or threshold selection.

## Commands

From the repository root:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/run_aos_dinov3_multimodal_detection.py train \
  --manifest outputs/aos/dinov3_rgb_thermal_detection_20261020/prepared/manifest.json \
  --output-dir outputs/aos/dinov3_thermal_weighted_fusion_20261020/training \
  --arms thermal,thermal_weighted_075,thermal_weighted_090 \
  --epochs 10 \
  --gradient-accumulation 4 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --seed 20261020 \
  --fp-budget 0.425 \
  --device mps
```

Only after that command completes successfully:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/run_aos_dinov3_multimodal_detection.py evaluate \
  --manifest outputs/aos/dinov3_rgb_thermal_detection_20261020/prepared/manifest.json \
  --training-dir outputs/aos/dinov3_thermal_weighted_fusion_20261020/training \
  --output-dir outputs/aos/dinov3_thermal_weighted_fusion_20261020/evaluation \
  --arms thermal,thermal_weighted_075,thermal_weighted_090 \
  --require-validation-selection \
  --device mps
```

Output directories are overwrite-protected. Existing thermal and mean-fusion
checkpoints and results remain untouched.

## Validation

Four focused tests pass, including exact 0.75 thermal-weighted feature
composition and normalization. Both implementation files compile, and the
training/evaluation CLI definitions load successfully. No training or TEST
evaluation was run in this task.

