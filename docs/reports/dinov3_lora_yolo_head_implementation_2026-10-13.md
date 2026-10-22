# DINOv3 ViT-S LoRA + YOLO-Style Head Implementation

Date: 2026-10-13

## Scope

This task implemented only the first proposed architecture: a YOLO-inspired
one-stage detector on top of DINOv3 ViT-S/16 patch features. Faster R-CNN and
Deformable DETR were not implemented.

Existing FCOS code, checkpoints, cached features, and evaluation outputs were
preserved. No full training and no val600 evaluation were run.

## Architecture

Backbone:

- `facebook/dinov3-vits16-pretrain-lvd1689m`
- Existing best ViT-S LoRA weights loaded from:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`
- Original DINOv3 backbone parameters remain frozen.
- LoRA adapters remain trainable for future training.

Detection head:

- New module: `src/forestpersons/dinov3_yolo.py`
- Single stride-16 feature map from DINOv3 patch tokens.
- Separate classification and regression towers.
- Single class: person.
- Distributional L/T/R/B regression with `reg_max=32` by default.
- Distribution Focal Loss for discrete distance-bin supervision.
- IoU regression loss on decoded L/T/R/B boxes.
- Task-aligned positive assignment using:
  `alignment = class_score^alpha * IoU^beta`
  with default `alpha=1.0`, `beta=6.0`, and `topk=10`.
- Inference decodes expected distance bins to boxes, applies confidence
  filtering, top-k pruning, and NMS.

The new head is initialized independently. No FCOS head weights are reused
because the prediction parameterization is different.

## Differences From FCOS

The existing FCOS head predicts continuous L/T/R/B distances and centerness. Its
assignment is center-sampling based.

The new YOLO-style head:

- does not predict centerness,
- predicts a discrete distribution for each box side,
- uses DFL plus IoU loss instead of Smooth L1/log-distance regression,
- uses task-aligned assignment based on current prediction quality,
- scores detections with the class probability only.

## Stride-16 Limitation

This implementation deliberately starts from the existing DINOv3 stride-16 patch
grid. Small or heavily occluded people may occupy very few cells. The assignment
code includes a nearest-center fallback for boxes with no interior patch center,
but this does not add true high-resolution features. This is a likely limitation
for 20%-visibility people and for AP75 localization quality.

## Smoke Test

Command:

```bash
PYTHONPATH=src .venv-dinov3/bin/python scripts/smoke_dinov3_lora_yolo_head.py \
  --output-dir outputs/dinov3_lora_yolo_head/smoke_train2_seed20261013 \
  --max-images 2 \
  --short-side 672 \
  --device mps
```

Result:

- Device: MPS
- Images: 2 real ForestPersons train400 images
- Forward/backward optimizer steps: 2
- Non-finite losses: none
- LoRA gradient tensors with nonzero gradients: 48/48
- Head gradient tensors with nonzero gradients: 18/18
- Total runtime: 4.24 s
- Peak MPS driver memory: 5,922,734,080 bytes, about 5.52 GiB

Parameter count:

| Component | Trainable | Total |
| --- | ---: | ---: |
| DINOv3 backbone with LoRA | 73,728 | 21,670,272 |
| YOLO-style head | 545,413 | 545,413 |
| Total | 619,141 | 22,215,685 |

Smoke losses:

| Image | Positives | Total loss | Classification | IoU | DFL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `088_FPV_Person_summer_forest-0006.jpg` | 4 | 1555.7352 | 1543.1132 | 0.9949 | 3.4400 |
| `336_FPV_Person_summer_forest-0043.jpg` | 8 | 479.7335 | 467.3512 | 0.9955 | 3.2771 |

The classification loss is large because the YOLO-style BCE term is summed over
all stride-16 cells and normalized by positive target score. This is similar in
spirit to modern YOLO losses, but it should be monitored in the train400 pilot.

## Tests

Focused tests added in `tests/test_dinov3_yolo_head.py` cover:

- head output shapes,
- DFL expectation,
- DFL finite gradients,
- IoU regression behavior for identical/partial/disjoint boxes,
- task-aligned positive assignment,
- tiny-box fallback assignment,
- no-positive loss stability,
- separate `scale_x`/`scale_y` coordinate decoding.

Passed:

```text
PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests/test_dinov3_yolo_head.py tests/test_dinov3_lora.py -q
15 passed
```

## Readiness

The implementation is ready for a controlled train400 pilot on the Mac. The
smoke test did not show MPS memory pressure, loss NaNs, or missing gradients.

Recommended next pilot:

- train400, 2 epochs,
- short-side 672,
- same seed and LoRA configuration as the current ViT-S LoRA baseline,
- initialize DINOv3 + LoRA from the best train5000 checkpoint,
- initialize the YOLO head freshly,
- evaluate on val600 with the verified COCO/custom framework.

The pilot should compare against the existing single-scale LoRA + FCOS result,
especially AP75, recall at FP/image <= 0.425, and recall at 20% visibility.
