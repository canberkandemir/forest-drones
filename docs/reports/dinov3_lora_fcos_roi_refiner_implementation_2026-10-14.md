# DINOv3 LoRA + FCOS RoI Refiner Implementation

Date: 2026-10-14

## Scope

This work adds a lightweight second-stage person/quality and box-refinement
head to the existing ForestPersons DINOv3 ViT-S/16 + LoRA + single-scale FCOS
detector. It is an implementation and MPS smoke test only. No train400 or
train5000 experiment was run, and the original FCOS checkpoint and inference
behavior were not changed.

Source detector:

- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- LoRA: rank 4 on attention `q_proj` and `v_proj` in all 12 layers.
- Proposal head: trained single-scale FCOS head.
- Checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

## Architecture

The frozen detector produces normalized DINOv3 patch features with shape
`[grid_h, grid_w, 384]` and FCOS candidates in the resized-image coordinate
system. Candidates are decoded before NMS using the unchanged FCOS score
`sqrt(class_probability * centerness_probability)`.

For each proposal, `torchvision.ops.roi_align` samples a 3x3 feature grid from
the stride-16 DINOv3 patch map with `spatial_scale=1/16`, `sampling_ratio=2`,
and `aligned=True`. The refinement head is:

1. 1x1 convolution from 384 to 64 channels.
2. GroupNorm and GELU.
3. Flattened 3x3 spatial representation.
4. Linear projection from 576 to 128 dimensions and GELU.
5. One person/quality logit and four class-agnostic box deltas.

The head has **99,269 trainable parameters**. Box-delta weights and biases are
initialized to zero, so the initial decoded box is the FCOS proposal.

The DINOv3 backbone, LoRA adapters, and FCOS head are explicitly frozen. The
first experiment will optimize only the refinement head.

## Targets and losses

All matching is performed in the resized-image coordinate system used by FCOS
and RoIAlign. Ground-truth boxes use separate `scale_x` and `scale_y` values.

- Positive proposal: maximum GT IoU >= 0.50.
- Negative proposal: maximum GT IoU < 0.30.
- Ignore region: IoU in [0.30, 0.50).
- Per-image sample: up to 128 proposals, with at most 25% positives.
- Score loss: binary cross entropy over sampled positives and negatives.
- Box target: standard center-offset and log-scale Faster R-CNN deltas with
  weights `(10, 10, 5, 5)`.
- Box loss: Smooth L1 (`beta=1/9`) over positive proposals only.

Inference applies the learned deltas, clips boxes in resized coordinates,
combines proposal and refinement probabilities with a geometric mean by
default, then applies NMS. The scoring mode is configurable and must be selected
using training/calibration data in a later experiment rather than val600.

## MPS compatibility

PyTorch 2.9.0 and torchvision 0.24.0 successfully executed RoIAlign forward and
backward on Apple MPS. The synthetic operator check returned finite input
gradients.

An initially heavier convolutional RoI encoder made a 1,000-RoI sanity pass
impractically slow on MPS. The final 99k-parameter encoder retains the 3x3
spatial layout while using only a 1x1 projection and a small MLP. Proposal
matching still considers the top 1,000 pre-NMS FCOS candidates; smoke inference
is bounded to the top 300 candidates.

## Real-image smoke result

Command:

```bash
PYTHONPATH=src .venv-dinov3/bin/python scripts/smoke_dinov3_roi_refiner.py
```

The test used the first four deterministic train400 images at short side 672.

| Measure | Result |
|---|---:|
| Status | completed |
| Device | MPS |
| Images / optimizer steps | 4 / 4 |
| Pre-NMS proposals per image | 1,000 |
| Sampled positives per image | 8 to 14 |
| Sampled negatives per image | 114 to 120 |
| Total loss, first to last image | 1.3143 to 1.0680 |
| Nonzero-gradient tensors | 10 / 10 each step |
| Frozen-detector trainable parameters | 0 |
| Runtime | 14.89 s |
| Runtime per image | 3.72 s |
| Peak MPS driver memory | 1.64 GiB |

The four-step loss change is only a numerical smoke observation and is not an
experimental learning curve. All classification and box losses were finite.

Artifacts:

- `outputs/dinov3_roi_refiner/smoke_train4_seed20261014/summary.json`
- `outputs/dinov3_roi_refiner/smoke_train4_seed20261014/roi_refiner_smoke.pt`

## Limitations and train400 readiness

The refiner can improve and rescore an FCOS candidate, but it cannot recover a
person for which FCOS produces no overlapping pre-NMS proposal. Its feature map
has stride 16. People whose resized boxes span roughly one patch or less have
limited spatial detail, and RoIAlign interpolates from that same coarse map.
This is particularly relevant to small and heavily occluded people.

The implementation is ready for a controlled train400 pilot. The pilot should
freeze the complete source detector, retain the top-1,000 proposal pool for
matching, train only the 99k-parameter refiner, and compare both refined boxes
and rescored original boxes against the unchanged FCOS baseline. Proposal
recall before refinement must be reported, because it is the upper bound for
this second stage.
