# DTSS ForestPersons Object-Detection Pilot

Date: 2026-10-20

## Scope

This isolated pilot tests whether the existing DTSS Spiking Transformer can
consume real ForestPersons images and learn bounding boxes. It does not modify
or use the DINOv3 or YOLO detector implementations.

## Model and data

- 128 x 128 RGB input with ImageNet normalization.
- 16 x 16 convolutional patch embedding: 8 x 8 = 64 tokens.
- One 32-dimensional, four-head DTSS Spiking Transformer block at `T_max=4`.
- One objectness logit and one normalized box per patch token.
- Nearest-patch one-to-one GT assignment, BCE objectness, and Smooth L1 box
  regression. Predictions use score filtering and class-agnostic NMS.
- 35,713 parameters.
- Train: 32 single-person images from the existing train400 manifest, eight at
  each visibility level. Validation: 16 single-person images from val600, four
  at each level. Official train/validation splits and SHA-256 checks gave zero
  image overlap.
- Eight epochs, batch size 4, AdamW at 0.002. Both arms used identical initial
  weights, sample order, optimizer, and training budget.
- Learned DTSS added a normalized mask-count penalty of 0.02; Fixed T=4 froze
  the same initially all-active timestep parameters.

Independent x/y resizing and COCO xywh-to-normalized-xyxy conversion are
tested. A four-image overfit gate reduced detection loss from 1.2160 to 0.00834
and passed before the matched experiment.

## Results

The operating point is fixed at confidence 0.05, NMS IoU 0.50. AP50 ranks all
saved predictions down to 0.001. With only 16 GT people, all metrics are
**inconclusive feasibility measurements**.

| Arm | AP50 | AR@100 | Recall@0.50 | Precision | FP/image | R20 | Train time | Peak MPS driver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed T=4 | 0.0004 | 0.0188 | 0.0625 | 0.0025 | 25.0625 | 0.0000 | 4.82 s | 0.035 GiB |
| Learned DTSS | 0.0011 | 0.0375 | 0.1250 | 0.0049 | 25.6250 | 0.0000 | 5.23 s | 0.037 GiB |

Fixed T detected one of four visibility-100 people; learned DTSS detected two.
Neither arm detected a visibility-20, -40, or -70 person at the fixed threshold.
Mean inference latency was 0.165 s/image for Fixed T and 0.113 s/image for the
sequentially run DTSS arm; warm-state effects make this small timing difference
non-comparative. Process high-water RSS was 728 MiB after the first arm and
1,164 MiB after the second, so it is also cumulative rather than per-arm peak.

## DTSS behavior

All seven learned timestep parameter vectors received finite gradients and
changed (L2 deltas 0.085-0.234). Nevertheless, every binary prefix mask remained
at 4/4 active steps. Thus this run demonstrated timestep-parameter learning but
no discrete timestep reduction. Eager PyTorch also evaluates all four steps;
the masks zero outputs rather than skipping computation.

## Limitations and decision

The forward, backward, annotation scaling, checkpointing, and evaluation paths
are valid, but the tiny detector is severely undertrained and poorly
calibrated. The apparent one-person DTSS gain is not evidence of superiority.
Any next experiment should first improve this isolated detector on a larger
but still bounded subset and define a mask schedule that crosses a timestep
boundary; comparing runtime before real step skipping would remain misleading.

Artifacts:

- `src/forestpersons/dtss_detector.py`
- `scripts/run_dtss_forestpersons_detection_pilot.py`
- `tests/test_dtss_detector.py`
- `outputs/dtss_detection/train32_val16_seed20261020_rerun1/`
- `outputs/dtss_detection_pilot_rerun1.log`

The earlier `train32_val16_seed20261020/summary.json` is invalid because its
pilot adapter omitted normalized annotation metadata, yielding zero recognized
GT. It is preserved for auditability and is superseded by `rerun1`.
