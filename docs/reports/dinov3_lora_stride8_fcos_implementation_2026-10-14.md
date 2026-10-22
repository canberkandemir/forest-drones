# DINOv3 LoRA Stride-8 FCOS Implementation

Date: 2026-10-14

## Scope

This task implemented and validated an optional stride-8, single-level FCOS
detector. It did not run train400/train5000 training or val600 evaluation. The
existing stride-16 detector, checkpoints, loss functions, optimizer settings,
and default inference behavior remain unchanged.

## Architecture

The input remains the `facebook/dinov3-vits16-pretrain-lvd1689m` patch grid at
stride 16. The trained 1x1 FCOS projection maps each 384-dimensional patch
feature to 160 channels. A new adapter then produces a 2x denser feature map:

1. depthwise 2x2 transposed convolution, stride 2;
2. 1x1 trainable channel mixing convolution;
3. the existing shared FCOS tower and class, L/T/R/B, and centerness predictors.

The adapter has 26,400 parameters. It is initialized to exact 2x2 spatial
replication with identity channel mixing. No coarser pyramid levels were added.

The full trainable configuration has 624,774 parameters: 73,728 LoRA
parameters, 26,400 new adapter parameters, and 524,646 reused FCOS parameters.

## Checkpoint Initialization

Source checkpoint:
`outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

Reused:

- all rank-4 q/v LoRA adapter weights in the 12 DINOv3 blocks;
- FCOS `proj`, `shared`, `classification`, `box`, and `centerness` tensors.

New:

- `stride8_adapter.upsample.weight`;
- `stride8_adapter.mix.weight`;
- `stride8_adapter.mix.bias`.

The standard checkpoint loader now recognizes `head_kind=stride8_fcos`. A saved
smoke checkpoint was reloaded successfully. Existing `fcos` and `fpn_fcos`
checkpoint paths remain unchanged.

## Coordinates and Assignment

The common target builder and decoder now accept an explicit feature stride;
their default remains 16. At stride 8:

- grid centers are `(x + 0.5) * 8`, `(y + 0.5) * 8`;
- L/T/R/B targets are measured in resized-image pixels and divided by 8;
- decoded distances are multiplied by 8;
- center sampling uses the same 1.5-cell radius;
- smaller-area GT precedence for overlapping boxes is unchanged;
- resize inversion uses independent `scale_x` and `scale_y`;
- clipping and class-aware single-class NMS are unchanged.

Tests cover target/decode round trips, independent x/y scaling, overlap-aware
assignment, finite gradients, and stride-16 default compatibility.

## Assignment Analysis

The existing deterministic train400 manifest contains 400 images and 525 GT
people at short side 672.

| Group | GT | Mean positives s16 | Mean positives s8 | s16 zero | s8 zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 525 | 7.89 | 8.95 | 1 | 0 |
| Visibility 20 | 112 | 7.19 | 8.89 | 0 | 0 |
| Visibility 40 | 119 | 7.87 | 8.92 | 0 | 0 |
| Visibility 70 | 132 | 7.99 | 8.95 | 1 | 0 |
| Visibility 100 | 162 | 8.29 | 9.00 | 0 | 0 |
| Max resized side 16-32 px | 10 | 3.20 | 8.40 | 0 | 0 |

Stride 8 yielded more final positive locations for 148/525 GT instances and
removed one stride-16 zero-positive case caused by overlap ownership. Train400
contains no GT with maximum resized side below 16 pixels, so the claim that
truly tiny boxes receive positives cannot be measured from this subset. The
production fallback rule is nevertheless retained and unit-tested with a
synthetic tiny box.

## MPS Smoke

Four real train400 images were processed at short side 672 with batch size 1
and one AdamW optimizer step per image. This was an implementation smoke, not a
training result.

- Runtime: 5.53 seconds total, 1.38 seconds/image.
- Peak observed MPS driver memory: 5.52 GiB.
- Loss range: 0.7010-1.1498; all losses finite.
- Positive cells: 9 for each sampled image.
- All 48 LoRA parameter tensors had finite nonzero gradients on every step.
- All 3 adapter tensors had finite nonzero gradients on every step.
- All 18 reused FCOS tensors had finite nonzero gradients on every step.
- Checkpoint save and reload succeeded.

## Validation

- Maintained test suite: 87 passed.
- A repository-root test collection also found the preserved standalone
  `test_dinov3.py` and `test_yolo.py`; these require Ultralytics, which is not
  installed in `.venv-dinov3`. They are not part of the maintained `tests/`
  suite and were not modified.

Artifacts:

- `outputs/dinov3_lora_stride8_fcos/assignment_train400_short672/summary.json`
- `outputs/dinov3_lora_stride8_fcos/assignment_train400_short672/assignments.csv`
- `outputs/dinov3_lora_stride8_fcos/smoke_train4_seed20261014/summary.json`
- `outputs/dinov3_lora_stride8_fcos/smoke_train4_seed20261014/dinov3_lora_stride8_fcos_smoke.pt`

## Conclusion

The stride-8 model is technically ready for a controlled train400 experiment
at short side 672 on this Mac. The smoke used materially less than 16 GB unified
memory and showed stable gradients. The assignment results support the intended
mechanism, especially for the smallest available boxes, but they do not prove
that detection recall, AR, AP75, or 20%-visibility recall will improve.
