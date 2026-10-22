# DINOv3 ViT-S LoRA + Lightweight FPN FCOS Implementation

Date: 2026-10-13

## Objective

Implement and validate a lightweight multi-scale FCOS-style detector for the
existing DINOv3 ViT-S + LoRA ForestPersons pipeline. This is infrastructure and
smoke validation only; no train5000 experiment was started.

## Sanity Review

Targeted review of the existing single-scale FCOS path found no verified bug in:

- Box coordinate scaling: existing decode uses separate `scale_x` and `scale_y`.
- FCOS target coordinates: ground-truth boxes are scaled to resized image space
  before assignment.
- Regression encoding: targets are left/top/right/bottom distances divided by
  the level stride; the single-scale stride is the DINO patch size, 16.
- Decoding: predicted distances are softplus-transformed, multiplied by stride,
  clipped in resized image coordinates, then divided by separate x/y scales.

No existing single-scale checkpoint, script, or output was modified.

## New Architecture

Added `DINOv3FPNFCOSHead` in `src/forestpersons/dinov3_detector.py`.

The head receives normalized DINOv3 patch features shaped `H x W x C` and builds
three feature levels:

| Level | Source | Stride |
| --- | --- | ---: |
| fine | projected DINO patch grid | 16 px |
| medium | average-pooled fine grid + lightweight adapter | 32 px |
| coarse | average-pooled medium grid + lightweight adapter | 64 px |

The FCOS tower and prediction heads are shared across levels. The module names
for `proj`, `shared`, `classification`, `box`, and `centerness` match the
existing single-scale FCOS head, so compatible trained weights can be reused.

## Assignment and Decoding

Added `build_fpn_fcos_targets`:

- Fine level handles small objects by default: 0-80 resized pixels.
- Medium level handles medium objects with overlap: 48-192 resized pixels.
- Coarse level handles larger objects: 128+ resized pixels.
- Candidate cells must lie inside the ground-truth box and inside a
  center-sampling window.
- If no level receives a positive candidate, a center fallback assigns the
  finest level.
- If multiple objects compete for the same cell, smaller-area ground truth wins.

Added `decode_fpn_fcos_predictions`:

- Decodes all levels in resized-image coordinates.
- Converts boxes back to original-image coordinates using separate
  `scale_x`/`scale_y`.
- Applies class-aware one-class NMS jointly across levels.

## Weight Reuse

The smoke test initialized from the current best ViT-S LoRA + FCOS checkpoint:

`outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`

Reused from the trained single-scale FCOS head:

- `proj.*`
- `shared.*`
- `classification.*`
- `box.*`
- `centerness.*`

Newly initialized:

- `pyramid_adapters.medium.*`
- `pyramid_adapters.coarse.*`

The existing LoRA adapters were loaded from the checkpoint for the smoke test.

## Tests

Added `tests/test_dinov3_fpn_fcos.py` covering:

- FPN output shapes for fine/medium/coarse feature maps.
- Scale-aware target assignment.
- Original-image coordinate decoding with separate x/y scales.
- Compatible single-scale FCOS weight reuse.

Full test suite:

`53 passed in 4.21s`

## MPS Smoke Test

Command:

```bash
PYTHONPATH="$PWD/src:$PWD" \
.venv-dinov3/bin/python \
scripts/smoke_dinov3_lora_fpn_fcos.py \
  --output-dir outputs/dinov3_lora_fpn_fcos/smoke_train2_seed20261013 \
  --max-images 2 \
  --short-side 672 \
  --device mps
```

Result:

- Device: MPS.
- Images: 2 real ForestPersons train400 images.
- Runtime: 3.32 seconds.
- Peak observed MPS driver memory: 5.95 GB.
- LoRA gradients: nonzero for 48/48 trainable tensors.
- FPN head gradients: nonzero for 26/26 trainable tensors.
- Losses were finite.

Smoke rows:

| Image | Loss | Positive cells | Decoded preds at conf 0.50 | MPS driver memory |
| --- | ---: | ---: | ---: | ---: |
| `088_FPV_Person_summer_forest-0006.jpg` | 0.6849 | 4 | 3 | 5.94 GB |
| `336_FPV_Person_summer_forest-0043.jpg` | 2.4094 | 7 | 0 | 5.95 GB |

Smoke output:

`outputs/dinov3_lora_fpn_fcos/smoke_train2_seed20261013/summary.json`

## Limitations

- This is not a trained detector result.
- The new pyramid adapters are randomly initialized and need training.
- The size ranges are a reasonable first pass, not yet tuned on a held-out set.
- The smoke loaded the current best LoRA checkpoint to validate compatibility;
  the eventual controlled training run should explicitly define whether it
  initializes from that checkpoint or from frozen-FCOS head weights only.

## Next Step

Run an approved bounded train400 or train5000 pilot for the FPN head, comparing
against the existing single-scale ViT-S LoRA + FCOS baseline at the same
confidence sweep and FP/image budgets.
