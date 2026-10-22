# DINOv3 ViT-L LoRA + FCOS Training Smoke - ForestPersons

This report records a resource-bounded training smoke for official DINOv3
ViT-L/16 web-pretrained and satellite-pretrained checkpoints. It does not train
a production detector and should not be interpreted as model performance.

## Goal

Test whether matched LoRA + FCOS detector training is technically feasible for:

- Web: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Satellite: `facebook/dinov3-vitl16-pretrain-sat493m`

Both use the same ViT-L/16 architecture, rank-4 q/v LoRA, the same FCOS head
architecture, the same training images, the same seed, and the same optimizer
settings.

## Code Changes

- `src/forestpersons/dinov3_lora.py`
  - Added optional `model_id` support for loading/saving LoRA checkpoints.
  - Saved `backbone_model_id` and `head_input_dim` in checkpoints.
  - Load now reconstructs the correct backbone and FCOS input dimension.
- `scripts/train_dinov3_lora_fcos.py`
  - Added `--model-id`.
  - Added `--head-init-seed` so random FCOS heads can be initialized
    identically across matched runs.
  - Head input dimension is inferred from the loaded backbone config.
- `scripts/evaluate_forestpersons_framework.py`
  - Added `--short-side` for online LoRA evaluation.
- `tests/test_dinov3_vitl_feasibility.py`
  - Existing ViT-L ID and parameter checks.

All ViT-S defaults are preserved. Existing ViT-S checkpoints and results were
not modified.

## Matched Short-Side 336 Training Smoke

Common settings:

- Train manifest:
  `outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json`
- Images: first 10 images
- Epochs: 1
- Batch size: 1
- Gradient accumulation: 4
- Optimizer: AdamW, same as existing LoRA script
- `lr_lora`: 0.0001
- `lr_head`: 0.0003
- Weight decay: 0.0001
- Seed: 20261013
- Head init seed: 20261013
- LoRA: rank 4, alpha 16, dropout 0.05
- LoRA targets: `attention.q_proj`, `attention.v_proj`
- Head initialization: random, identical seed; no ViT-S head weights reused
- Device: MPS

| Model | Images | Optimizer steps | Loss | Cls loss | Box loss | Centerness loss | Time | Peak observed MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 10 | 3 | 1.1591 | 0.02235 | 0.19944 | 0.73783 | 9.76 s | 7.21 GiB |
| Satellite ViT-L | 10 | 3 | 1.1571 | 0.02247 | 0.20198 | 0.73071 | 8.47 s | 7.21 GiB |

Gradient checks:

- Web ViT-L: LoRA gradients nonzero for 96/96 trainable LoRA tensors; head
  gradients nonzero for 18/18 tensors.
- Satellite ViT-L: LoRA gradients nonzero for 96/96 trainable LoRA tensors; head
  gradients nonzero for 18/18 tensors.

Parameter counts:

| Component | Parameters |
| --- | ---: |
| Frozen ViT-L backbone total | 303,522,816 |
| Trainable q/v LoRA | 393,216 |
| Trainable ViT-L FCOS head | 627,046 |
| Total trainable | 1,020,262 |
| Total including frozen backbone | 304,149,862 |

Artifacts:

- Web checkpoint:
  `outputs/dinov3_vitl_lora_smoke/web_vitl_train10_short336_seed20261013/dinov3_lora_fcos.pt`
- Satellite checkpoint:
  `outputs/dinov3_vitl_lora_smoke/sat_vitl_train10_short336_seed20261013/dinov3_lora_fcos.pt`
- Logs:
  `outputs/dinov3_vitl_lora_smoke_web_train10_short336.log`
  and `outputs/dinov3_vitl_lora_smoke_sat_train10_short336.log`

## Evaluation/Inference Sanity Check

Each short-336 smoke checkpoint was loaded through the canonical evaluation
framework and run on the first 5 val600 images at short side 336.

| Model | Val images | GT | Raw predictions | End-to-end sec/image |
| --- | ---: | ---: | ---: | ---: |
| Web ViT-L smoke | 5 | 6 | 1,500 | 0.385 |
| Satellite ViT-L smoke | 5 | 6 | 1,500 | 0.364 |

The AP/recall values were zero, which is expected and not meaningful because the
heads were randomly initialized and trained on only 10 images for one epoch. The
sanity check verifies checkpoint loading, image preprocessing, ViT-L feature
shape handling, FCOS decoding, COCO export, and visualization output.

## Short-Side 672 Attempt

After both short-336 runs succeeded, a conservative short-side 672 smoke was
attempted for the web ViT-L model with 20 images.

Result: failed before completing the first epoch due to MPS memory exhaustion.

Error excerpt:

```text
RuntimeError: MPS backend out of memory (MPS allocated: 19.36 GiB,
other allocations: 626.77 MiB, max allowed: 20.13 GiB). Tried to allocate
607.55 MiB on private pool.
```

The satellite short-side 672 run was not attempted after the web model failed.
No memory-watermark override was used because that could risk system instability
on a 16 GB unified-memory Mac.

## Conclusion

- Matched ViT-L LoRA + FCOS training is feasible at short side 336 for a tiny
  10-image smoke.
- Checkpoint save/load and evaluation integration work for both official ViT-L
  checkpoints.
- Canonical short side 672 is not currently feasible for ViT-L LoRA training on
  this Mac M4 16 GB setup with the current full-precision MPS path.
- A fair web-vs-satellite train400 comparison should not be started at short
  side 672 on this machine without changing the memory strategy.

## Recommended Next Step

For a scientifically useful but feasible matched comparison, run a train400
pilot at short side 336 for both ViT-L checkpoints, or move the short-side 672
experiment to a CUDA machine with substantially more GPU memory. If short-side
336 is used, clearly label it as a reduced-resolution ViT-L comparison and do
not compare it directly against the existing ViT-S short-side-672 results
without noting the resolution difference.
