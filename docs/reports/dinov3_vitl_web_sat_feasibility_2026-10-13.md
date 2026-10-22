# DINOv3 ViT-L Web vs Satellite Feasibility Smoke - ForestPersons

This report evaluates whether a controlled DINOv3 ViT-L/16 comparison is practical
for the ForestPersons project. It does not train a detector and does not modify
existing ViT-S checkpoints.

## Models

| Role | Hugging Face id | Architecture | Hidden dim | Layers | Heads | Patch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Web-pretrained | `facebook/dinov3-vitl16-pretrain-lvd1689m` | ViT-L/16 | 1024 | 24 | 16 | 16 |
| Satellite-pretrained | `facebook/dinov3-vitl16-pretrain-sat493m` | ViT-L/16 | 1024 | 24 | 16 | 16 |

Both repositories are public but gated with manual access on Hugging Face. In
this local environment, config/preprocessor files and weights were accessible
through the existing authenticated Hugging Face setup. No license or access
restriction was bypassed.

Reference pages:

- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m

## Resource Estimate Before Smoke

- Weight file per model: 1,212,559,808 bytes, about 1.13 GiB.
- Both ViT-L checkpoints: about 2.26 GiB cached weights.
- Available local disk before the run was about 139 GiB.
- Each ViT-L backbone has 303,522,816 parameters.
- A ViT-L FCOS head parameterized with `input_dim=1024` has 627,046 parameters.
- Rank-4 q/v LoRA across all 24 layers has 393,216 trainable LoRA parameters.
- Total trainable parameters for ViT-L LoRA + FCOS head would be about 1,020,262.

## Smoke Setup

- Script: `scripts/benchmark_dinov3_vitl_feasibility.py`.
- Output: `outputs/dinov3_vitl_feasibility/web_vs_sat_val10_short336_20261013/summary.json`.
- Device: MPS on MacBook Pro M4 with 16 GB unified memory.
- Images: first 10 images from `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`.
- Initial conservative forward: one image at short side 224.
- Benchmark forward: 10 images at short side 336.
- Backward smoke: one image at short side 224 with rank-4 q/v LoRA and a ViT-L FCOS head.
- No optimizer step, no training loop, no checkpoint save.

## Results

| Model | Load time | Forward short 336, 10 img | Sec/img | Peak observed MPS driver memory | Backward short 224 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 94.21 s | 3.65 s | 0.365 s | 2.41 GiB | 1.64 s |
| Satellite ViT-L | 391.74 s | 3.37 s | 0.337 s | 2.41 GiB | 0.32 s |

Feature shapes:

- Short side 224 on a 1920 x 1080 image resized to 400 x 224 produced a
  14 x 25 x 1024 patch feature grid.
- Short side 336 on 1920 x 1080 images resized to 592 x 336 produced a
  21 x 37 x 1024 grid.
- A wider 2288 x 1080 image resized to 704 x 336 produced a 21 x 44 x 1024 grid.

Backward smoke:

- LoRA targets were found for q/v projections in all 24 transformer layers:
  48 wrapped Linear modules.
- The LoRA state has 96 tensors (`lora_a` and `lora_b` per wrapped module).
- 48/96 LoRA tensors had nonzero gradients in the first backward pass. This is
  expected with zero-initialized `lora_b`: `lora_b` receives gradients first,
  while `lora_a` starts receiving gradients after `lora_b` becomes nonzero.
- FCOS head gradients were nonzero for 18/18 head tensors.

## Compatibility With Existing Detector Code

The existing ViT-S detector stack can be parameterized for ViT-L, but not by
loading current checkpoints directly:

- Current canonical ViT-S model id: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Current FCOS head default input dimension: 384.
- ViT-L feature dimension: 1024.
- Therefore the FCOS head must be initialized as `DINOv3FCOSHead(input_dim=1024)`.
- Existing ViT-S FCOS and LoRA checkpoints are not shape-compatible with ViT-L.
- Existing frozen ViT-S feature caches cannot be reused for ViT-L.
- LoRA target suffixes remain compatible:
  `attention.q_proj` and `attention.v_proj`.

## Feasibility Judgment

Forward inference at reduced resolution is feasible locally on MPS. A minimal
backward pass is also feasible at short side 224. However, this does not prove
that matched LoRA training at the existing short side 672 is practical:

- The current ViT-S smoke benchmark at short side 672 was about 0.455 s/image
  for frozen forward extraction.
- The ViT-L smoke at short side 336 was about 0.34-0.37 s/image, but it used
  roughly one quarter of the patch tokens used by short side 672.
- Attention memory and runtime grow strongly with token count, so short side 672
  ViT-L training could be substantially slower and more memory-sensitive.

For a future matched train400 experiment, the safest protocol is:

1. Run a 5-10 image ViT-L LoRA + FCOS training smoke at short side 336.
2. If stable, run a 20-50 image smoke at short side 672.
3. Only then attempt train400, keeping web ViT-L and satellite ViT-L matched:
   same train400 subset, same val600 evaluation, same LoRA rank, same FCOS
   initialization strategy, same epochs, same seed, same preprocessing, and
   same evaluation metrics.

Estimated train400 cost is uncertain until a short-side-672 backward smoke is
measured. Based on model size and token scaling, expect ViT-L train400 to be
materially slower than ViT-S train400; a practical range is several hours per
two-epoch matched pair on the Mac, with risk of MPS memory pressure at short
side 672.

## Limitations

- The smoke used short side 336 for forward benchmarking, not the canonical 672.
- The backward smoke used short side 224 and a synthetic scalar loss, not the
  full FCOS target/loss path.
- No AP, recall, or detector metrics were produced.
- No train400 or train5000 ViT-L adaptation was started.
