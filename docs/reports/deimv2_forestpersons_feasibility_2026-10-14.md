# DEIMv2 Feasibility for ForestPersons

Date: 2026-10-14

## Scope

This is a read-only feasibility assessment for using DEIMv2 in the LogXY
ForestPersons project. No DEIMv2 model was trained, no checkpoints were
downloaded, and existing FCOS/YOLO code and outputs were not modified.

Primary sources inspected:

- Paper/arXiv: <https://arxiv.org/abs/2509.20787>
- Official repository listed by arXiv: <https://github.com/Intellindust-AI-Lab/DEIMv2>
- Official Hugging Face checkpoints:
  - <https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO>
  - <https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO>

The arXiv metadata confirms the paper title, abstract, and official code URL.
The repository was cloned into `/tmp` for inspection only.

## What DEIMv2 Is

DEIMv2 extends the earlier DEIM detector with DINOv3-style features. The paper
abstract states that X/L/M/S variants use DINOv3-pretrained or distilled
backbones and a Spatial Tuning Adapter (STA), while ultra-light models use
HGNetv2. The official repository contains model configs, training code, and
Hugging Face model loading examples.

The official model zoo reports these relevant COCO results:

| Variant | Config | COCO AP | Params | GFLOPs |
| --- | --- | ---: | ---: | ---: |
| S | `deimv2_dinov3_s_coco.yml` | 50.9 | 9.7M | 25.6 |
| M | `deimv2_dinov3_m_coco.yml` | 53.0 | 18.1M | 52.2 |
| L | `deimv2_dinov3_l_coco.yml` | 56.0 | 32.2M | 96.7 |
| X | `deimv2_dinov3_x_coco.yml` | 57.8 | 50.3M | 151.6 |

These are COCO results and should not be assumed to transfer to ForestPersons.

## Architecture From Official Code

The official DEIMv2 detector is:

`input image -> DINOv3STAs backbone -> HybridEncoder -> DEIMTransformer -> PostProcessor`

### DINOv3 / Distilled Backbone

The backbone wrapper is `engine/backbone/dinov3_adapter.py`.

Important details:

- `DINOv3STAs` can use either:
  - a custom tiny `VisionTransformer` for S/M, or
  - the repo's local `DinoVisionTransformer` for L/X.
- The S config uses:
  - `name: vit_tiny`
  - `embed_dim: 192`
  - `weights_path: ./ckpts/vitt_distill.pt`
- The M config uses:
  - `name: vit_tinyplus`
  - `embed_dim: 256`
  - `weights_path: ./ckpts/vittplus_distill.pt`
- The L config uses:
  - `name: dinov3_vits16`
  - `weights_path: ./ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
  - `interaction_indexes: [5, 8, 11]`
  - `conv_inplane: 32`
  - `hidden_dim: 224`
- The X config uses:
  - `name: dinov3_vits16plus`
  - `weights_path: ./ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

So the official DEIMv2 S/M variants are not our
`facebook/dinov3-vits16-pretrain-lvd1689m`; they are distilled tiny variants.
The official DEIMv2 L variant is the closest to our ViT-S/16 backbone.

### Spatial Tuning Adapter

`DINOv3STAs` includes `SpatialPriorModulev2`, a lightweight convolutional
detail branch producing 1/8, 1/16, and 1/32 features. DINOv3 semantic token
features are resized to these scales and concatenated with STA detail features,
then projected and normalized. This is not equivalent to our existing
single-scale FCOS over DINOv3 patch features.

### Encoder, Decoder, Queries

The official configs use:

- `HybridEncoder` over three feature levels with strides `[8, 16, 32]`.
- `DEIMTransformer` with:
  - `num_queries: 300`
  - `num_denoising: 100`
  - `reg_max: 32`
  - `num_points: [3, 6, 3]`
  - usually 4 decoder layers for S/M/L and 6 for X.
- Query selection is produced from encoder outputs and anchors before the
decoder, rather than dense-grid FCOS or YOLO assignment.

### Losses and Matching

The base DEIMv2 criterion uses:

- classification-like `loss_mal`;
- L1 box loss `loss_bbox`;
- GIoU `loss_giou`;
- fine-grained localization `loss_fgl`;
- decoupled distillation focal/local loss `loss_ddf`.

The matcher is Hungarian matching with optional late-stage matcher change.
For DINOv3 S/M configs, the matcher changes after many epochs to an IoU-aware
classification/IoU ordering. This differs from our FCOS positive assignment and
YOLO task-aligned assignment.

## Dependency and Hardware Findings

Official `requirements.txt` pins:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `faster-coco-eval>=1.6.7`
- `PyYAML`
- `tensorboard`
- `scipy`
- `calflops`
- `transformers`

The repo uses CUDA-aware utilities in several places:

- AMP imports from `torch.cuda.amp`.
- Distributed utilities call CUDA helpers when available.
- Logging utilities include CUDA tensor communication helpers.

However, the deformable attention path inspected in `engine/backbone/ms_deform_attn.py`
and decoder code has a pure PyTorch deformable-attention implementation for
forward computation. The custom CUDA extension import is optional in that file,
and the active decoder path imports a PyTorch implementation from D-FINE-style
code. This suggests CPU/MPS forward may be possible, but MPS training is not
verified by the official project.

Potential MPS blockers:

- `nn.SyncBatchNorm` appears in the STA and feature projections. It may run in
  single-process mode, but it is not an ideal MPS training choice and may need
  conversion to regular BatchNorm or GroupNorm for local experiments.
- The repo is written around CUDA/DistributedDataParallel assumptions.
- Official training uses large augmentation pipelines: Mosaic, MixUp, CopyBlend,
  RandomIoUCrop, and 640x640 resizing.
- Full official schedules are long: 68 epochs for DINOv3-L, 102 for M, and
  132 for S. A 2-epoch train400 smoke would only test feasibility, not
  reproduce DEIMv2 performance.

## Checkpoint Availability

The relevant Hugging Face checkpoints are public and not gated as of this
inspection.

Approximate `model.safetensors` sizes from HEAD requests:

| Checkpoint | Size |
| --- | ---: |
| `Intellindust/DEIMv2_DINOv3_S_COCO` | 37.6 MiB |
| `Intellindust/DEIMv2_DINOv3_L_COCO` | 124.5 MiB |

These sizes are modest. Downloading them later would be reasonable after
approval, but was not done in this task.

## Can We Reuse Our Existing DINOv3 Backbone and LoRA Weights?

Not directly.

Our current best detector uses Hugging Face:

`facebook/dinov3-vits16-pretrain-lvd1689m`

with rank-4 LoRA on separate `q_proj` and `v_proj` modules.

DEIMv2's local DINOv3 implementation uses fused `qkv` attention layers in
`engine/backbone/dinov3/layers/attention.py`, not separate Hugging Face
`q_proj`/`k_proj`/`v_proj` modules. Therefore:

- Our LoRA adapter state is not plug-and-play with DEIMv2.
- Our FCOS head weights are not reusable because DEIMv2 has STA, a multi-scale
  encoder, object queries, and a transformer decoder.
- Our DINOv3 ViT-S base weights may be conceptually compatible with the
  official DEIMv2-L config, but state-dict conversion between Hugging Face and
  DEIMv2's local DINOv3 implementation would need verification.
- The most faithful path is to use official DEIMv2 checkpoints and model code,
  not retrofit our existing LoRA checkpoint.

## Feasibility on MacBook Pro M4 16 GB

### Faithful DEIMv2 Reproduction

Not recommended on the Mac.

Reasons:

- Official training schedules are long and designed for detection training, not
  2-epoch small pilots.
- MPS support is unverified by the authors.
- SyncBatchNorm and CUDA-oriented training utilities may require code changes.
- DINOv3-L config is closer to our backbone but has 32.2M parameters and 96.7
  GFLOPs at 640x640; training at 672 on 16 GB unified memory is risky.

Faithful reproduction should be done on a CUDA machine, ideally using the
official repo and official config with a COCO-format ForestPersons adapter.

### Small Local Feasibility Smoke

Possible, but should be framed as a smoke test only.

Smallest useful local plan:

1. Clone official DEIMv2 as a separate external baseline directory.
2. Create ForestPersons COCO train400/val600 exports with a single `person`
   category and remap disabled.
3. Start from `Intellindust/DEIMv2_DINOv3_L_COCO` if we want the DINOv3 ViT-S
   family, or `DEIMv2_DINOv3_S_COCO` if we want the lightest official DINO-like
   DEIMv2 checkpoint.
4. Run inference-only on 10 val600 images at 640x640 or 512x512 on MPS/CPU.
5. If inference works, attempt one optimizer step on 1-2 train images.
6. Only after that consider train400.

For an actual train400 pilot on Mac:

- Use batch size 1.
- Disable or simplify Mosaic/MixUp/CopyBlend initially.
- Consider 512x512 or 640x640 before 672.
- Expect runtime to be much slower than our existing FCOS head.
- Treat results as DEIMv2-inspired unless official config/losses/checkpoint are
  preserved.

## Feasibility on CUDA

CUDA is the better path.

Recommended CUDA pilot:

- Use official DEIMv2 repo.
- Use official `deimv2_dinov3_l_coco.yml` for closest DINOv3 ViT-S comparison,
  but modify:
  - `num_classes: 1`
  - dataset paths to ForestPersons train400/val600 COCO exports
  - `remap_mscoco_category: False`
  - output directory
- Initialize from official DEIMv2 DINOv3-L COCO checkpoint if possible.
- Run 10-image inference, then 1-epoch or short train400 fine-tuning smoke.
- Evaluate exported COCO predictions through our existing evaluation framework.

Estimated resource needs:

- Inference: likely fine on a T4/A10-style GPU.
- Fine-tuning train400: likely feasible on a 16 GB T4 with batch size 1-2,
  though augmentation and DINOv3-L memory must be checked.
- Full train5000: would be a longer CUDA experiment, not a Mac experiment.

## Faithful vs Inspired Options

### Faithful DEIMv2

Use:

- Official repo.
- Official DINOv3-L config or official S config.
- Official pretrained checkpoint.
- Official criterion/matcher/decoder.
- ForestPersons COCO adapter only.

Pros:

- Scientifically cleanest.
- Citable as DEIMv2 fine-tuning/pilot.

Cons:

- Harder on Mac.
- Does not reuse our LoRA or FCOS work.
- Requires careful one-class checkpoint adaptation.

### DEIMv2-Inspired Custom Adaptation

Implement STA + multi-scale DINOv3 features + query decoder inside our current
codebase.

Pros:

- Could reuse our existing Hugging Face DINOv3 and LoRA infrastructure.
- Easier to keep evaluation consistent.

Cons:

- Not a faithful DEIMv2 reproduction.
- High implementation complexity.
- Easy to accidentally create another weak custom detector like the YOLO-style
  head.

### More Practical DETR-Family Alternative

If we want a DETR-family comparison quickly, a more practical option is to use
official DETR/RT-DETR/D-FINE/DEIM components on CUDA with COCO-pretrained
weights and ForestPersons COCO exports. This would be less DINOv3-specific but
more likely to run reliably than rebuilding DEIMv2 locally.

## Recommendation

Proceed with DEIMv2 only as an official-code CUDA-oriented pilot, not as a
from-scratch local custom architecture.

Recommended next action:

1. Prepare a CUDA/Colab-style DEIMv2 smoke workflow in a separate directory.
2. Start with official `DEIMv2_DINOv3_L_COCO` for architectural closeness to
   DINOv3 ViT-S/16.
3. Run inference on 10-30 val600 images and export COCO predictions.
4. If inference works, run a tiny train400 smoke on CUDA.
5. Do not attempt Mac train400 until a one-image MPS train step is verified.

Decision:

- For the Friday ForestPersons meeting, DEIMv2 should be presented as a
  promising next DETR-family/DINOv3 detector to test, not as an already
  validated result.
- If CUDA access is limited, a more practical next experimental path remains
  improving the current DINOv3 LoRA + FCOS baseline, because it is proven to
  train and evaluate on the Mac.

## Remaining Risks

- Need to verify official checkpoint fine-tuning with `num_classes=1`.
- Need to verify whether official Hugging Face model loading exposes trainable
  components cleanly or whether YAML training is easier.
- Need to verify MPS compatibility empirically; code inspection alone is not
  enough.
- Need to ensure DEIMv2's non-commercial license is acceptable for LogXY's
  internship/research context.
- Need to avoid overstating COCO performance as ForestPersons performance.
