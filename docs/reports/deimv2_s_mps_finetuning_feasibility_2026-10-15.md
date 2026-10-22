# DEIMv2-S ForestPersons Fine-Tuning Feasibility on Apple M4

Date: 2026-10-15

## Decision

DEIMv2-S can perform a real training step on this 16 GB Apple M4, but the
official trainer is not MPS-ready without a small local execution layer.
Full-model and frozen-transformer one-class smoke tests both completed with
finite official losses and gradients. Training requires:

1. an algebraically equivalent MPS implementation of the official D-FINE
   distribution integral; and
2. `PYTORCH_ENABLE_MPS_FALLBACK=1`, because PyTorch 2.9 does not implement
   `aten::grid_sampler_2d_backward` on MPS.

The fallback executes the unsupported backward operator on CPU. It does not
replace deformable attention, the matching rule, or any detection loss. A
small train400 pilot is technically feasible. It is not yet a push-button run
through the upstream CUDA-oriented training CLI.

## Verified implementation

- Official source: <https://github.com/Intellindust-AI-Lab/DEIMv2>, pinned at
  commit `1d2ca42171570c713e78fc6a766ec5104b7f4724`.
- Checkpoint: `Intellindust/DEIMv2_DINOv3_S_COCO`, already downloaded and
  verified locally; no additional model was downloaded for this audit.
- DEIMv2-S backbone: 192-dimensional ViT-Tiny distilled from DINOv3-S. This
  is not the full `facebook/dinov3-vits16-pretrain-lvd1689m` backbone.
- Detector: Spatial Tuning Adapter, stride 8/16/32 features, LiteEncoder,
  four-layer 300-query DEIM decoder, Hungarian matching, MAL classification,
  L1/GIoU box loss, FGL distribution loss, and DDF distillation loss.
- Official input: fixed 640 x 640, ImageNet mean/std normalization.
- One-class model: 9,688,435 parameters. The smaller count than the
  9,779,838-parameter COCO model is due only to replacing 80-class output
  heads with one-class heads.

The active deformable-attention path is implemented in ordinary PyTorch using
`grid_sample`; no compiled CUDA extension is required. CUDA assumptions still
exist in the official trainer's distributed setup, logging, AMP scaler, and
device selection, so the local smoke uses a narrow training harness around
the unchanged model and criterion.

## Environment and MPS compatibility

| Item | Verified value | Status |
| --- | --- | --- |
| Host | Apple M4 MacBook Pro, 16 GB unified memory | Supported |
| Python | 3.13.5 | Smoke passed |
| PyTorch / torchvision | 2.9.0 / 0.24.0 | Smoke passed; newer than upstream 2.5.1 / 0.20.1 pins |
| Precision | float32 | Recommended; CUDA AMP/GradScaler is not used |
| D-FINE integral | Upstream one-dimensional `F.linear` aborts in MPSGraph | Exact weighted-sum MPS workaround verified against CPU values and gradients |
| Deformable attention backward | `grid_sampler_2d_backward` absent on MPS | Exact operation works through PyTorch CPU fallback |
| Hungarian matcher | SciPy CPU assignment | Supported |
| Native MPS-only training | No | Blocked by `grid_sampler_2d_backward` |
| Hybrid MPS/CPU-fallback training | Yes | Two optimizer steps passed |
| CPU-only training | Yes | Two optimizer steps passed |

The weighted-sum compatibility function is algebraically identical to the
upstream integral. A unit test compares both outputs and gradients. The
official repository itself remains unmodified.

## Dataset and one-class adaptation

The existing COCO exports remain suitable and preserve original image IDs:

| Split/artifact | Images | Valid annotations | Category |
| --- | ---: | ---: | --- |
| train400 manifest | 400 | 524 raw annotations | Person metadata retained in source JSON |
| `forestpersons_train5000_oneclass_coco.json` | 5,000 | 6,415 | COCO category ID 1 |
| `forestpersons_val600_oneclass_coco.json` | 600 | 749 | COCO category ID 1 |

All train400 and train5000 manifest entries resolve by `image_id`, agree with
their recorded filenames, and have readable local images. The DEIM criterion
requires zero-based internal labels, so category ID 1 is mapped to label 0 at
runtime. For initialization, the COCO person row (internal class 0) is copied
to every one-class encoder/decoder score head and the denoising embedding;
the denoising padding row is zeroed. Shape-compatible pretrained parameters
are retained exactly.

## Gradient smoke results

Command prerequisites:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src \
  .venv-deimv2/bin/python scripts/smoke_deimv2_finetune_mps.py \
  --root data \
  --manifest outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json \
  --official-repo experiments/deimv2_local/official_DEIMv2 \
  --checkpoint-dir experiments/deimv2_local/checkpoints/DEIMv2_DINOv3_S_COCO \
  --output outputs/deimv2_local/finetune_smoke/deimv2_s_oneclass_mps_cpu_fallback_smoke.json \
  --device mps --input-size 640 --seed 20261015
```

| Arm | Trainable parameters | Total loss | Pre-clip grad norm | Changed tensors | Step time | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen distilled transformer; train STA/encoder/decoder | 4,202,225 | 30.1222 | 89.43 | 332 | 1.59 s | 1.34 GiB |
| Full model | 9,688,433 | 24.6675 | 590.49 | 479 | 0.88 s | 2.45 GiB |

The two losses are from different images and are not a model-quality
comparison. Every individual loss and every recorded gradient group was
finite. Global clipping at the official-style norm 0.1 was applied before the
optimizer step. The timing includes MPS graph/fallback warm-up and is only a
two-image engineering measurement. A separate CPU-only control also completed
both exact steps with finite losses and gradients.

Without CPU fallback, forward and loss calculation succeed on MPS, but
backward stops at the missing `aten::grid_sampler_2d_backward` operation. The
failure is preserved in `deimv2_s_oneclass_mps_smoke.log`.

## Feasible pilot configuration

For a first controlled train400 experiment:

- full one-class DEIMv2-S fine-tuning, because the measured 2.45 GiB driver
  allocation leaves substantial memory headroom;
- batch size 1 and gradient accumulation 4;
- float32, no mixed precision;
- `PYTORCH_ENABLE_MPS_FALLBACK=1` from process startup;
- AdamW with upstream S learning rates: 2.5e-5 for the distilled transformer,
  5e-4 for STA/encoder/decoder, betas 0.9/0.999, weight decay 1e-4, and zero
  decay for normalization/bias parameters;
- preserve upstream matching and loss weights: MAL 1, L1 5, GIoU 2, FGL 0.15,
  DDF 1.5;
- save a fresh one-class checkpoint and retain the original COCO checkpoint.

The upstream S schedule is 132 epochs with 2,000 warm-up iterations and heavy
Mosaic/Mixup/CopyBlend augmentation. It cannot be copied blindly into a short
train400 pilot. A two-epoch engineering pilot would contain 800 physical
samples and 200 accumulated optimizer steps. From the measured 0.88-1.59
seconds per smoke step, its model-compute floor is about 12-21 minutes;
allowing for data transforms, first-use compilation, evaluation, and I/O, a
practical estimate is 15-35 minutes. A meaningful 10-20 epoch fine-tune is
more plausibly 1.5-5 hours. These are estimates, not completed runs.

Before that pilot, a 20-image timing run should compare CPU-only against
hybrid MPS/CPU fallback: on this small model, fallback synchronization can
erase MPS's advantage.

## DEIMv2-L zero-shot outlook

The official DEIMv2-L is the variant that actually uses DINOv3 ViT-S/16. Its
published configuration has 32.2M parameters, 96.7 GFLOPs, 224-dimensional
multi-scale features, and the same four-layer query decoder. A float32
checkpoint should be roughly 123 MiB before packaging, approximately 3.3x the
local S checkpoint. No L weights were downloaded here.

Zero-shot L inference is likely feasible on this Mac because it does not need
the unsupported grid-sampler backward and the existing ViT-S and DEIMv2-S
inference paths already run on MPS. A cautious unverified estimate is 2-4 GiB
peak MPS driver memory and about 0.20-0.35 seconds per 640 x 640 image, based
on the 3.78x FLOP increase over S. A one-image then val32 smoke is required to
replace these estimates with measurements.

## Licensing

The pinned DEIMv2 source, architecture, and official weights are under the
DEIMv2 License, which permits academic/non-commercial research but prohibits
commercial use without a separate license from Intellindust. Because this is
a company internship, research evaluation should be kept clearly separate
from product use. LogXY must obtain authorization or a commercial license
before commercial deployment, integration, paid service, or redistribution.
Third-party DINOv3 and other dependency terms also remain applicable.

## Conclusion

Proceeding to a resource-bounded train400 pilot is technically justified.
The strongest starting point is full-model, one-class fine-tuning with batch
1, accumulation 4, float32, and explicit CPU fallback for deformable-attention
backward. The main risks are fallback synchronization cost, short-schedule
design, high pre-clip gradient norms, and the non-commercial license. This
feasibility result does not demonstrate training convergence or improved
ForestPersons metrics.

## Validation

`PYTHONPATH=src .venv-deimv2/bin/python -m pytest -q tests` completed with
105 passing tests in 3.83 seconds. Python syntax checks and JSON validation of
the saved MPS smoke summary also passed.

