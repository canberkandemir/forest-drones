# DEIMv2-S Zero-Shot ForestPersons Benchmark

Date: 2026-10-15

## Scope

This experiment evaluates the official COCO-pretrained DEIMv2-S checkpoint on
the existing ForestPersons val32 smoke subset and selected val600 pilot subset.
No model training, fine-tuning, checkpoint conversion, or dataset download was
performed. Results are zero-shot and are not an official ForestPersons
benchmark.

Primary sources:

- Paper: [Real-Time Object Detection Meets DINOv3](https://arxiv.org/abs/2509.20787)
- Official code: [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
- Official weights: [Intellindust/DEIMv2_DINOv3_S_COCO](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO)
- Transformers documentation: [DEIMv2](https://huggingface.co/docs/transformers/model_doc/deimv2)

## Exact model and loading path

- Official repository commit: `1d2ca42171570c713e78fc6a766ec5104b7f4724`.
- Checkpoint repository revision: `cf0540f3f319bb8ecbe132624358a35a7beb86d5`.
- Checkpoint: `Intellindust/DEIMv2_DINOv3_S_COCO`.
- Measured parameter count: 9,779,838.
- Weight file: 39.4 MB (`model.safetensors`).
- Weight SHA-256: `fe545b150766b8761a696f7b1d92ea138fafe8e02a7398823e0c62d2a5867956`.
- Official model-zoo claims: 50.9 COCO AP, 9.7M parameters, and 25.6 GFLOPs.

DEIMv2-S does **not** use the same backbone as our Hugging Face DINOv3
ViT-S/16. It uses the project's 192-dimensional ViT-Tiny distilled from
DINOv3-S, followed by a Spatial Tuning Adapter, three feature levels at strides
8/16/32, a hybrid encoder, and a four-layer 300-query DEIM transformer decoder.
DEIMv2-L is the variant that uses DINOv3 ViT-S.

Transformers now contains a DEIMv2 implementation, but the official
Intellindust S repository is a custom `PyTorchModelHubMixin` artifact rather
than a Transformers `PreTrainedModel`; `AutoConfig.from_pretrained` failed on
this exact repository. The benchmark therefore uses the official wrapper from
`hf_models.ipynb` and the original checkpoint without conversion. This is more
faithful than substituting a community Transformers checkpoint.

The DEIMv2 license permits academic/research evaluation but forbids commercial
use without a separate license. Any later LogXY product use requires a license
review.

## Local environment

The experiment is isolated in `.venv-deimv2`; `.venv-dinov3` and `.venv-yolo`
were not changed.

| Component | Version / value |
|---|---|
| macOS hardware | Apple M4 MacBook Pro, 16 GB unified memory |
| Python | 3.13.5 |
| PyTorch | 2.9.0 |
| torchvision | 0.24.0 |
| Device | MPS; CPU fallback was not needed |
| Official input | fixed 640 x 640 resize, no letterboxing |
| Normalization | ImageNet mean/std from the official ViT config |
| Output | 300 ranked query/class pairs; no NMS |
| Person mapping | zero-based COCO label 0 to ForestPersons category 1 |

The Hub client downloaded the weight file but hit an `httpx2` decompression
error while fetching the small config. The config was retrieved from the same
pinned Hugging Face revision with normal HTTPS; SSL verification remained
enabled. File hashes were recorded.

## Validation checks

- Boxes from the official postprocessor are in original-image `xyxy` pixels.
- Original sizes are passed as `[width, height]`, matching upstream inference.
- Boxes are clipped independently to original width and height before export.
- All 600 val600 image IDs and 749 visibility-labelled person annotations were
  evaluated.
- The 14,381 person-class predictions were exported as one-class COCO results.
- Local AP/AR and pycocotools values match to floating-point precision.
- The official query-detector inference remains NMS-free.
- Focused preprocessing, class mapping, clipping, and formatting tests pass.

## Results

### Smoke subsets

| Subset | Images | GT | AP | AP50 | AP75 | AR@100 | Recall at FP/image <= 0.425 |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke4 | 4 | 4 | 0.2389 | 0.2938 | 0.2574 | 0.3250 | 0.2500 at 0.250 FP/image |
| val32 | 32 | 48 | 0.3656 | 0.6232 | 0.3397 | 0.5854 | 0.4583 at 0.375 FP/image |

The val32 subset is intentionally balanced by target visibility and is only a
smoke test.

### Val600 pilot

The operating point below is the highest measured recall with FP/image no more
than 0.425 in the deterministic threshold sweep. Its threshold was selected on
val600 and is exploratory, not independently validated.

| Metric | Value |
|---|---:|
| AP@[0.50:0.95] | 0.3408 |
| AP50 | 0.5693 |
| AP75 | 0.3544 |
| AR@1 | 0.3742 |
| AR@10 | 0.5746 |
| AR@100 | 0.6016 |
| Operating confidence | 0.175 |
| Recall at IoU 0.50 | 0.5033 |
| Precision | 0.6294 |
| FP/image | 0.3700 |
| R20 | 0.1782 |
| R40 | 0.4059 |
| R70 | 0.5907 |
| R100 | 0.7689 |

At score 0, maximum IoU-0.50 recall over the saved person queries is 0.9146,
but it costs 22.83 FP/image. This threshold-specific maximum recall is not
COCO AR. It shows that useful boxes often exist but are poorly calibrated for
the ForestPersons domain.

### Comparison with trained references

| Model | ForestPersons training | AP | AP50 | AP75 | AR@100 | Recall | Precision | FP/image | R20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s fine-tuned train5000 | yes | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.9039 | 0.7650 | 0.3467 | 0.7701 |
| DINOv3 ViT-S LoRA + FCOS train5000 | yes | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8465 | 0.7172 | 0.4167 | 0.6667 |
| DEIMv2-S COCO zero-shot | no | 0.3408 | 0.5693 | 0.3544 | 0.6016 | 0.5033 | 0.6294 | 0.3700 | 0.1782 |

The comparison is deliberately not presented as matched training. YOLO11s and
our FCOS detector saw 5,000 ForestPersons training images; DEIMv2-S saw none.

## Localization and confidence interpretation

DEIMv2-S is not a stronger deployed detector at the zero-shot operating point.
Its recall and especially R20 are substantially below both trained references.
However, it is a promising **foundation**:

- AR@100 is 0.6016, close to fine-tuned YOLO11s (0.6339) and well above our
  custom FCOS detector (0.4704).
- Its AP75/AP50 ratio is 0.623, versus 0.292 for our FCOS model. Conditional on
  recognizing a person, its localization is markedly stronger.
- Its maximum low-threshold IoU-0.50 recall is high, but the false-positive cost
  is unacceptable. The dominant zero-shot limitation is ForestPersons-domain
  classification/confidence calibration, not an absence of plausible boxes.
- Fully visible people are much easier (R100 0.7689) than 20%-visible people
  (R20 0.1782), confirming that COCO pretraining does not solve heavy vegetation
  occlusion by itself.

## Runtime and memory

| Measurement | Value |
|---|---:|
| Val600 model inference time | 47.63 s |
| Steady mean latency | 0.0790 s/image |
| Steady median latency | 0.0776 s/image |
| Steady p95 latency | 0.0875 s/image |
| Throughput from steady mean | about 12.7 images/s |
| Observed peak MPS driver allocation | 1.16 GiB |
| Observed peak process RSS | 0.86 GiB |

The very first standalone MPS call took 2.35 s due to initialization/compilation;
subsequent fresh-script first calls were about 0.33 s. The steady-state values
are the useful local inference estimate. These are local MPS measurements, not
comparable to the official model-zoo GPU latency.

## Decision

1. **Is pretrained DEIMv2-S promising?** Yes as a localization and proposal
   foundation, but no as an immediately deployable zero-shot ForestPersons
   detector.
2. **Does it localize or rank better?** Localization is substantially better
   than our FCOS head by AP75/AP50 and AR. Confidence ranking/calibration under
   vegetation is much worse at the operational FP budget.
3. **Is DEIMv2-L justified?** A resource-bounded val32 smoke is scientifically
   justified because L uses DINOv3 ViT-S rather than the distilled ViT-Tiny in
   S, making it closer to our backbone comparison. It should remain zero-shot
   and gated on local MPS feasibility before val600.
4. **Is ForestPersons fine-tuning worth pursuing?** Yes. The high zero-shot
   AR ceiling makes a controlled train400 fine-tuning pilot higher-value than
   further modifications to the custom FCOS head. Licensing must be resolved
   before any commercial use.

Recommended next experiment: a separate DEIMv2-S train400 fine-tuning
feasibility smoke and short pilot, using the same val600 metrics. Do not scale
to train5000 until MPS backward memory and training stability are measured.

## Artifacts

- `src/forestpersons/deimv2_adapter.py`
- `scripts/evaluate_deimv2_forestpersons.py`
- `experiments/deimv2_local/requirements-macos.txt`
- `experiments/deimv2_local/official_DEIMv2/`
- `experiments/deimv2_local/checkpoints/DEIMv2_DINOv3_S_COCO/`
- `outputs/deimv2_zero_shot/smoke4_mps/`
- `outputs/deimv2_zero_shot/val32_mps/`
- `outputs/deimv2_zero_shot/val600_mps/`
- `outputs/deimv2_zero_shot/val600_comparison.csv`
