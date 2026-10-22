# DINOv3 ViT-S LoRA + FPN FCOS Train400 Pilot

Date: 2026-10-13

## Scope

This was the first bounded training pilot for the lightweight FPN-style FCOS
head on top of the existing DINOv3 ViT-S + LoRA detector. It used only the
existing ForestPersons train400 subset and evaluated on the existing val600
subset. No train5000 or longer training was started.

## Configuration

- Dataset: ForestPersons train400.
- Evaluation: ForestPersons val600.
- Resolution: short side 672.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Device: Apple MPS.
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Head: lightweight FPN FCOS with strides 16, 32, and 64.
- Checkpoint initialization:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

## Initialization Details

Loaded from the existing trained ViT-S LoRA + single-scale FCOS train5000
checkpoint:

- LoRA adapter weights.
- Compatible FCOS head weights:
  - `proj.*`
  - `shared.*`
  - `classification.*`
  - `box.*`
  - `centerness.*`

Newly initialized:

- `pyramid_adapters.medium.*`
- `pyramid_adapters.coarse.*`

Trainable parameters:

- LoRA: 73,728.
- FPN FCOS head: 986,406.
- Total trainable: 1,060,134.

## Training Results

| Epoch | Loss | Cls | Box | Centerness | Positive cells | Optimizer steps | Seconds | MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2371 | 0.0109 | 0.1317 | 0.9628 | 4,245 | 100 | 569.0 | 10.56 GiB |
| 2 | 1.0790 | 0.0048 | 0.0712 | 0.9319 | 4,245 | 200 | 435.2 | 10.56 GiB |

Total training time: 1004.4 seconds, approximately 16.7 minutes.

No out-of-memory error occurred.

## Val600 Metrics

Evaluation used the canonical ForestPersons framework with COCO-style metrics and
the existing confidence-threshold sweep. The matched-FP row is the best recall
among evaluated thresholds with FP/image <= 0.425.

| Model | AP | AP50 | AP75 | AR@100 | Conf | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-scale LoRA + FCOS baseline | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| FPN LoRA + FCOS train400 pilot | 0.3062 | 0.7836 | 0.1622 | 0.4579 | 0.60 | 0.8024 | 0.7365 | 0.3583 | 0.6149 | 0.7471 | 0.8964 | 0.9151 |

## Conclusion

The FPN pilot trained successfully, but it did not improve the target metrics.

- AP decreased from 0.3587 to 0.3062.
- AP75 decreased from 0.2379 to 0.1622.
- Recall at FP/image <= 0.425 decreased from 0.8465 to 0.8024.
- Recall at 20% visibility decreased from 0.6667 to 0.6149.
- Precision at the matched-FP operating point increased slightly, but recall
  dropped.

This suggests that simply adding the lightweight FPN adapters and training for
2 epochs on train400 is not enough. The new pyramid levels likely need either a
better initialization/training schedule, more data, loss rebalancing across
levels, or a revised assignment strategy. Based on this pilot alone, the current
single-scale ViT-S LoRA + FCOS remains the stronger DINOv3 detector.

## Artifacts

- Checkpoint:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_fcos.pt`
- Training summary:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/summary.json`
- Evaluation summary:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/summary.json`
- Comparison CSV:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/comparison_single_scale_baseline.csv`

## Recommended Next Step

Do not scale this exact FPN setup to train5000 yet. First inspect its false
positives and level usage, then either tune level assignment/loss weighting or
try a safer single-scale improvement such as localization-loss refinement aimed
at AP75.
