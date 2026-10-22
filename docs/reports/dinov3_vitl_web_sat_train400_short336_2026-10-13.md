# DINOv3 ViT-L Web vs Satellite LoRA + FCOS Train400 Pilot

Date: 2026-10-13

## Scope

This is a reduced-resolution ForestPersons pilot for Davide's Phase 3 question: compare web-pretrained and satellite-pretrained DINOv3 weights while keeping the detector training protocol matched.

This is not a full ForestPersons benchmark and should not be compared directly to the earlier ViT-S short-side-672 train5000 experiments as a model-ranking claim. The run used only train400, short-side 336, and a randomly initialized FCOS head for both ViT-L models.

## Models

| Role | Checkpoint |
| --- | --- |
| Web ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| Satellite ViT-L/16 | `facebook/dinov3-vitl16-pretrain-sat493m` |

Both checkpoints use ViT-L/16 with 1024-dimensional patch features. The FCOS head was parameterized for 1024 input channels.

## Matched Training Protocol

- Dataset: existing ForestPersons train400 subset.
- Validation: existing val600 subset.
- Train/validation overlap: none from the existing verified manifests.
- Resolution: short side 336.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- FCOS head initialization: random, with matched `--head-init-seed 20261013`.
- LoRA: rank 4, alpha 16, dropout 0.05.
- LoRA targets: q/v attention projections in all 24 ViT-L transformer layers.
- Trainable parameters: 1,020,262 total, consisting of 393,216 LoRA parameters and 627,046 FCOS-head parameters.
- Original DINOv3 backbone weights remained frozen.
- Optimizer and loss settings reused the existing LoRA + FCOS training script defaults.

No ViT-S head weights were reused because the ViT-S head has 384 input channels and the ViT-L head has 1024 input channels.

## Training Results

| Model | Epoch 1 loss | Epoch 2 loss | Optimizer steps | Total time | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.9464 | 0.7823 | 200 | 640.4 s | 7.26 GiB |
| Satellite ViT-L | 0.9879 | 0.8291 | 200 | 635.2 s | 7.26 GiB |

Both runs completed with finite losses and nonzero LoRA/head gradients. No short-side-672 training was attempted in this experiment because the previous bounded smoke test already hit MPS out-of-memory at that resolution.

## Val600 Detection Metrics

Evaluation used the canonical ForestPersons evaluation framework on the same val600 subset, with IoU 0.50 for the custom threshold metrics and COCO-style AP over IoU 0.50:0.95. The prediction confidence floor was 0.001 before threshold sweeping, and NMS IoU was 0.50.

| Model | AP | AP50 | AP75 | AR@100 | Eval sec/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.1409 | 0.4562 | 0.0332 | 0.2987 | 0.345 |
| Satellite ViT-L | 0.0656 | 0.2745 | 0.0085 | 0.2435 | 0.354 |

## Matched FP/Image Budget

Closest measured operating point to 0.425 FP/image:

| Model | Confidence | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.60 | 0.4379 | 0.5857 | 0.3867 | 0.2586 | 0.3412 | 0.4870 | 0.6179 |
| Satellite ViT-L | 0.65 | 0.3057 | 0.4862 | 0.4033 | 0.1322 | 0.2176 | 0.3472 | 0.4811 |

At the fixed confidence threshold 0.60:

| Model | Recall | Precision | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.4379 | 0.5857 | 0.3867 | 0.2586 |
| Satellite ViT-L | 0.3778 | 0.3980 | 0.7133 | 0.2011 |

## Interpretation

In this reduced-resolution train400 pilot, the web-pretrained ViT-L checkpoint outperformed the satellite-pretrained ViT-L checkpoint on AP, AP50, AP75, AR@100, matched-FP recall, and low-visibility recall.

This should be interpreted cautiously. The result shows that the satellite checkpoint did not help under this small-data, low-resolution, RGB-UAV ForestPersons setting. It does not prove that web pretraining is intrinsically better for the full research problem. Important confounders remain:

- ForestPersons RGB UAV imagery may be visually closer to web/natural-image pretraining than to satellite pretraining.
- Short-side 336 reduces small-person detail and is lower than the canonical ViT-S short-side-672 experiments.
- The FCOS heads were randomly initialized and trained for only 2 epochs on 400 images.
- The satellite model's pretraining domain may require different resolution, augmentation, or training budget to be useful.

## Artifacts

- Web checkpoint: `outputs/dinov3_vitl_lora_train400/web_vitl_short336_seed20261013_e2/dinov3_lora_fcos.pt`
- Satellite checkpoint: `outputs/dinov3_vitl_lora_train400/sat_vitl_short336_seed20261013_e2/dinov3_lora_fcos.pt`
- Web evaluation: `outputs/dinov3_vitl_lora_train400/web_vitl_short336_seed20261013_e2/eval_val600_short336/summary.json`
- Satellite evaluation: `outputs/dinov3_vitl_lora_train400/sat_vitl_short336_seed20261013_e2/eval_val600_short336/summary.json`
- Comparison CSV: `outputs/dinov3_vitl_lora_train400/comparison_short336_seed20261013/summary_metrics.csv`

## Conclusion

The matched train400 short-side-336 pilot is complete. Web ViT-L is the stronger of the two ViT-L checkpoints in this constrained setting. The next scientifically clean step would be either a short-side-336 train5000 comparison if time allows, or moving a short-side-672 ViT-L comparison to a CUDA machine with more memory.
