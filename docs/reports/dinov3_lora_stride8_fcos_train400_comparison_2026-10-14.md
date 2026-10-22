# DINOv3 LoRA FCOS Stride-16 vs Stride-8 Train400

Date: 2026-10-14

## Scope and Protocol

This controlled pilot compared the original stride-16 DINOv3 ViT-S + LoRA +
FCOS detector against the learned stride-8 adapter implementation. Both runs:

- started from the same completed LoRA+FCOS train5000 checkpoint;
- used the same train400 manifest and val600 manifest;
- trained for two additional epochs at short side 672 on MPS;
- used batch size 1, gradient accumulation 4, seed 20261014;
- used AdamW, LoRA LR 1e-4, head LR 3e-4, weight decay 1e-4;
- used unchanged focal loss (alpha 0.75, gamma 2), Smooth L1 box loss,
  centerness BCE, and box-loss weight 2;
- used no LR scheduler, GIoU term, or visibility weighting.

A dedicated seeded DataLoader generator made image ordering independent of
head initialization. The per-epoch SHA-256 order digests match exactly across
the runs. Stride-dependent target assignment necessarily differs.

The stride-16 run loaded the source FCOS head strictly. The stride-8 run loaded
the same LoRA weights and every compatible FCOS tensor, while newly initializing
only the 26,400-parameter upsampling adapter.

## Training

| Model | Epoch losses | Positives/epoch | Steps | Runtime | Peak observed MPS driver memory |
| --- | --- | ---: | ---: | ---: | ---: |
| Stride 16 | 0.6181, 0.6099 | 4,140 | 200 | 910.3 s | 10.62 GiB |
| Stride 8 | 0.7311, 0.6518 | 4,697 | 200 | 822.1 s | 10.62 GiB |

All losses and gradients remained finite. Both LoRA and head tensors had
nonzero gradients. Stride 8 created 557 additional positives per epoch (+13.5%)
and reduced overlap overwrites from 16 to 6. Runtime differences may reflect
system/cache variation and are not evidence that stride 8 is faster.

## COCO Metrics on Selected Val600

| Model | AP | AP50 | AP75 | AR@100 | Inference s/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stride 16 control | 0.3503 | 0.8116 | 0.2421 | 0.4714 | 0.4471 |
| Stride 8 | 0.2188 | 0.6693 | 0.0576 | 0.3943 | 0.4543 |

These are selected-val600 pilot metrics, not official full-split benchmark
results.

## Recall and False Positives

The fixed inference configuration uses confidence 0.60 and NMS IoU 0.50.

| Model | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stride 16, conf 0.60 | 0.8211 | 0.7555 | 0.3317 | 0.6207 | 0.7824 | 0.8860 | 0.9575 |
| Stride 8, conf 0.60 | 0.7250 | 0.5934 | 0.6200 | 0.5287 | 0.6824 | 0.7565 | 0.8915 |

For exploratory matched-budget comparison, thresholds were selected on val600
from the same dense 0.005 grid. They are not independently validated.

| Model | Threshold | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stride 16 | 0.570 | 0.8438 | 0.7166 | 0.4167 | 0.6724 | 0.8000 | 0.9067 | 0.9623 |
| Stride 8 | 0.630 | 0.6662 | 0.6662 | 0.4167 | 0.4598 | 0.6176 | 0.7047 | 0.8396 |

At other FP budgets, stride-16 recall versus stride-8 recall was 0.7944 versus
0.5688 at <=0.25 FP/image, 0.8545 versus 0.6822 at <=0.50, and 0.8745 versus
0.7784 at <=1.0.

Maximum custom recall is defined here as IoU-0.50 recall at the minimum saved
confidence 0.001, after per-image top-300 filtering and NMS. It is not COCO AR
and it has no practical FP constraint:

- Stride 16: recall 0.9399 at 76.275 FP/image.
- Stride 8: recall 0.9159 at 52.177 FP/image.

Even the unconstrained saved-prediction ceiling is lower for stride 8.

## Recovered, Lost, and Box Size

At the exploratory equal point of 0.4167 FP/image, stride 8 recovered 22 GT
people missed by stride 16 but lost 155 previously detected people, a net loss
of 133. For visibility 20, it recovered 6 and lost 43, a net loss of 37.

Using COCO original-image box-area groups at the matched budget:

| Size | GT | Stride-16 recall | Stride-8 recall | Recovered | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small (<32^2) | 1 | 0.0000 | 0.0000 | 0 | 0 |
| Medium (32^2-96^2) | 210 | 0.7143 | 0.6905 | 17 | 22 |
| Large (>=96^2) | 538 | 0.8959 | 0.6580 | 5 | 133 |

At fixed confidence 0.60, stride 8 increased medium-object recall from 0.6619
to 0.7333, but FP/image increased from 0.3317 to 0.6200 and overall/R20 recall
still fell. The medium-object gain disappears at matched false positives.
Val600 has only one COCO-small GT, so it cannot support a reliable conclusion
about genuinely tiny people.

## Interpretation

Confirmed findings:

- The denser target grid increased training positives but did not improve AR,
  maximum recall, matched-budget recall, R20, AP, or AP75.
- Stride 8 produced substantially poorer localization (AP75 -0.1845).
- Its few complementary detections were outweighed by many lost detections.
- Any apparent medium-size gain at the fixed threshold required a large false-
  positive increase and did not survive matched-FP comparison.

Hypothesis, not directly proven: exact feature replication followed by only two
epochs of adapter learning may not create genuinely new fine spatial detail.
The reused stride-16 tower also begins poorly calibrated for the denser target
geometry. More data or training might change this, but the controlled pilot
provides no evidence that doing so is the best use of compute.

## Decision

The stride-8 adapter does **not** deserve train5000 scaling in its current form.
Retain the implementation and results as a negative controlled experiment. The
existing stride-16 LoRA+FCOS detector remains the stronger recall model, with
the optional RoI box refiner remaining the only tested component that improved
localization without a major recall loss.

Artifacts are under:

- `outputs/dinov3_lora_stride8_fcos/train400_control_stride16_seed20261014_from_lora5k_e2/`
- `outputs/dinov3_lora_stride8_fcos/train400_stride8_seed20261014_from_lora5k_e2/`
- `outputs/dinov3_lora_stride8_fcos/train400_stride16_vs_stride8_seed20261014_comparison/`
