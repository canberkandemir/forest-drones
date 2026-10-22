# DINOv3 LoRA + FCOS RoI Refiner Train400 Pilot

Date: 2026-10-14

## Experiment

This selected-subset pilot tested whether a small RoI stage improves the best
ForestPersons DINOv3 ViT-S + LoRA + single-scale FCOS detector. The source
backbone, LoRA adapters, and FCOS head were frozen. Only the 99,269-parameter
RoI refinement head was trained.

- Training data: deterministic ForestPersons train400 manifest.
- Evaluation data: unchanged ForestPersons val600 manifest, 749 GT persons.
- Input: short side 672, stride-16 DINOv3 patch features.
- FCOS proposal pool: top 1,000 pre-NMS candidates for training.
- RoI sample: 128 proposals/image, at most 32 positives.
- Matching: positive IoU >= 0.50, negative IoU < 0.30.
- Optimization: AdamW, learning rate 3e-4, weight decay 1e-4, batch size 1,
  two epochs, seed 20261014.
- Loss: BCE person/quality score plus positive-only Smooth L1 box deltas.

Three val600 modes were generated in one shared inference pass:

- **A: FCOS**: unchanged canonical FCOS decoding and scoring.
- **B: refined boxes**: learned box deltas, original FCOS proposal scores.
- **C: refined + rescored**: learned box deltas and
  `FCOS_score * RoI_probability` before NMS.

All modes used top-300 pre-NMS candidates, NMS IoU 0.50, the same GT matching,
and maxDets `[1, 10, 100]`. Confidence thresholds selected from val600 are
exploratory and are not independent validation results.

## Training

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Mean total loss | 0.7145 | 0.6120 |
| Classification loss | 0.1416 | 0.0778 |
| Box loss | 0.5728 | 0.5342 |
| Positive proposals | 9,896 | 9,896 |
| Negative proposals | 41,304 | 41,304 |
| Images without positives | 0 | 0 |
| Runtime | 384.3 s | 374.7 s |

Total training runtime was 759.0 seconds (12.65 minutes), with 800 optimizer
steps and 2.27 GiB peak MPS driver memory. Frozen-detector trainable parameters
were verified as zero.

## Detection results

| Mode | AP | AP50 | AP75 | AR@100 | Maximum recall | FP/image at maximum |
|---|---:|---:|---:|---:|---:|---:|
| A. FCOS | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.9212 | 76.493 |
| B. Refined boxes | **0.3666** | **0.8248** | **0.2719** | 0.4734 | 0.9212 | 75.558 |
| C. Refined + rescored | 0.3634 | 0.8153 | 0.2675 | **0.4737** | **0.9306** | 77.668 |

Maximum recall means IoU 0.50 recall at confidence threshold 0.0 after each
mode's NMS and 300-candidate limit. It is separate from COCO AR@100. Its very
high FP/image makes it an upper-bound diagnostic rather than a useful operating
point.

## Recall at matched FP budgets

| FP budget | Mode | Threshold | Recall | Precision | Actual FP/image |
|---:|---|---:|---:|---:|---:|
| 0.25 | FCOS | 0.615 | 0.7984 | 0.8016 | 0.2467 |
| 0.25 | Refined boxes | 0.590 | **0.8184** | 0.8045 | 0.2483 |
| 0.25 | Refined + rescored | 0.445 | 0.7984 | 0.8005 | 0.2483 |
| 0.425 | FCOS | 0.550 | 0.8465 | 0.7172 | 0.4167 |
| 0.425 | Refined boxes | 0.530 | **0.8545** | 0.7183 | 0.4183 |
| 0.425 | Refined + rescored | 0.360 | 0.8158 | 0.7088 | 0.4183 |
| 0.50 | FCOS | 0.530 | 0.8545 | 0.6816 | 0.4983 |
| 0.50 | Refined boxes | 0.510 | **0.8598** | 0.6837 | 0.4967 |
| 0.50 | Refined + rescored | 0.330 | 0.8251 | 0.6784 | 0.4883 |
| 1.00 | FCOS | 0.460 | **0.8745** | 0.5261 | 0.9833 |
| 1.00 | Refined boxes | 0.445 | 0.8718 | 0.5262 | 0.9800 |
| 1.00 | Refined + rescored | 0.240 | 0.8491 | 0.5171 | 0.9900 |

Box-only refinement gives its largest recall gain at the stricter 0.25 FP
budget (+0.0200). The gain is +0.0080 at the target 0.425 budget and disappears
at the 1.0 budget.

## Visibility recall at FP/image <= 0.425

| Mode | R20 | R40 | R70 | R100 |
|---|---:|---:|---:|---:|
| FCOS | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Refined boxes | **0.6724** | **0.8176** | **0.9223** | 0.9717 |
| Refined + rescored | 0.5690 | 0.7471 | **0.9223** | **0.9764** |

The box-only R20 gain is one net GT detection (+0.0057). RoI rescoring loses
substantial low-visibility recall at the matched FP budget, indicating that the
score branch suppresses occluded people.

## Recovered and lost detections

At each mode's exploratory FP/image <= 0.425 operating point, box refinement
recovers 14 GT instances missed by FCOS and loses 8 FCOS detections, a net gain
of 6. The recovered/lost counts by visibility are:

| Visibility | Recovered | Lost | Net |
|---:|---:|---:|---:|
| 20 | 4 | 3 | +1 |
| 40 | 5 | 3 | +2 |
| 70 | 3 | 2 | +1 |
| 100 | 2 | 0 | +2 |

At the same fixed FCOS threshold of 0.55, box refinement recovers 8 and loses 8
GT instances. The matched-budget gain therefore combines changed geometry/NMS
with the ability to lower the threshold from 0.55 to 0.53 while staying within
the FP budget.

Rescoring recovers 18 but loses 41 GT instances at the matched budget, a net
loss of 23. Twenty of the lost detections are visibility 20. Its higher maximum
recall occurs only at an unconstrained 77.7 FP/image.

## Conclusion

The lightweight second stage improves localization: AP75 increases by 0.0340,
AP by 0.0079, and AR@100 by 0.0031 with original proposal scores. It does not
materially improve the project's primary low-visibility recall target: matched
R20 improves by only one GT instance, and maximum recall is unchanged.

The learned person/quality score should not replace FCOS scores in its current
form because it suppresses heavily occluded true positives. These train400
results do not justify scaling the complete refiner to train5000 solely for
recall. If localization quality becomes the next priority, the box-only branch
is worth retaining; for recall under vegetation, effort should instead target
proposal generation or finer-resolution features.

## Artifacts

- Training checkpoint and history:
  `outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k/`.
- Evaluation summary and comparison CSV:
  `outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k/eval_val600/`.
- Persistent logs:
  `outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k_run.log` and
  `outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k_eval_val600.log`.
