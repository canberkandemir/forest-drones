# DINOv3 Full MVR Label-Efficiency Results

Date: 2026-10-23
Status: completed on nested ForestPersons train100/400/1000/5000 and val600

## Experiment

Four Full MVR + FCOS heads were trained from the same deterministic
initialization for 20 epochs on exact nested prefixes of a fixed train5000
pool. DINOv3 remained frozen and existing block-3/6/9/12 feature caches were
reused. The unchanged val600 protocol used IoU 0.50 one-to-one matching,
candidate floor 0.001, NMS IoU 0.50, top-300 decoding, COCO AP/AR, and the
predeclared confidence sweep.

All checkpoints record the same initialization hashes:

- FCOS: `5ef06d7af762954d4e9cf8bc89a6c5545136d4e7f39488bf40ab9a346ff7a2d5`
- Full MVR: `346a513dd0453e645c06002340c9c062e871e9b7d750646f17211dff25125ef6`

Each checkpoint completed 20/20 epochs, matched its frozen manifest SHA-256,
and had finite loss histories.

## Training

| Images | Optimizer steps | First loss | Final loss | Runtime | Peak MPS driver memory |
|---:|---:|---:|---:|---:|---:|
| 100 | 2,000 | 1.2676 | 0.6392 | 2 m 45 s | 1.09 GiB |
| 400 | 8,000 | 1.1564 | 0.6341 | 10 m 12 s | 1.10 GiB |
| 1,000 | 20,000 | 1.0626 | 0.6287 | 27 m 45 s | 1.10 GiB |
| 5,000 | 100,000 | 0.8856 | 0.6232 | 2 h 15 m 19 s | 1.10 GiB |

## Threshold-independent metrics

| Training images | AP | AP50 | AP75 | AR@100 |
|---:|---:|---:|---:|---:|
| 100 | 0.0844 | 0.3083 | 0.0151 | 0.2430 |
| 400 | 0.1735 | 0.5280 | 0.0547 | 0.3387 |
| 1,000 | 0.2673 | 0.7068 | 0.1290 | 0.4099 |
| 5,000 | **0.3353** | **0.7756** | **0.2169** | **0.4780** |

Performance improves monotonically at every scale. Incremental AP gains are
`+0.0890` (100 to 400), `+0.0938` (400 to 1,000), and `+0.0680` (1,000 to
5,000). AP50 begins to saturate after 1,000 images (`+0.0688`), while AP75
continues to gain substantially (`+0.0880`), indicating that additional labels
still improve localization quality.

## Recall at matched false-positive budgets

Each checkpoint selects its threshold separately from the same val600 sweep.
These operating points are exploratory because val600 labels determine the
threshold.

| Images | Budget | Threshold | Recall | R20 | Actual FP/image |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.150 | 0.64 | 0.1896 | 0.0805 | 0.1300 |
| 400 | 0.150 | 0.68 | 0.3632 | 0.1494 | 0.1383 |
| 1,000 | 0.150 | 0.70 | 0.6075 | 0.3333 | 0.1467 |
| 5,000 | 0.150 | 0.73 | **0.6168** | **0.3621** | 0.1367 |
| 100 | 0.250 | 0.60 | 0.2684 | 0.1092 | 0.2467 |
| 400 | 0.250 | 0.64 | 0.4686 | 0.1782 | 0.2200 |
| 1,000 | 0.250 | 0.66 | 0.6702 | 0.3793 | 0.2467 |
| 5,000 | 0.250 | 0.66 | **0.7717** | **0.5115** | 0.2467 |
| 100 | 0.350 | 0.58 | 0.3071 | 0.1264 | 0.3417 |
| 400 | 0.350 | 0.61 | 0.5247 | 0.2184 | 0.3317 |
| 1,000 | 0.350 | 0.63 | 0.7170 | 0.4425 | 0.3400 |
| 5,000 | 0.350 | **0.8104** | **0.5690** | 0.3300 |
| 100 | 0.425 | 0.57 | 0.3231 | 0.1264 | 0.3967 |
| 400 | 0.425 | 0.59 | 0.5607 | 0.2644 | 0.4167 |
| 1,000 | 0.425 | 0.62 | 0.7250 | 0.4713 | 0.3867 |
| 5,000 | 0.425 | 0.60 | **0.8224** | **0.5977** | 0.4017 |

At the 0.425 budget, overall recall rises by 0.2377, 0.1642, and 0.0975 across
successive scales. R20 rises from 0.1264 to 0.2644 to 0.4713 to 0.5977. The
largest low-visibility gain occurs between 400 and 1,000 images, but 5,000
labels still add 0.1264 R20.

Visibility-specific recall at the 0.425 budget:

| Images | R20 | R40 | R70 | R100 | Recall | FP/image |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.1264 | 0.2765 | 0.4197 | 0.4340 | 0.3231 | 0.3967 |
| 400 | 0.2644 | 0.4647 | 0.6425 | 0.8066 | 0.5607 | 0.4167 |
| 1,000 | 0.4713 | 0.6353 | 0.8135 | 0.9245 | 0.7250 | 0.3867 |
| 5,000 | **0.5977** | **0.7706** | **0.9171** | **0.9623** | **0.8224** | 0.4017 |

## Interpretation

The controlled curve gives strong evidence that Full MVR benefits from more
labeled ForestPersons images. Gains are monotonic in AP, AP50, AP75, AR@100,
matched-budget recall, and every visibility stratum. The model is not saturated
for heavily occluded people at 5,000 images: R20 and AP75 continue to improve.

The 1,000-image subset already covers all 258 train5000 sequence folders, while
the 5,000 arm adds more frames within those sequences. Therefore, the final gain
reflects denser within-sequence supervision as well as more labeled images, not
5x more independent scenes.

## Limitations

1. This is one selection seed and one training seed. Additional seeds are needed
   for variance estimates and a defensible label-efficiency curve.
2. Fixed 20-epoch training means 2,000 to 100,000 optimizer steps. The study
   measures practical fixed-epoch scaling and does not isolate label count from
   optimization budget.
3. val600 is a selected pilot subset that has been inspected repeatedly. It is
   not the official full ForestPersons validation benchmark, and matched-budget
   thresholds are selected on it.
4. Target-image visibility is exactly balanced, but annotation counts are not,
   because images can contain multiple people.
5. Only the MVR readout and FCOS head are trained; DINOv3 features remain frozen.
   These results do not measure end-to-end or LoRA label efficiency.
6. The matched train5000 AP is 0.3353 versus 0.3623 in the earlier Full MVR run
   using the same image set but a different manifest order. This 0.0270 gap
   demonstrates order/optimization sensitivity and is another reason not to
   generalize from a single run.

## Decision

The first-seed trend is meaningful and supports the roadmap condition for
running additional seeds. The highest-value continuation would be seeds 2 and
3 for all four nested subsets, without changing architecture or evaluation.
That recommendation is not executed here.

## Artifacts

- Subset audit: `outputs/subsets/label_efficiency/label_efficiency_nested_seed20261023_audit.json`
- Checkpoints and histories: `outputs/dinov3_mvr/label_efficiency/train*_mvr_e20_seed20261012_nested20261023/`
- Evaluation: `outputs/dinov3_mvr/label_efficiency/evaluation_val600_seed20261012_nested20261023/`
- Readiness report: `docs/reports/dinov3_mvr_label_efficiency_readiness_2026-10-23.md`
