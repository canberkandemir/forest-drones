# DINOv3 MVR Native Readout Ablation Results

Date: 2026-10-23
Status: completed on ForestPersons train5000/val600

## Question

The completed full MVR readout combines two changes: intermediate DINOv3 layers
and fine/native/coarse spatial context. This ablation inserts `mvr_native`, which
uses blocks 3/6/9/12 only at their native stride-16 grid, between the final-layer
baseline and full MVR.

All arms use the same frozen DINOv3 ViT-S/16 features, train5000/val600 manifests,
FCOS loss and decoding, 20 epochs, AdamW (`lr=3e-4`, `weight_decay=1e-4`), and
seed 20261012. Their trainable parameter counts are 524,646 (baseline), 721,894
(`mvr_native`), and 923,622 (full MVR).

## Training

`mvr_native` completed 20/20 epochs on Apple MPS. Loss fell from 0.85827 to
0.61632. The measured sum of epoch runtimes was 7,511.16 seconds (2 h 05 m
11 s). Every epoch saved an atomic checkpoint; the final checkpoint contains
all 20 history rows. Positive assignments were stable at 50,525 per epoch.

## COCO-style val600 metrics

| Readout | AP | AP50 | AP75 | AR@100 | Cached head time/image |
|---|---:|---:|---:|---:|---:|
| Final layer baseline | 0.3267 | 0.7915 | 0.2022 | 0.4550 | 6.43 ms |
| Native multi-layer | 0.3303 | 0.7926 | 0.2082 | 0.4615 | 14.34 ms |
| Full multi-resolution MVR | **0.3623** | **0.8162** | **0.2675** | **0.4900** | 18.55 ms |

Native multi-layer readout improved baseline AP by 0.0036, AP75 by 0.0060, and
AR@100 by 0.0065. Adding the fine/coarse branches on top of native multi-layer
features added a further 0.0320 AP, 0.0593 AP75, and 0.0284 AR@100. Most of the
full MVR gain therefore comes from multi-resolution spatial context, not merely
reading intermediate layers at the original patch grid.

## Matched false-positive budgets

Thresholds are selected separately from the same predeclared sweep and must be
treated as exploratory because val600 labels are used to select the operating
point.

| Budget | Readout | Threshold | Recall | R20 | R40 | R70 | R100 | Actual FP/image |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | Baseline | 0.68 | 0.6916 | 0.4425 | 0.6176 | 0.7772 | 0.8774 | 0.1383 |
| 0.15 | Native | 0.70 | 0.6796 | 0.4368 | 0.5882 | 0.7617 | 0.8774 | 0.1317 |
| 0.15 | Full MVR | 0.70 | **0.7210** | **0.4885** | **0.6471** | **0.8135** | **0.8868** | 0.1450 |
| 0.25 | Baseline | 0.63 | 0.7757 | **0.5575** | 0.7059 | 0.8549 | **0.9387** | 0.2483 |
| 0.25 | Native | 0.65 | 0.7410 | 0.5172 | 0.6471 | 0.8187 | 0.9292 | 0.2267 |
| 0.25 | Full MVR | 0.66 | **0.7810** | 0.5517 | **0.7235** | **0.8912** | 0.9151 | 0.2300 |
| 0.35 | Baseline | 0.60 | 0.8011 | 0.5862 | 0.7647 | 0.8705 | 0.9434 | 0.3500 |
| 0.35 | Native | 0.61 | **0.8131** | **0.6322** | 0.7235 | 0.8705 | **0.9811** | 0.3250 |
| 0.35 | Full MVR | 0.63 | 0.8077 | 0.5862 | **0.7765** | **0.9016** | 0.9292 | 0.3183 |
| 0.425 | Baseline | 0.59 | 0.8198 | 0.6207 | 0.7647 | 0.8912 | 0.9623 | 0.4000 |
| 0.425 | Native | 0.58 | **0.8318** | **0.6322** | 0.7706 | 0.8964 | **0.9858** | 0.4183 |
| 0.425 | Full MVR | 0.61 | 0.8278 | 0.6092 | **0.8118** | **0.9119** | 0.9434 | 0.3867 |

The native arm ranks best for recall at the two higher budgets, but it is worse
at 0.15 and 0.25. Full MVR is consistently strongest in AP/AR and at the strict
0.15 budget. This difference indicates score calibration/ranking effects; a
single selected threshold should not replace the threshold-independent result.

## Conclusion

Intermediate layers at native resolution alone provide a real but small
threshold-independent benefit. The large AP75 and AR@100 improvements require
the full fine/native/coarse readout, supporting the hypothesis that spatial
multi-resolution context is the main useful MVR component. `mvr_native` is not
a replacement for full MVR, although its higher exploratory recall at the
0.35/0.425 budgets is worth noting.

These are selected-val600 results, not an independent benchmark. No checkpoint
was selected or retrained from val600 performance.

An independent cached-feature evaluation rerun on 2026-10-23 reproduced all
three COCO prediction files, the full confidence sweep, and matched-budget CSV
byte-for-byte. Metric JSON values were also exact after excluding runtime-only
fields. The rerun used a separate output directory and changed no checkpoint.

## Artifacts

- Training: `outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012/`
- Evaluation: `outputs/dinov3_mvr/evaluation_val600_readout_ablation_seed20261012/`
- Deterministic verification: `outputs/dinov3_mvr/evaluation_val600_readout_ablation_seed20261012_rerun1/`
- Implementation report: `docs/reports/dinov3_mvr_native_ablation_implementation_2026-10-22.md`

`train_history.json` and the history embedded in `dinov3_head.pt` are the
authoritative training records. A duplicate monitoring-side launch was rejected
by the non-overwrite guard and contaminated only the convenience `tee` log; the
non-overwriting `verified_run_summary.json` records this provenance issue.
