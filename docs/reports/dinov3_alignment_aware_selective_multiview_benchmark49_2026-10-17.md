# DINOv3 Alignment-Aware Selective Multi-View Fusion on ForestPersons Benchmark49

## Scope

This is an inference-only exploratory comparison on the existing ForestPersons
benchmark49 development set. It uses the unchanged DINOv3 ViT-S/16 q/v LoRA +
stride-16 FCOS train5000 checkpoint. No model was trained, no data were
downloaded, and no detector threshold was tuned.

The experiment asks whether local correspondence gating can prevent the severe
damage caused by equal averaging of homography-warped source features. It does
not establish independent generalization because the benchmark49 failure was
used to design the selective rule.

## Prior failure and design

Equal feature averaging reduced fixed-threshold recall from 0.8706 to 0.5882.
It rescued 2 GT people but lost 26. The prior correspondence audit showed that
only 23/85 GT boxes aligned at IoU >= 0.50 in both source views: a homography
that fits static forest background does not guarantee correspondence for a
moving person, depth parallax, pose change, or dynamic occlusion.

The new rule was predeclared before running C:

- preserve the reference feature at every patch;
- warp source features with the existing accepted homographies and validity
  masks;
- require both pointwise and local 3x3 DINOv3 feature cosine agreement;
- assign zero weight below cosine 0.90 and linearly increase trust to 1.0 at
  cosine 0.98;
- require at least 0.75 local valid support and exact source-grid validity;
- cap the two sources' combined weight at 0.50, making the reference at least
  two-thirds of the normalized weighted mixture;
- use the unmodified reference feature when no source is reliable.

The gate uses no annotations, person boxes, visibility, or detector outputs.

## Protocol and integrity

- 49 reference images, 49 unique ForestPersons sequences, and 85 unique GT
  people.
- Visibility 20/40/70/100 counts: 12/15/22/36.
- Short side 672; fixed confidence 0.55; NMS IoU 0.50; matching IoU 0.50.
- A and B were loaded from frozen COCO prediction dumps. Recomputed recall,
  FP/image, AP, and AR@100 reproduced their saved values exactly before C ran.
- C reused the same encoder, FCOS head, preprocessing, homographies, reference
  images, and evaluation code.
- Sequence was the paired bootstrap resampling unit (10,000 replicates).

## A/B/C results

| Metric | A: single view | B: equal mean | C: selective |
|---|---:|---:|---:|
| TP / FP / FN | 74 / 12 / 11 | 50 / 21 / 35 | 75 / 11 / 10 |
| Recall | 0.8706 | 0.5882 | **0.8824** |
| Precision | 0.8605 | 0.7042 | **0.8721** |
| FP/image | 0.2449 | 0.4286 | **0.2245** |
| R20 | 0.6667 | 0.6667 | 0.6667 |
| R40 | 0.8000 | 0.6000 | 0.8000 |
| R70 | 0.9091 | 0.5909 | 0.9091 |
| R100 | 0.9444 | 0.5556 | **0.9722** |
| AP | 0.4142 | 0.2291 | **0.4183** |
| AP50 | 0.8778 | 0.6124 | **0.8971** |
| AP75 | **0.3099** | 0.0628 | 0.2758 |
| AR@100 | 0.5094 | 0.3600 | **0.5235** |

At the fixed operating point, C recovered all 26 A detections destroyed by B,
lost no A detections, and counted one additional GT. It also removed one false
positive relative to A. The one added GT is visibility 100, not an occluded
person. Its best IoU changed from 0.4888 (A) to 0.5062 (C), just crossing the
IoU 0.50 boundary. The gate weight at that GT center was zero, so this was an
indirect contextual shift rather than direct aligned evidence at the person.

The sequence-cluster bootstrap C-minus-A recall difference is +0.0118 with a
95% percentile interval of [0.0000, 0.0380]. The probability of a positive
difference is 0.631. R20, R40, and R70 differences are exactly zero. These data
do not establish a reliable recall improvement.

## Gate and alignment analysis

Across groups, 77.5% of reference patches received some source weight and
22.5% used exact reference-only fallback. Mean combined source weight was
0.365, below the 0.50 cap. At GT centers, 44/85 annotations had zero source
weight; the median was zero and the mean was 0.198.

C-minus-A best-candidate IoU increased for 45 GT and decreased for 40. The
mean change was +0.0051 and the median +0.0011, with a range from -0.0908 to
+0.2805. AP50 and AR@100 increased slightly, but AP75 fell by 0.0341. Local
feature agreement therefore protects the reference detector much better than
equal averaging, yet it is not a dependable target-correspondence signal.

## Runtime and memory

- A reference-only inference: 0.461 s/group in the original run.
- B full three-view pipeline: 1.513 s/group.
- C full three-view pipeline: 1.384 s/group (67.8 s total).
- C peak MPS driver allocation: 2.26 GiB, effectively unchanged from B.

Timing differences between B and C include normal run-to-run variance. Both
multi-view paths require three encoder passes and are about three times the
single-view latency.

## Decision

Selective fusion successfully prevents the destructive behavior of equal
averaging, but it does not demonstrate meaningful new multi-view value:

- no improvement at visibility 20, 40, or 70;
- only one borderline visibility-100 recovery;
- no independent confirmation;
- worse AP75 than single view;
- roughly threefold inference cost.

Further ForestPersons homography-gate tuning is not justified. The next
multi-view experiment should use AOS with camera poses, calibration/KML and
the official rectification mapping so correspondence can be geometry-aware.
The required official artifacts identified in the prior audit are
`RAW_lowresRGB.zip` (raw RGB, KML and rectification routine) and `YOLO.zip`
(labels, splits and integral images). They were not downloaded in this task.

## Artifacts

- Complete experiment:
  `outputs/multiview/forestpersons_val_benchmark49_selective_seed20261017_rerun1/`
- Reproducible evaluator:
  `scripts/evaluate_dinov3_selective_multiview_fusion.py`
- Figure/table generator:
  `scripts/summarize_dinov3_selective_multiview_fusion.py`
- Core selective fusion:
  `src/forestpersons/multiview.py`
- An earlier run completed inference but failed while serializing tensor-rich
  checkpoint metadata; its partial artifacts are preserved at
  `outputs/multiview/forestpersons_val_benchmark49_selective_seed20261017/`
  and are not the canonical result.

