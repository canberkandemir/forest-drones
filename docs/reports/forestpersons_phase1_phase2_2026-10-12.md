# ForestPersons Phase 1-2 Report - 2026-10-12

## 1. Research Objective And Scope

The LogXY internship objective is to develop and evaluate DINOv3-based methods
for detecting partially or heavily occluded people in UAV/MAV-style forest
imagery, especially under dense vegetation.

This report finalizes the ForestPersons-only status for:

- Phase 1: dataset preparation, subset verification, and evaluation protocol.
- Phase 2: published detector benchmark review and our internal baseline
  consolidation.

Out of scope for this report:

- New training runs.
- DETR, Deformable Faster R-CNN, or EfficientAD reproduction.
- Colab/MMDetection setup continuation.
- Fusion, thermal imagery, other datasets, or edge deployment.
- DINOv3 LoRA pilot training. The LoRA infrastructure exists, but pilot
  training belongs to Phase 3.

## 2. ForestPersons Dataset And Local Subsets

Primary paper:

- ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person
  Detection, arXiv:2603.02541v1 / ICLR 2026 accepted.
- Local PDF: `docs/papers/3-forestpersons.pdf`.
- Online HTML source used for table verification:
  `https://arxiv.org/html/2603.02541v1`.

The paper describes ForestPersons as an RGB under-canopy missing-person dataset
with bounding boxes, pose labels, and visibility labels. It reports 96,482
images and 204,078 annotations. Our local official annotation JSON files match
the official split sizes:

| Split | Images | Annotations |
| --- | ---: | ---: |
| Train | 67,686 | 145,816 |
| Validation | 18,243 | 37,395 |
| Test | 10,553 | 20,867 |
| Total | 96,482 | 204,078 |

Official category mapping in the annotation files:

- Category IDs 1-12 encode pose and visibility combinations such as
  `sitting_100`, `lying_40`, and `standing_20`.
- Category ID 13 is `unknown_0`.
- All categories have `supercategory: person`.
- For our one-class detector evaluation, categories are mapped to one `person`
  class while visibility metadata is preserved separately.

Official annotation visibility distribution:

| Split | 20 | 40 | 70 | 100 | Unknown / missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 12,326 | 14,708 | 27,259 | 91,523 | 0 |
| Validation | 3,303 | 4,376 | 7,945 | 21,770 | 1 |
| Test | 1,946 | 2,454 | 4,499 | 11,968 | 0 |

## 3. Local train5000 And val600 Subsets

We use two local pilot subsets:

- Train subset:
  `outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json`
- Validation subset:
  `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`

These subsets are reproducible, visibility-stratified pilot subsets. They are
not the full ForestPersons benchmark.

### train5000

Selection strategy:

- Training split only.
- Deterministic seed: `20261012`.
- Includes the previous 400-image subset where possible.
- Target: 1,250 images per target visibility level.
- Selection considers season, camera, place, weather, and sequence folder.

Distribution:

| Item | Value |
| --- | ---: |
| Images | 5,000 |
| Target visibility images 20/40/70/100 | 1,250 / 1,250 / 1,250 / 1,250 |
| Annotation visibility counts 20/40/70/100 | 1,423 / 1,440 / 1,595 / 1,957 |
| Estimated downloaded image size | 2,417,548,434 bytes |
| Estimated frozen DINOv3 feature cache | about 11.41 GiB |

The YOLO conversion summary reports 5,000 images and 6,431 train annotations,
with zero clipped/dropped boxes and zero images overlapping validation.

### val600

Selection strategy:

- Validation split only.
- Deterministic seed: `20261010`.
- Image-based stratified sample.
- Requested: 150 images per target visibility level.

Distribution:

| Item | Value |
| --- | ---: |
| Images | 600 |
| Target visibility images 20/40/70/100 | 150 / 150 / 150 / 150 |
| Evaluated annotations | 749 |
| Skipped unknown-visibility annotation | 1 |
| Annotation visibility counts 20/40/70/100 | 174 / 170 / 193 / 212 |
| Estimated downloaded image size | 292,958,727 bytes |

Important limitation:

- Image-level stratification does not guarantee equal annotation counts per
  visibility group because images can contain multiple people and mixed
  visibility labels.

## 4. Data Integrity, Leakage, And Evaluation Protocol

Verified local evidence:

- Official annotation JSON files are present under `data/annotations/`.
- Local image subsets are under `data/images/`.
- The reusable loader is `src/forestpersons/dataset.py`.
- The verified evaluation framework is `src/forestpersons/evaluation.py` and
  `scripts/evaluate_forestpersons_framework.py`.
- The YOLO conversion summary reports:
  - `overlap_images: 0`.
  - `class_mapping: {"0": "person"}`.
  - no clipped or dropped boxes.
  - 1 unknown-visibility val annotation skipped.

Evaluation protocol for our detector pilots:

- Dataset: selected `val600`, not official full validation/test.
- Task: one-class person detection.
- Category mapping: all pose/visibility person categories -> `person`.
- COCO metrics: pycocotools bbox AP/AR with maxDets `[1, 10, 100]`.
- AP is AP@[0.50:0.95].
- AP50 and AP75 are reported separately.
- AR@100 is the primary recall-style COCO summary.
- Custom metrics use explicit confidence thresholds, IoU 0.50, one-to-one
  greedy matching, precision, recall, FP/image, and visibility-specific recall.

The official test split has not been evaluated by our code.

## 5. Published ForestPersons Benchmarks

The paper defines AP and AR as averages over IoU thresholds from 0.50 to 0.95 in
steps of 0.05. It reports AP, AP50, AP75, and AR. Table 3 reports object
detection performance on the official ForestPersons validation and test splits.

Important interpretation:

- The paper's `DINO` detector is a DETR-family object detector implemented with
  detrex according to the paper's implementation notes. It is not DINOv3.
- The published numbers below are full-split paper benchmarks.
- Our local pilot numbers are not directly comparable because they use
  train5000/val600, not the full official train/validation/test protocol.

| Paper model | Val AP | Val AP50 | Val AP75 | Val AR | Test AP | Test AP50 | Test AP75 | Test AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv11 | 65.3 | 95.4 | 76.6 | 71.5 | 65.6 | 93.4 | 75.6 | 71.7 |
| DETR | 55.3 | 93.0 | 59.9 | 68.0 | 53.9 | 88.7 | 59.4 | 67.9 |
| Deformable R-CNN | 65.0 | 94.7 | 78.5 | 70.0 | 66.3 | 93.4 | 77.5 | 71.3 |
| Faster R-CNN | 64.2 | 95.6 | 76.5 | 69.6 | 64.4 | 92.7 | 75.4 | 70.0 |
| DINO object detector | 59.9 | 91.7 | 69.1 | 70.1 | 65.3 | 94.0 | 76.2 | 77.7 |

Other paper Table 3 context:

- CZ Det has the best validation AP in Table 3: 69.9.
- Deformable R-CNN has the highest test AP among the listed rows above: 66.3.
- DINO object detector has the highest test AR in the table: 77.7.

## 6. Our ForestPersons Pilot Detector Results

All rows below use our selected `val600` subset with 600 images and 749 evaluated
ground-truth persons. These are pilot results only.

| Model | Train data | AP | AP50 | AP75 | AR@100 | Conf. | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned | train5000, 47 epochs | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.05 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |
| Frozen DINOv3 + FCOS head | train5000 cached features | 0.3113 | 0.7564 | 0.1738 | 0.4368 | 0.60 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| Frozen DINOv3 + linear patch head | train5000 cached features | 0.0614 | 0.2615 | 0.0050 | 0.2889 | 0.30 | 0.1041 | 0.3939 | 0.2000 | 0.0460 | 0.1294 | 0.0984 | 0.1368 |

Notes:

- YOLO11s is our own internal fine-tuned baseline on the same train5000 subset.
- Frozen DINOv3 + FCOS is the strongest current DINOv3 detector pilot.
- The linear patch head is intentionally simple and performs poorly, confirming
  that neighboring-patch context, FCOS-style assignment, and centerness are
  important for this task.
- The confidence thresholds shown above are selected operating points from
  existing confidence sweeps, not globally optimized official benchmark
  thresholds.

## 7. Patch-Level Anomaly Results

These methods produce patch-level anomaly/localization scores, not bounding-box
detections. They should not be compared directly against detector AP/AR.

| Method | Feature / model | AUROC | AP | AUROC@20 | AP@20 | AUROC@40 | AP@40 | AUROC@70 | AP@70 | AUROC@100 | AP@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3 patch-kNN | frozen DINOv3 patch features | 0.8037 | 0.1338 | 0.7770 | 0.0216 | 0.7941 | 0.0339 | 0.8036 | 0.0441 | 0.8248 | 0.0583 |
| RX Mahalanobis | RGB patch mean/std | 0.5787 | 0.0314 | 0.5499 | 0.0047 | 0.5871 | 0.0073 | 0.5723 | 0.0089 | 0.5942 | 0.0118 |

EfficientAD status:

- EfficientAD was assessed for feasibility only.
- No genuine EfficientAD result artifact was found.
- It must be reported as not executed.

## 8. Fair Comparison And Metric Limitations

Published ForestPersons Table 3 numbers:

- Use full official training, validation, and test splits.
- Are reported by the paper authors.
- Include the official full validation/test protocol.

Our pilot numbers:

- Use train5000 and a selected val600 subset.
- Are useful for internal controlled comparison.
- Are not official benchmark reproductions.
- Cannot be used to claim superiority over paper-reported full-dataset models.

Detection and anomaly metrics are separate:

- Detector AP/AR use bounding boxes and COCO-style matching.
- Custom recall/precision/FP-image use explicit confidence thresholds and IoU
  0.50.
- Patch anomaly AUROC/AP classify person-overlapping versus background patches.
  They do not represent person bounding-box detection without an additional
  proposal method.

## 9. Phase 1 And Phase 2 Completion Status

Phase 1 - ForestPersons dataset preparation:

- Complete for the ForestPersons-only pilot scope.
- Official annotations are present and validated.
- train5000 and val600 subsets are documented and reproducible.
- Annotation/category mapping is clear.
- Leakage check for train5000/val600 is zero overlapping images.
- Evaluation limitations are documented.

Phase 2 - Published baseline review and internal baselines:

- Complete for the current ForestPersons-only scope.
- Published Table 3 benchmark values for YOLOv11, DETR, Deformable R-CNN,
  Faster R-CNN, and DINO object detector are documented.
- We will not retrain DETR/Deformable R-CNN this week.
- Our internal YOLO11s fine-tuned pilot is consolidated.
- RX anomaly and DINOv3 patch-kNN anomaly pilots are consolidated.
- EfficientAD is explicitly marked not executed.

Remaining evidence gap:

- We did not reproduce published DETR or Deformable R-CNN full-dataset
  benchmarks locally.
- We did not evaluate the official ForestPersons test split.
- We did not run EfficientAD.

## 10. Remaining Phase 3 Work

ForestPersons-only Phase 3 priorities:

1. Run the first real DINOv3 LoRA + FCOS pilot training experiment.
2. Compare LoRA + FCOS against frozen DINOv3 + FCOS using the same train/val
   splits and metrics.
3. Analyze recall under heavy occlusion, especially visibility 20 and 40.
4. Decide whether LoRA improves enough to justify larger training or whether to
   revise the lightweight detection head.

The next approved experiment should be a small LoRA pilot, not full-dataset
training.
