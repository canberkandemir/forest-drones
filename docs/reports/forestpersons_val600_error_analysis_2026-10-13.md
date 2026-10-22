# ForestPersons val600 Detection Error And Recall Analysis - 2026-10-13

Scope: saved predictions only; no inference or training was run. Metrics use the selected val600 subset with 600 images and 749 visibility-labeled person annotations. COCO AR@100 is separate from threshold-specific recall at IoU 0.50.

## Artifact Check
- YOLO11s ft train5000: `outputs/evaluation_framework/val600_yolo11s_finetuned/summary.json`, `coco_results.json`, and `confidence_sweep.csv` found.
- Frozen DINOv3 FCOS train5000: `outputs/evaluation_framework/val600_dinov3_fcos5k/summary.json`, `coco_results.json`, and `confidence_sweep.csv` found.
- DINOv3 LoRA FCOS train5000: `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/summary.json`, `coco_results.json`, and `confidence_sweep.csv` found.

## COCO Metrics
| Model | AP | AP50 | AP75 | AP50-AP75 gap | AR@100 | Raw predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.5541 | 0.8941 | 0.5687 | 0.3254 | 0.6339 | 2622 |
| Frozen DINOv3 FCOS train5000 | 0.3113 | 0.7564 | 0.1738 | 0.5826 | 0.4368 | 46087 |
| DINOv3 LoRA FCOS train5000 | 0.3587 | 0.8139 | 0.2379 | 0.5760 | 0.4704 | 46586 |

Large AP50-AP75 gaps for DINOv3 models indicate that many detections overlap enough to count at IoU 0.50 but localization is much weaker at stricter IoU 0.75.

## Metrics At Confidence 0.60
| Model | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 | matched IoU mean | matched IoU median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.7490 | 0.9413 | 0.0583 | 0.4943 | 0.6882 | 0.8342 | 0.9292 | 0.831 | 0.858 |
| Frozen DINOv3 FCOS train5000 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 | 0.719 | 0.731 |
| DINOv3 LoRA FCOS train5000 | 0.8158 | 0.7833 | 0.2817 | 0.6149 | 0.7588 | 0.8912 | 0.9575 | 0.731 | 0.740 |

Note: YOLO11s was originally selected at confidence 0.05 for its own operating point, so the confidence-0.60 row is mainly a common-threshold diagnostic, not YOLO's best operating point.

## Matched FP/Image Budgets
### FP/image <= 0.25
| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.15 | 0.8838 | 0.8390 | 0.2117 | 0.7356 | 0.8471 | 0.9585 | 0.9670 |
| Frozen DINOv3 FCOS train5000 | 0.64 | 0.7316 | 0.7908 | 0.2417 | 0.4885 | 0.6647 | 0.8031 | 0.9198 |
| DINOv3 LoRA FCOS train5000 | 0.62 | 0.7917 | 0.8035 | 0.2417 | 0.5747 | 0.7294 | 0.8705 | 0.9481 |

### FP/image <= 0.425
| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.05 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |
| Frozen DINOv3 FCOS train5000 | 0.6 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| DINOv3 LoRA FCOS train5000 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |

### FP/image <= 0.5
| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.025 | 0.9146 | 0.6976 | 0.4950 | 0.7816 | 0.9000 | 0.9793 | 0.9764 |
| Frozen DINOv3 FCOS train5000 | 0.6 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| DINOv3 LoRA FCOS train5000 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |

### FP/image <= 1.0
| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.01 | 0.9266 | 0.5886 | 0.8083 | 0.8103 | 0.9118 | 0.9845 | 0.9811 |
| Frozen DINOv3 FCOS train5000 | 0.55 | 0.8198 | 0.6158 | 0.6383 | 0.6437 | 0.7706 | 0.8808 | 0.9481 |
| DINOv3 LoRA FCOS train5000 | 0.5 | 0.8652 | 0.6249 | 0.6483 | 0.6897 | 0.8294 | 0.9482 | 0.9623 |

## Maximum Threshold-Sweep Recall
Maximum recall here means the highest custom recall across the evaluated confidence thresholds at IoU 0.50, with no false-positive budget. It is not COCO AR.

| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 0.001 | 0.9519 | 0.2719 | 3.18 | 0.8851 | 0.9353 | 0.9896 | 0.9858 |
| Frozen DINOv3 FCOS train5000 | 0.001 | 0.9159 | 0.0149 | 75.67 | 0.7989 | 0.8941 | 0.9741 | 0.9764 |
| DINOv3 LoRA FCOS train5000 | 0.001 | 0.9212 | 0.0148 | 76.49 | 0.8161 | 0.8882 | 0.9793 | 0.9811 |

## Matched IoU Distribution At Primary Operating Points
Primary points: YOLO confidence 0.05; frozen DINOv3 and LoRA DINOv3 confidence 0.60.

| Model | TP | IoU mean | median | p25 | p75 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s ft train5000 | 677 | 0.811 | 0.839 | 0.735 | 0.900 | 0.504 | 0.985 |
| Frozen DINOv3 FCOS train5000 | 583 | 0.719 | 0.731 | 0.647 | 0.795 | 0.501 | 0.934 |
| DINOv3 LoRA FCOS train5000 | 611 | 0.731 | 0.740 | 0.663 | 0.809 | 0.501 | 0.933 |

## Cross-Model Hit/Miss Overlap
Primary points: YOLO confidence 0.05; frozen DINOv3 and LoRA DINOv3 confidence 0.60.

| Case | Count |
| --- | ---: |
| yolo_succeeds_lora_fails | 91 |
| lora_succeeds_yolo_fails | 25 |
| yolo_succeeds_frozen_fails | 112 |
| frozen_succeeds_yolo_fails | 18 |
| lora_recovers_frozen_misses | 43 |
| frozen_hits_lora_misses | 15 |
| all_three_hit | 550 |
| all_three_miss | 47 |

Example annotations are saved to `outputs/error_analysis/val600_detector_error_analysis_20261013/overlap_examples.json`.

## Findings
- YOLO11s remains the strongest detector overall: best AP, AP75, AR@100, and recall at every matched FP/image budget tested.
- LoRA improves DINOv3 over the frozen FCOS head: AP rises from 0.3113 to 0.3587, AR@100 from 0.4368 to 0.4704, and R@20 at confidence 0.60 from 0.5632 to 0.6149.
- At FP/image <= 0.425, LoRA DINOv3 reaches recall 0.8465 and R@20 0.6667, versus frozen DINOv3 recall 0.7784 and R@20 0.5632. This suggests LoRA helps both missed detections and low-visibility recall.
- DINOv3 still trails YOLO most strongly at low visibility: at the 0.425 FP/image budget, YOLO R@20 is higher than LoRA R@20. The remaining gap is mainly missed low-visibility detections, not just false positives.
- Localization is also a DINOv3 weakness: the AP50-to-AP75 gap is larger for DINOv3 models than YOLO, so some DINO detections are coarse boxes that pass IoU 0.50 but fail stricter localization.
- False positives are not the main LoRA-vs-frozen problem at confidence 0.60; LoRA has lower FP/image than frozen DINOv3 and better recall. The bigger remaining gap to YOLO is recall and localization quality.

## Limitations
- These are pilot val600 results, not official full ForestPersons validation scores.
- The analysis uses saved predictions. It does not inspect image content manually beyond saved visualization examples.
- Primary operating points differ by model because the confidence calibration differs. Matched FP/image tables are the fairest operating comparison here.
