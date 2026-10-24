# LogXY Comprehensive Experiment Comparison

**Date:** 2026-10-24
**Project:** Finding People Under Dense Vegetation Using DINOv3
**Supervisor:** Dr. Davide Baroli
**Scope:** Evidence-based audit of completed, interrupted, prepared, blocked, and planned work through 2026-10-24.

## 1. Executive Summary

The project has established a reproducible ForestPersons pipeline, verified COCO-style evaluation, several YOLO and DINOv3 detector families, anomaly baselines, cross-dataset studies, multi-view and RGB-thermal studies, and two spiking-neural-network tracks. The strongest single ForestPersons detector remains fine-tuned YOLO11s (val600 AP 0.5541, AR@100 0.6339). DINOv3 is nevertheless useful: LoRA + FCOS reaches recall 0.8465 and R20 0.6667 at at most 0.425 FP/image, while multi-level/multi-resolution readout improves threshold-independent localization over a matched frozen final-layer baseline. Selective YOLO+DINO fusion independently increases holdout1200 recall from 0.9062 to 0.9185, but reduces AP/AR.

The newest matched Full-MVR continuation shows a modest LoRA advantage over a frozen backbone: AP 0.3652 versus 0.3533, AP75 0.2722 versus 0.2362, and R20 0.6552 versus 0.6264 at the matched false-positive budget. This costs approximately 2.69 times the training runtime and about 5.10 times the observed peak MPS memory. The result uses one seed and a two-epoch continuation from a shared trained MVR checkpoint; it is not a scratch-training comparison and is not evidence of statistical significance.

## 2. Research Objective

The central objective is high-recall person detection in UAV imagery under severe vegetation occlusion, with explicit attention to 20%, 40%, 70%, and 100% visibility. DINOv3 is evaluated as a frozen dense representation, a LoRA-adapted backbone, a multi-level/multi-resolution encoder, an anomaly representation, and a shared encoder for multi-view or RGB-thermal fusion. YOLO11s provides a strong practical detector baseline. Spiking studies assess conversion and dynamic timestep feasibility without claiming hardware energy savings.

## 3. Dataset Description

### 3.1 Official ForestPersons annotations

| Split | Images | Person annotations | V20 | V40 | V70 | V100 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 67,686 | 145,816 | 12,326 | 14,708 | 27,259 | 91,523 |
| Validation | 18,243 | 37,395 | 3,303 | 4,376 | 7,945 | 21,770 |
| Test | 10,553 | 20,867 | 1,946 | 2,454 | 4,499 | 11,968 |

The 12 pose/visibility categories and the unknown category are mapped to one `person` detection class while retaining visibility and pose metadata. Evidence: `docs/reports/forestpersons_phase1_phase2_2026-10-12.md` and `data/annotations/*.json`.

### 3.2 Experimental subsets

| Subset | Images | Evaluated annotations | Construction/use |
|---|---:|---:|---|
| train400 | 400 | 519 after image-ID-safe correction | Small controlled pilots |
| train5000 | 5,000 | 6,415 after image-ID-safe correction | Main local training pool |
| val32 | 32 | 48 | Initial real-image smoke test |
| val600 | 600 | 749 | Selected pilot benchmark; 174/170/193/212 GT at V20/V40/V70/V100 |
| holdout1200 | 1,200 | 1,461 in saved fusion evaluation | Image-disjoint selective-fusion confirmation |

The nested label-efficiency subsets contain 141/519/1,300/6,415 annotations for train100/400/1000/5000 and 89/222/258/258 sequence folders. A historical YOLO-conversion artifact reports 6,431 train5000 annotations; the canonical corrected image-ID lookup reports 6,415. This discrepancy must remain documented rather than silently reconciled.

Train and validation manifests have zero split-qualified filename overlap. Numeric image IDs are split-local and must not be used alone for identity. The subset audit found 45 numeric-ID collisions across train and validation and duplicate filenames within training annotations; current training lookup resolves by `image_id` and verifies filename. Scene independence is not guaranteed for every historical pilot.

## 4. Experimental Methodology

ForestPersons detection uses one-class COCO output (`xywh` in original image coordinates), COCO AP@[0.50:0.95], AP50, AP75, and AR with maxDets 1/10/100. Custom operating metrics use IoU 0.50, one-to-one GT matching, confidence sweeps, precision, recall, FP/image, and recall by visibility. Matched-budget values select each model's highest recall without exceeding the stated FP/image budget; values selected on val600 are exploratory, not independent validation.

The local COCO implementation was checked against `pycocotools`. The official ForestPersons test split has not been used for project model selection. Published full-dataset paper numbers, selected val600 pilots, holdout1200 fusion results, AOS results, and patch-level anomaly metrics are separate protocols and must not be ranked in one table as though equivalent.

## 5. Model Architectures

- **YOLO11s:** Ultralytics one-stage detector, fine-tuned end to end on train5000 for 47 completed epochs.
- **DINOv3 final-layer FCOS:** frozen `facebook/dinov3-vits16-pretrain-lvd1689m` stride-16 patch features with a lightweight FCOS-style objectness, centerness, and L/T/R/B head.
- **DINOv3 LoRA FCOS:** rank-4 adapters on `q_proj` and `v_proj` in all 12 transformer blocks; original backbone weights frozen; FCOS remains trainable.
- **Full MVR FCOS:** layers 3/6/9 plus final-layer features, native and fine/coarse readout, then FCOS. The frozen readout/head has 923,622 trainable parameters.
- **LoRA Full MVR:** Full MVR plus 73,728 fresh q/v LoRA parameters, totaling 997,350 trainable parameters in the matched continuation.
- **Hybrid ANN2SNN:** only six MVR ReLU sites are converted; signed DINO features, residual normalization, FCOS GroupNorm/GELU, classification/centerness, analog box regression, decoding, and NMS remain ANN.
- **DTSS transformer/detector:** a separate compact spiking architecture with learned prefix timestep masks; it is not a conversion of DINOv3.

## 6. Complete Experiment Inventory

The canonical row-level inventory is `docs/experiment_registry/experiments.csv`. It preserves smoke tests, interrupted runs, legacy artifacts, and prepared workflows rather than listing only successful models.

| Area | Completed evidence | Prepared/blocked/planned evidence |
|---|---|---|
| Dataset | Official annotations, manifests, verified subsets, visualization, leakage checks | Full local corpus has separate completeness history; Git excludes images |
| Published baselines | YOLO11n smoke, YOLO11n/s grid, YOLO11s fine-tune, RX | DETR and Deformable Faster R-CNN CUDA workflows prepared but not run; EfficientAD blocked |
| DINOv3 detectors | v1, v2, v3/train400, FCOS/train5000, linear, LoRA, ViT-L web/satellite, FPN, GIoU, YOLO-style, quality branches, RoI, stride-8, focal normalization, DEIMv2 zero-shot | DEIMv2 train400 prepared only; no full official test benchmark |
| Representation | Frozen feature caches, final-layer/native/full MVR, nested label efficiency, Frozen-vs-LoRA Full MVR | Additional seeds and independently held-out model selection remain |
| Fusion | YOLO+DINO union/NMS/WBF diagnostics, selective fusion, independent1200, ForestPersons multi-view equal/selective/learned, AOS RGB-thermal | Exact raw AOS geometry and temporal fusion unresolved |
| Anomaly | DINOv3 patch-kNN, RX | EfficientAD not executed because valid normal-only data are absent |
| SNN | DTSS prototype, synthetic learning, small detection pilot, train5000 fixed/learned, MVR ANN2SNN T4/T8/T16 | Real sparse-hardware energy/latency study not performed |
| Cross-dataset | Weitefeld review, SARD pilot, AOS integral-image studies | Weitefeld cannot support confirmed-person recall; broader external validation remains |
| Edge | None | Export, quantization, and device deployment not started |

## 7. Comprehensive Comparison Tables

### 7.1 Main ForestPersons val600 detector results

| Model/protocol | Train | AP | AP50 | AP75 | AR100 | Recall | Precision | FP/img | R20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s fine-tuned, conf 0.05 | 5,000 | **0.5541** | **0.8941** | 0.5687 | **0.6339** | **0.9039** | 0.7650 | 0.3467 | **0.7701** |
| Frozen DINO FCOS, conf 0.60 | 5,000 | 0.3113 | 0.7564 | 0.1738 | 0.4368 | 0.7784 | 0.7127 | 0.3917 | 0.5632 |
| LoRA DINO FCOS, conf 0.60 | 5,000 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8158 | 0.7833 | 0.2817 | 0.6149 |
| LoRA DINO FCOS, matched <=0.425 FP/img | 5,000 | same ranked predictions | same | same | same | 0.8465 | not primary | 0.4167 | 0.6667 |
| Linear patch head, conf 0.30 | 5,000 | 0.0614 | 0.2615 | 0.0050 | 0.2889 | 0.1041 | 0.3939 | 0.2000 | 0.0460 |
| DEIMv2-S COCO zero-shot, conf 0.175 | none | 0.3408 | 0.5693 | **0.3544** | 0.6016 | 0.5033 | 0.6294 | 0.3700 | 0.1782 |

AP/AR are threshold-independent COCO-style metrics; recall rows use the documented operating confidence and must not be conflated with AR.

### 7.2 Historical DINO head progression at approximately YOLO's early 0.425 FP/image reference

| Head | Training | Confidence | Recall | Precision | FP/img | R20 |
|---|---|---:|---:|---:|---:|---:|
| v1 center-patch | train400 | 0.97 | 0.1175 | 0.2914 | 0.3567 | 0.0460 |
| v2 grid/focal | train400 | 0.39 | 0.2430 | 0.4365 | 0.3917 | 0.1034 |
| v3 FCOS | train400 | 0.62 | 0.6395 | 0.6447 | 0.4400 | 0.3678 |
| FCOS | train5000 | 0.60 | 0.7784 | 0.7127 | 0.3917 | 0.5632 |

The original v1 had a width-derived shared resize scale; rechecked comparisons use separate x/y scaling and did not reveal a material invalidating error.

## 8. YOLO Baseline Analysis

Pretrained YOLO11s reached recall 0.4673 and R20 0.1782 at confidence 0.05 and 0.425 FP/image. Fine-tuning on train5000 produced the strongest project detector: AP 0.5541, AP50 0.8941, AP75 0.5687, AR100 0.6339, recall 0.9039, and R20 0.7701 at 0.3467 FP/image. Its 47-epoch checkpoint and evaluation are under `outputs/yolo11_finetune/` and `outputs/evaluation_framework/val600_yolo11s_finetuned/` in the artifact root.

Published paper results use full official splits and cannot validate this pilot. The paper reports test AP 65.6 for YOLOv11, but the architecture/training scale and test protocol differ. DETR and Deformable Faster R-CNN local workflows are prepared only; no project GPU result exists.

## 9. DINOv3 + FCOS Analysis

The v1-to-v3 progression shows that neighboring context, focal classification, multi-positive FCOS assignment, and more data matter much more than a linear decoder. LoRA train5000 improves the frozen FCOS model from AP 0.3113 to 0.3587 and improves practical low-visibility recall. Candidate analysis nevertheless identifies localization and confidence-ranking weaknesses: at confidence 0.55, 634/749 GT are detected, with 59 localization and 56 confidence misses; 20%-visible people account for 32 localization and 26 confidence misses.

Several targeted modifications did not supplant the main stride-16 model: FPN AP 0.3062, stride-8 AP 0.2188, YOLO-style quality model AP 0.1870, and IoU-quality FCOS AP 0.3485. GIoU and positive-count focal normalization produced small R20 changes without a decisive AP75 gain. The frozen-detector RoI box refiner gave the most credible localization improvement (AP75 0.2719 and recall 0.8545), but the recall gain was too small to justify train5000 scaling.

## 10. MVR Ablation Analysis

| Readout, frozen DINOv3 | Parameters | AP | AP50 | AP75 | AR100 | Recall/R20 at <=0.425 FP/img |
|---|---:|---:|---:|---:|---:|---:|
| Final layer | 524,646 | 0.3267 | 0.7915 | 0.2022 | 0.4550 | 0.8198 / 0.6207 |
| Native intermediate layers | 721,894 | 0.3303 | 0.7926 | 0.2082 | 0.4615 | 0.8318 / 0.6322 |
| Full multi-level/multi-resolution | 923,622 | **0.3623** | **0.8162** | **0.2675** | **0.4900** | 0.8278 / 0.6092 |

Native layers alone provide small gains. Fine/coarse multi-resolution context accounts for most AP, AP75, and AR improvement. The matched-budget recall result does not favor Full MVR at 0.425 FP/image, although Full MVR is stronger at the stricter 0.15 budget (recall/R20 0.7210/0.4885 versus 0.6916/0.4425). The deterministic rerun reproduced predictions and sweeps byte-for-byte. Evidence: `docs/reports/dinov3_mvr_native_ablation_results_2026-10-23.md`.

## 11. Label-Efficiency Analysis

| Images | Optimizer steps | Runtime (s) | AP | AP50 | AP75 | AR100 | Recall | R20 at <=0.425 FP/img |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2,000 | 165 | 0.0844 | 0.3083 | 0.0151 | 0.2430 | 0.3231 | 0.1264 |
| 400 | 8,000 | 612 | 0.1735 | 0.5280 | 0.0547 | 0.3387 | 0.5607 | 0.2644 |
| 1,000 | 20,000 | 1,665 | 0.2673 | 0.7068 | 0.1290 | 0.4099 | 0.7250 | 0.4713 |
| 5,000 | 100,000 | 8,119 | 0.3353 | 0.7756 | 0.2169 | 0.4780 | 0.8224 | 0.5977 |

Performance increases monotonically, with no saturation demonstrated by 5,000 images. The study has one seed, fixed epochs cause training steps to scale with data, train1000 already covers all 258 sequence folders, and the matched train5000 AP is 0.0270 below the earlier differently ordered Full MVR run. Thus it measures the combined effect of labels, updates, and ordering rather than a pure sample-complexity curve.

## 12. Frozen vs LoRA Analysis

Both arms start from the same trained Full-MVR checkpoint and continue for two epochs on the same ordered train5000 stream. The frozen arm trains MVR/FCOS only; the LoRA arm adds fresh rank-4 q/v adapters.

| Arm | AP | AP50 | AP75 | AR100 | Threshold | Recall | Precision | FP/img | R20 | R40 | R70 | R100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 0.3532597 | 0.8082086 | 0.2362244 | 0.4846462 | 0.58 | 0.8504673 | 0.7181511 | 0.4166667 | 0.6264368 | 0.8352941 | 0.9326425 | 0.9716981 |
| LoRA | **0.3651900** | **0.8143840** | **0.2722363** | **0.4873164** | 0.59 | **0.8531375** | **0.7269625** | **0.4000000** | **0.6551724** | 0.8235294 | 0.9222798 | **0.9764151** |

Frozen loss decreased 0.616008 to 0.610457; LoRA loss decreased 0.622114 to 0.615904. Each arm used 2,500 optimizer steps. Frozen training took 4,916 s with 2.33 GiB observed peak MPS driver memory; LoRA took 13,232 s with 11.89 GiB. The strongest change is AP75 (+0.0360) and R20 (+0.0287), but one seed and reused val600 prohibit significance claims. Primary evidence: `outputs/dinov3_mvr/frozen_vs_lora/evaluation_val600_seed20261012/metrics.json` under the external artifact root.

## 13. YOLO + DINO Fusion Analysis

Naive union, class-aware NMS, and weighted-box fusion did not beat YOLO reliably because score scales differ and DINO adds false positives. Selective Fusion B uses frozen support rules and was independently evaluated on image-disjoint holdout1200.

| Holdout1200 method | AP | AP50 | AR100 | Recall | R20 | Precision | FP/img | Net effect vs YOLO |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO-only | 0.5710 | 0.9027 | 0.6373 | 0.9062 | 0.7859 | 0.7662 | 0.3367 | reference |
| Selective A | 0.5506 | 0.8756 | 0.6238 | 0.9172 | 0.8035 | 0.7334 | 0.4058 | recall up, AP/AR down |
| Selective B | 0.5510 | 0.8762 | 0.6260 | **0.9185** | **0.8035** | 0.7599 | 0.3533 | +23 rescued, -5 lost, net +18 GT, +20 FP |

The paired bootstrap CI for Selective B overall recall difference is [+0.0049, +0.0202]; its R20 CI crosses zero. Holdout1200 is image-disjoint but not proven scene-disjoint. Fusion B is defensible as an optional recall mode, not as a replacement for YOLO on AP/AR.

## 14. ANN2SNN Analysis

| Mode | AP50 | Recall | R20 | FP/img | MPS latency/image | Mean spike activity |
|---|---:|---:|---:|---:|---:|---:|
| ANN | 0.8162 | 0.8077 | 0.5862 | 0.3183 | 13.0 ms | n/a |
| T=4 | 0.7622 | 0.7156 | 0.4425 | 0.2700 | 23.2 ms | 0.0246 |
| T=8 | 0.8002 | 0.7891 | 0.5632 | 0.2967 | 31.1 ms | 0.0320 |
| T=16 | 0.8083 | 0.7997 | 0.5862 | 0.2933 | 49.1 ms | 0.0356 |

Increasing timesteps recovers accuracy, but T16 remains slightly below ANN and is 3.77 times slower in dense MPS simulation. Spike counts do not establish energy savings. Because much of the pipeline remains analog and eager execution does not exploit event sparsity, this is a numerical conversion study rather than an edge-efficiency demonstration.

The separate DTSS detector was technically valid but ineffective: learned DTSS recall 0.1242, AP50 0.0045, R20 0.0402, and 43.97 FP/image versus fixed T4 recall 0.0961, AP50 0.0029, and 35.09 FP/image. Learned masks collapsed several paths to zero timesteps, while eager execution still computed all T=4 steps.

## 15. Visibility-Specific Detection Performance

| Model/protocol | FP/img | R20 | R40 | R70 | R100 |
|---|---:|---:|---:|---:|---:|
| YOLO11s fine-tuned, conf 0.05 | 0.3467 | **0.7701** | 0.8765 | 0.9741 | 0.9717 |
| LoRA FCOS train5000, matched budget | 0.4167 | 0.6667 | not recorded in headline artifact | not recorded | not recorded |
| Frozen Full-MVR continuation | 0.4167 | 0.6264 | **0.8353** | **0.9326** | 0.9717 |
| LoRA Full-MVR continuation | 0.4000 | 0.6552 | 0.8235 | 0.9223 | **0.9764** |
| Selective Fusion B holdout1200 | 0.3533 | 0.8035 | protocol-specific | protocol-specific | protocol-specific |

Visibility-20 remains the central failure mode. LoRA improves R20 over its matched frozen MVR arm, while YOLO remains stronger on val600. Holdout1200 values are not directly interchangeable with val600 values.

## 16. Computational Cost and Efficiency

| Experiment | Trainable parameters | Training/inference cost | Observed memory |
|---|---:|---|---:|
| Linear patch head | 1,925 | lightweight, poor detection | low |
| LoRA FCOS | 598,374 in original setup | 2-epoch train5000 local MPS | about 10.5 GiB in train400 pilot |
| Full MVR frozen | 923,622 | 20-epoch train5000 about 2.25 h in label-efficiency run | about 1.10 GiB cached-feature training |
| Frozen Full-MVR continuation | 923,622 | 1.37 h for two online epochs | 2.33 GiB |
| LoRA Full-MVR continuation | 997,350 | 3.68 h for two online epochs | 11.89 GiB |
| DEIMv2-S zero-shot | 9.78M model parameters | 0.079 s/image MPS | 1.16 GiB driver allocation |
| ANN2SNN T16 | hybrid | 49.1 ms/image, slower than ANN | dense simulation |

Cached frozen-feature training is markedly cheaper than online LoRA. The M4 16 GB machine can execute ViT-S LoRA Full MVR, but with little memory margin. ViT-L/16 at short-side 672 failed around 19.36 GiB MPS allocation; matched ViT-L web/satellite pilots therefore used short-side 336.

## 17. Limitations and Reproducibility

1. Most ForestPersons comparisons use selected val600 repeatedly; threshold selection there is exploratory.
2. Many experiments have one seed, so observed differences are not statistical significance.
3. Continuation studies inherit a trained checkpoint and are not equivalent to matched scratch training.
4. Label-efficiency fixed epochs imply unequal update budgets; ordering affected train5000 AP.
5. Published benchmark values use official full splits and different training budgets.
6. Holdout1200 is image-disjoint but not conclusively scene-disjoint.
7. The historical 6,431 versus corrected 6,415 train5000 annotation count reflects the fixed filename/image-ID lookup issue.
8. Weitefeld class 3 is not confirmed-person ground truth; 0 confirmed, 134 explicit non-person/no-abnormality, and 21 uncertain cases were found.
9. SARD is sequence-correlated and annotation exhaustiveness/license details were not fully verified.
10. AOS integral-image results are a distinct scene-disjoint multimodal protocol; raw-view geometry remains incomplete.
11. Spiking simulations do not measure deployment energy.
12. Large data, caches, checkpoints, and metrics live under the local `$LOGXY_ARTIFACT_ROOT`; Git contains code and documentation only.

## 18. Remaining Experiments

- **Prepared, not executed:** local DEIMv2-S train400 fine-tuning pending supervisor/license confirmation; CUDA DETR R50 and Deformable Faster R-CNN workflows; exact deployment/export work.
- **Blocked:** EfficientAD lacks verified normal-only ForestPersons imagery; exact AOS raw-view virtual-camera reprojection lacks all required geometry/terrain transformations.
- **Scientifically useful repeats:** additional seeds for Frozen-vs-LoRA MVR and label efficiency; scene-disjoint independent ForestPersons confirmation.
- **Deferred:** full official ForestPersons split training; ViT-L web/satellite matched larger training; edge hardware measurement; temporal fusion.
- **Stopped lines:** more ForestPersons homography-gate tuning, current stride-8 adapter, current DINO-YOLO head, and current DTSS detector without a revised hypothesis.

## 19. Prioritized Next Research Steps

1. **Repeat the Frozen-vs-LoRA Full-MVR continuation with at least two additional seeds and a held-out scene-grouped evaluation.** This directly tests whether the AP75/R20 gains survive seed and selection variance.
2. **Run a controlled MVR data/update-budget study.** Match optimizer steps or sampled images across 1,000 and 5,000 examples to separate label diversity from update count and order effects.
3. **Advance geometry-aware AOS fusion only after resolving projection metadata.** Use thermal-only as the current multimodal reference; equal/weighted RGB fusion has not beaten it.

For a supervisor meeting with limited compute time, priority 1 has the clearest connection to the project's main DINOv3/occlusion hypothesis.

## 20. Conclusions

The project has progressed from dataset validation and simple frozen heads to a verified evaluation framework, strong YOLO baseline, LoRA-adapted DINOv3 FCOS, multi-resolution readouts, independent selective fusion, multimodal studies, and bounded SNN investigations. Three conclusions are supported:

1. Fine-tuned YOLO11s is the strongest complete detector on val600.
2. DINOv3 provides complementary low-visibility detections, and both Full MVR and LoRA improve specific aspects of localization/recall, but gains are modest relative to cost and need multi-seed confirmation.
3. Fusion helps only when carefully constrained: Selective Fusion B raises independent holdout recall, while naive feature averaging and RGB-thermal averaging can destroy useful evidence.

The research is reproducible at the pilot level, but it is not yet an official full-split ForestPersons benchmark or an edge-deployment result. The next defensible claim should come from repeated, scene-aware Frozen-vs-LoRA Full-MVR evaluation rather than another unbounded architecture search.

## Evidence Index

- Canonical inventory: `docs/experiment_registry/experiments.csv`
- Dataset and published benchmarks: `docs/reports/forestpersons_phase1_phase2_2026-10-12.md`
- MVR ablation: `docs/reports/dinov3_mvr_native_ablation_results_2026-10-23.md`
- Label efficiency: `docs/reports/dinov3_mvr_label_efficiency_results_2026-10-23.md`
- Frozen-vs-LoRA metrics: `$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/frozen_vs_lora/evaluation_val600_seed20261012/metrics.json`
- Independent fusion: `docs/reports/forestpersons_yolo_dinov3_selective_fusion_independent1200_2026-10-15.md`
- ANN2SNN: `docs/reports/dinov3_mvr_ann2snn_val600_2026-10-22.md`
- Multi-view: `docs/reports/dinov3_multiview_feature_fusion_benchmark49_2026-10-17.md`
- AOS: `docs/reports/aos_dataset_geometry_multiview_audit_2026-10-18.md` and `$LOGXY_ARTIFACT_ROOT/outputs/aos/`
- DTSS: `docs/reports/dtss_learning_validation_2026-10-20.md` and `$LOGXY_ARTIFACT_ROOT/outputs/dtss_detection/`
