# Scripts Directory

This directory contains reproducible command-line entry points for the LogXY
ForestPersons project.

Scripts are currently stored flat for backward compatibility. The official work
plan should be used as the organizing principle when choosing what to run.

For the path-level experiment map, see:

- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

## Phase 1 — Dataset Preparation

- `create_forestpersons_train_subset.py`
- `create_forestpersons_val_subset.py`
- `download_forestpersons_subset.py`
- `download_forestpersons_subset_verified.py`
- `update_manifest_hf_sizes.py`
- `visualize_forestpersons.py`

Prefer `download_forestpersons_subset_verified.py` for reproducible subset
downloads. Do not download the full ForestPersons dataset without explicit
approval.

## Phase 2 — Published And Reproduced Baselines

YOLO11:

- `evaluate_yolo11_forestpersons.py`
- `run_yolo11_validation_grid.py`
- `prepare_yolo_forestpersons.py`
- `summarize_yolo11_finetune.py`

COCO/evaluation adapters:

- `evaluate_forestpersons_framework.py`
- `evaluate_external_coco_results.py`
- `export_forestpersons_coco_subset.py`

CUDA/MMDetection preparation:

- `mmdet_infer_forestpersons.py`
- `prepare_colab_smoke_bundle.py`
- `verify_mmdet_cuda_env.py`

Anomaly baselines:

- `run_rx_anomaly_baseline.py`

## Phase 3 — DINOv3 Development

- `benchmark_dinov3_features.py`
- `cache_dinov3_features.py`
- `train_dinov3_detector_head.py`
- `evaluate_dinov3_detector.py`
- `analyze_dinov3_assignments.py`
- `analyze_dinov3_errors.py`
- `run_dinov3_patch_knn_anomaly.py`

Current research priority after cleanup is ForestPersons DINOv3 development:
LoRA adaptation, lightweight detection heads, and recall under heavy occlusion.

## Phase 4 — Multi-View Fusion

Canonical source: `src/forestpersons/multiview.py`.

- `prepare_forestpersons_multiview_benchmark.py`
- `evaluate_dinov3_multiview_benchmark.py`
- `evaluate_dinov3_selective_multiview_fusion.py`
- `train_dinov3_learnable_multiview_fusion.py`
- `audit_aos_stage1.py`
- `validate_aos_f8_geometry.py`

ForestPersons fusion experiments are complete. Current work is AOS geometry
feasibility; exact official virtual-camera reprojection is not yet available.

## Phase 5 — Edge Deployment

Not started.

## Preservation Rules

Do not rename scripts casually. Many reports, notebooks, tests, and progress
notes refer to current script paths. If scripts are later grouped into
subdirectories, preserve compatibility through wrappers or explicit migration
notes and rerun the test suite.
