# Outputs Directory

This directory contains generated artifacts from LogXY ForestPersons
experiments: manifests, metrics, checkpoints, plots, visualizations, cached
features, Colab bundles, and exploratory outputs.

Generated contents are intentionally excluded from Git. GitHub-safe result
summaries live under `docs/reports/` and `docs/results/`; paths below describe
the canonical local working layout.

Results here are organized historically, not yet perfectly by the official
five-phase work plan. For the canonical experiment index, use:

- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

## Phase-Oriented Map

- Phase 1 dataset preparation:
  - `subsets/`
  - `visualizations/`
- Phase 2 published and reproduced baselines:
  - `yolo11_baseline/`
  - `yolo11_validation_grid/`
  - `yolo11_finetune/`
  - `baseline_feasibility/`
  - `rx_anomaly/`
  - `colab_upload/`
- Phase 3 DINOv3 development:
  - `dinov3_detector/`
  - `dinov3_lora/`
  - `dinov3_patch_knn/`
- Legacy exploratory work:
  - `yolo_exploration/`
  - `dinov3_exploration/`
- Shared verified evaluation outputs:
  - `evaluation_framework/`

## Canonical Versus Historical Artifacts

Canonical pilot outputs include the selected `train5000` and `val600` subset
manifests, verified evaluation-framework outputs, YOLO11 fine-tuning outputs,
DINOv3 train5000 outputs, DINOv3 patch-kNN outputs, and RX anomaly outputs.

Historical or exploratory outputs should be preserved but not cited as current
scientific results unless the experiment registry marks them as completed.

All selected `val600` metrics are pilot results, not official full-split
ForestPersons benchmark scores.

## Preservation Rules

Do not overwrite, move, delete, or compact outputs unless a migration has been
explicitly approved. When adding new experiments, create a clearly named new
output directory and update the experiment registry.
