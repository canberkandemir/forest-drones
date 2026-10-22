# LogXY Repository Audit — 2026-10-12

This audit is a read-only map of the current LogXY ForestPersons repository before
any cleanup or migration. It records what exists, what has actually run, and how
to reorganize safely without breaking reproducibility.

## Scope

Inspected areas:

- Root documentation: `README.md`, `AGENTS.md`, `PROGRESS.md`
- Source code: `src/forestpersons/`
- Scripts: `scripts/`
- Experiments: `experiments/`
- Outputs and reports: `outputs/`
- Tests: `tests/`
- Local datasets: `data/`
- Local checkpoints and exploratory scripts: `models/`, `test_dinov3.py`, `test_yolo.py`

No files were moved, renamed, deleted, downloaded, trained, or evaluated as part
of the audit.

## Current Repository Shape

Top-level project areas:

- `src/forestpersons/`: reusable code for dataset loading, detection
  evaluation, DINOv3 detector heads, DINOv3 patch-kNN anomaly scoring, and RX
  anomaly scoring.
- `scripts/`: reproducible scripts for dataset subsets, visualization, YOLO,
  DINOv3, anomaly baselines, evaluation, and CUDA/MMDetection preparation.
- `experiments/published_baselines_cuda/`: prepared CUDA/Colab workflow for DETR
  R50 and Deformable Faster R-CNN R50-FPN/DCN.
- `outputs/`: generated experiment outputs, metrics, plots, checkpoints,
  feature caches, visualizations, and Colab upload bundles.
- `data/`: official ForestPersons annotations plus approved local train/val
  image subsets. This is not the full ForestPersons image dataset.
- `docs/papers/`: local copies of research papers and dataset references.
- `docs/specs/`: intended location for internship specifications, but root
  specification PDFs still exist and should be moved only after approval.

Approximate disk usage observed during the audit:

- Full project: about 16 GB
- `data/`: about 2.7 GB
- `data/images/`: about 2.5 GB
- `outputs/`: about 13 GB
- `outputs/dinov3_detector/`: about 13 GB
- DINOv3 feature cache: about 13 GB
- `.venv-yolo/`: about 346 MB
- `.venv-dinov3/`: about 161 MB

## Experiment Classification

See `docs/experiment_registry/experiments.csv` for the path-level registry. The
high-level classification is below.

### Phase 1 — Dataset Preparation

Completed pilot-scale dataset work:

- Official ForestPersons `train.json`, `val.json`, and `test.json` are present
  under `data/annotations/`.
- A reusable COCO-style dataset loader exists in `src/forestpersons/dataset.py`.
- The current approved local image subsets are present under `data/images/`.
- The selected `train5000` and `val600` manifests are present under
  `outputs/subsets/`.
- Visualization scripts and examples are present under `scripts/` and
  `outputs/visualizations/`.

Not completed:

- The full official ForestPersons image dataset has not been downloaded.

### Phase 2 — Published And Reproduced Baselines

Completed pilot experiments:

- YOLO11 pretrained baseline on a small 32-image validation smoke subset.
- YOLO11n/YOLO11s pretrained baseline grid on the selected `val600` subset.
- YOLO11s fine-tuned on the selected `train5000` subset and evaluated on
  `val600`.
- RX RGB anomaly scoring baseline on `val600`.

Prepared but not successfully run:

- DETR R50 CUDA/Colab workflow.
- Deformable Faster R-CNN R50-FPN/DCN CUDA/Colab workflow.

Feasibility only:

- EfficientAD. No EfficientAD experiment has actually been executed.

Important status notes:

- DETR and Deformable Faster R-CNN are not reproduced ForestPersons benchmark
  results yet. The current files are setup/configuration/inference preparation.
- The selected `val600` results are pilot results and are not official
  full-split ForestPersons benchmark scores.

### Phase 3 — DINOv3 Decoders And Feature Experiments

Completed pilot experiments:

- Frozen DINOv3 ViT-S/16 feature extraction and caching for `train400`,
  `train5000`, and `val600`.
- DINOv3 detection head v1.
- DINOv3 grid/focal head v2.
- DINOv3 FCOS-style head v3 trained on `train400`.
- DINOv3 FCOS-style head trained on `train5000`.
- DINOv3 linear patch detection head trained on `train5000`.
- DINOv3 patch-kNN anomaly scoring on `val600`.

Not implemented:

- DINOv3 LoRA.
- DINOv3 full-dataset training.

Status note:

- The FCOS-style DINOv3 work should be preserved as an existing prototype, but
  the official work plan now prioritizes simpler DINOv3 decoders and published
  baseline reproduction before further architecture development.

### Phase 4 — Fusion

No temporal, multi-view, or RGB-thermal fusion experiments were found.

### Phase 5 — Edge Deployment

No edge deployment, export, quantization, or device-runtime experiments were
found.

## Completed Versus Partial Work

Completed and reusable:

- Dataset loader and annotation validation.
- Dataset subset manifests and approved local image subsets.
- Verified COCO-style and custom evaluation framework.
- YOLO11 pretrained and fine-tuned pilot evaluations.
- DINOv3 FCOS, linear patch, and patch-kNN pilot experiments.
- RX anomaly pilot experiment.
- CUDA/Colab packaging tests for published baseline preparation.

Partial or unexecuted:

- DETR R50 inference/training on Colab.
- Deformable Faster R-CNN R50-FPN/DCN inference/training on Colab.
- EfficientAD anomaly baseline.
- DINOv3 LoRA.
- Fusion.
- Edge deployment.
- Full ForestPersons train/validation benchmark.

## Confusing Or Duplicate Areas

These paths should be preserved for now, but they should be labeled or migrated
carefully later:

- Root PDFs: `Internship_DINOv3_DIC.pdf` and
  `Internship_DINOv3_People_Under_Vegetation.pdf` are outside `docs/specs/`.
- `README.md` and `AGENTS.md` were originally Phase-1-focused and need to point
  future sessions to the audit and experiment registry.
- `outputs/dinov3_detector/` contains multiple generations of heads, threshold
  sweeps, feature caches, plots, and historical rechecks in one large folder.
- `outputs/dinov3_detector/README.md` is v1-specific but sits at the root of the
  DINOv3 output tree.
- `outputs/dinov3_detector/head_linear_patch_train5000_seed20261013` appears to
  be an interrupted or incomplete linear-head run; the `_rerun` directory is the
  completed run.
- `outputs/dinov3_detector/head_linear_patch_train5000_smoke` is smoke-only.
- `outputs/colab_upload/LogXY` and `outputs/colab_upload/LogXY 2` look like
  duplicate Colab upload bundles.
- `outputs/yolo_exploration/` and `outputs/dinov3_exploration/` are legacy
  exploratory outputs.
- `runs/detect/` contains Ultralytics-style output outside the main organized
  output tree.
- `src/forestpersons/detection.py` overlaps with the newer verified
  `src/forestpersons/evaluation.py`; callers should eventually converge on the
  verified evaluator, but the older file should not be removed until references
  are checked.
- `.DS_Store`, `__pycache__`, and `.pytest_cache` files/directories are present
  and should stay ignored by version control.

## Proposed Clean Structure

Target structure aligned with Davide's work plan:

```text
docs/
  specs/
  papers/
  audit/
  experiment_registry/

src/forestpersons/
  data/
  evaluation/
  models/
    dinov3/
  anomaly/

scripts/
  data/
  evaluation/
  baselines/
    yolo/
    mmdetection/
    anomaly/
  dinov3/

experiments/
  phase1_dataset/
  phase2_published_baselines/
  phase3_dinov3_decoders/
  phase4_fusion/
  phase5_edge/

outputs/
  phase1_dataset/
  phase2_baselines/
  phase3_dinov3/
  phase4_fusion/
  phase5_edge/

archive/
  exploratory/
  interrupted/
  legacy/
```

## Safe Migration Plan

Recommended order:

1. Keep existing data, checkpoints, cached features, and experiment outputs in
   place.
2. Use this audit and the experiment registry as the current map of record.
3. Update root documentation to point future Codex sessions to the registry.
4. Add small README files to high-risk folders before moving anything.
5. Move documentation/spec PDFs first, because this is low risk.
6. Group scripts only after checking imports and command references.
7. If outputs are moved later, preserve compatibility with old paths using
   symlinks or explicit migration notes.
8. After any path migration, run the full test suite and at least one evaluation
   smoke test.

## Recommended Next Step

Do not move large outputs yet. The next safest cleanup step is to add folder
README files for:

- `data/`
- `outputs/`
- `outputs/dinov3_detector/`
- `outputs/colab_upload/`
- `scripts/`

Then, in a separate approved step, move the root internship PDFs into
`docs/specs/` and update links.
