# Finding People Under Dense Vegetation with DINOv3

Research code and reproducibility documentation for the LogXY internship on
detecting partially or heavily occluded people in UAV imagery. ForestPersons is
the primary benchmark; DINOv3 feature readouts, lightweight detectors,
multi-view/multimodal fusion, anomaly detection, and hybrid spiking inference
are investigated incrementally.

All ForestPersons `train5000`/`val600` numbers are selected-subset pilot results,
not official full-split benchmark scores.

## Research Objective

The project studies person detection in UAV forest imagery, with emphasis on
people visible at only 20% or 40%. The main questions are whether DINOv3 dense
features improve occlusion recall, whether lightweight adaptation and
multi-level readouts improve localization, and whether DINOv3 adds useful
complementary detections to a conventional YOLO detector.

## Current Results

### Frozen DINOv3 MVR + FCOS

Matched 20-epoch training on train5000, evaluated on val600:

| Model | AP | AP50 | AP75 | AR@100 |
|---|---:|---:|---:|---:|
| Frozen final-layer DINOv3 + FCOS | 0.3267 | 0.7915 | 0.2022 | 0.4550 |
| Frozen DINOv3 MVR + FCOS | **0.3623** | **0.8162** | **0.2675** | **0.4900** |

At the exploratory 0.15 FP/image budget, MVR recall/R20 were 0.7210/0.4885,
compared with 0.6916/0.4425 for the matched baseline.

### Frozen versus LoRA Full MVR continuation

Two matched additional epochs from the same trained Full-MVR checkpoint:

| Arm | AP | AP50 | AP75 | AR@100 | Recall / R20 at <=0.425 FP/image |
|---|---:|---:|---:|---:|---:|
| Frozen DINOv3 + MVR/FCOS | 0.3533 | 0.8082 | 0.2362 | 0.4846 | 0.8505 / 0.6264 |
| Fresh rank-4 q/v LoRA + MVR/FCOS | **0.3652** | **0.8144** | **0.2722** | **0.4873** | **0.8531 / 0.6552** |

This is a one-seed continuation study, not a scratch-training or statistical
significance result. LoRA used 3.68 hours and 11.89 GiB observed peak MPS
driver memory, versus 1.37 hours and 2.33 GiB for the frozen arm.

### Hybrid ANN2SNN MVR inference

The trained MVR checkpoint was converted only at its six ReLU readout sites.
Frozen signed DINOv3 features, residual normalization, GroupNorm/GELU FCOS,
classification, centerness, box regression, decoding, and NMS remain analog.
Train-only calibration was reused for all modes; val600 used fixed confidence
0.63 and no threshold tuning.

| Mode | AP50 | Recall | R20 | R40 | R70 | R100 | FP/image | ms/image | Spike activity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN | **0.8162** | **0.8077** | **0.5862** | **0.7765** | **0.9016** | 0.9292 | 0.3183 | **13.0** | n/a |
| SNN T=4 | 0.7622 | 0.7156 | 0.4425 | 0.6765 | 0.8135 | 0.8821 | 0.2700 | 23.2 | 0.0246 |
| SNN T=8 | 0.8002 | 0.7891 | 0.5632 | 0.7412 | 0.8756 | 0.9340 | 0.2967 | 31.1 | 0.0320 |
| SNN T=16 | 0.8083 | 0.7997 | **0.5862** | 0.7529 | 0.8808 | **0.9387** | 0.2933 | 49.1 | 0.0356 |

Spike activity is an activity proxy only. It does not demonstrate hardware
energy savings, and MPS latency is not neuromorphic deployment latency.

### YOLO11 and selective fusion

The internal YOLO11s baseline was fine-tuned for 47 completed epochs on
train5000. On val600 it achieved AP 0.5541, AP50 0.8941, AR@100 0.6339, and
recall/R20 0.9039/0.7701 at confidence 0.05 and 0.3467 FP/image. It remains the
strongest single detector in the project.

Frozen Selective Fusion B combines supported YOLO and DINOv3 FCOS predictions.
On the image-disjoint holdout1200 it increased recall from 0.9062 to 0.9185 and
R20 from 0.7859 to 0.8035, while AP and AR decreased. Fusion is therefore a
recall-oriented optional mode, not a replacement for YOLO. The holdout is not
proven scene-disjoint.

## Detection Pipeline

The canonical DINOv3 path is:

```text
RGB image
  -> DINOv3 ViT-S/16 patch tokens
  -> optional rank-4 q/v LoRA adaptation
  -> final-layer or multi-level/multi-resolution (MVR) readout
  -> lightweight FCOS classification, centerness, and L/T/R/B regression
  -> confidence filtering and NMS
  -> COCO-format person detections
```

The Full MVR readout uses blocks 3, 6, 9, and 12 and derives native, fine, and
coarse feature grids. The frozen-backbone version can train from cached
features. LoRA changes backbone activations and therefore requires online image
forward passes; detached caches cannot train LoRA.

Hybrid ANN2SNN converts only six MVR ReLU sites. The DINOv3 encoder, residual
normalization, most of FCOS, analog box regression, decoding, and NMS remain
analog. Separate DTSS experiments use a compact spiking transformer and are
indexed in the registry.

See [the experiment registry](docs/experiment_registry/experiments.csv),
[the comprehensive evidence audit](docs/reports/logxy_comprehensive_experiment_comparison_2026-10-24.md),
[MVR preparation and protocol](docs/reports/dinov3_frozen_mvr_fcos_training_preparation_2026-10-21.md),
and [ANN2SNN val600 results](docs/reports/dinov3_mvr_ann2snn_val600_2026-10-22.md).

## Repository Map

```text
src/forestpersons/       Reusable datasets, models, metrics, and fusion modules
scripts/                 Training, evaluation, analysis, and preparation CLIs
tests/                   Focused unit and integration tests
docs/reports/            Experiment reports and verified conclusions
docs/experiment_registry Canonical experiment status and artifact index
docs/audit/              Repository and reproducibility audits
experiments/             Preserved experiment configs and external adapters
data/                    Local-only datasets; ignored except README files
models/                  Local-only weights; ignored except README
outputs/                 Local-only caches/checkpoints/results; ignored except README
```

Start with [CODE_REVIEW_GUIDE.md](docs/CODE_REVIEW_GUIDE.md) to study the
pipeline in implementation order. `PROGRESS.md` is the chronological research
log; the experiment registry is the authoritative concise index.

## ForestPersons Data

The official annotation totals are 67,686/18,243/10,553 images for
train/validation/test. The project maps all pose/visibility categories to one
`person` class while retaining visibility metadata. Main local subsets are:

| Subset | Images | Purpose |
|---|---:|---|
| train400 | 400 | Controlled pilots |
| train5000 | 5,000 | Main local training set |
| val600 | 600 | Selected pilot evaluation; 749 GT people |
| holdout1200 | 1,200 | Image-disjoint selective-fusion confirmation |

ForestPersons is gated. Dataset files are never distributed by this repository.
After authorized access, use this local layout:

```text
data/
  annotations/{train,val,test}.json
  images/...
outputs/subsets/*.json
```

Resolve annotations by split-qualified image identity and `image_id`, with a
filename verification step. Raw numeric IDs are split-local. See
[the Phase 1/2 report](docs/reports/forestpersons_phase1_phase2_2026-10-12.md)
for category mapping, subset statistics, and leakage checks.

## Environment

The primary environment was verified on an Apple M4 MacBook Pro with Python
3.13.5 and PyTorch MPS:

```bash
python3 -m venv .venv-dinov3
source .venv-dinov3/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src:$PWD"
```

YOLO experiments use a separate environment:

```bash
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install -r requirements-yolo.txt
```

The recorded primary environment uses Python 3.13.5, PyTorch 2.9.0,
torchvision 0.24.0, Transformers 5.19.0, and MPS on an Apple M4. Exact pinned
packages are in `requirements.txt`; test dependencies are in
`requirements-dev.txt`. CUDA-only published-baseline preparation is isolated
under `experiments/published_baselines_cuda/` and is not required for the main
Mac workflow.

Run the maintained test suite:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q tests
```

## Data and Checkpoints

Datasets, annotations, images, feature caches, checkpoints, prediction dumps,
and generated experiment directories are excluded from Git. Their expected
local layout is documented in [data/README.md](data/README.md),
[outputs/README.md](outputs/README.md), and each experiment report.

ForestPersons is gated at
[etri/ForestPersons](https://huggingface.co/datasets/etri/ForestPersons). Do not
redistribute its images or annotations. DINOv3 and other pretrained weights
must also be obtained from their official sources under applicable terms.

## Reproducible Commands

Commands run from the repository root. Keep large artifacts outside the Git
checkout if desired:

```bash
export LOGXY_REPO="$PWD"
export LOGXY_ARTIFACT_ROOT="/path/to/local/LogXY-artifacts"
export PYTHONPATH="$LOGXY_REPO/src:$LOGXY_REPO"
```

### Train cached-feature final-layer or Full-MVR FCOS

The required feature caches must already exist. The commands below use the
actual trainer interface and never overwrite an existing output directory.

```bash
python scripts/train_dinov3_mvr_fcos.py \
  --arm baseline \
  --root "$LOGXY_ARTIFACT_ROOT/data" \
  --manifest "$LOGXY_ARTIFACT_ROOT/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json" \
  --final-cache-root "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_detector/features/train_5000_seed20261012_short672" \
  --output-dir "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/example_baseline" \
  --epochs 20 --seed 20261012 --device mps

python scripts/train_dinov3_mvr_fcos.py \
  --arm mvr \
  --root "$LOGXY_ARTIFACT_ROOT/data" \
  --manifest "$LOGXY_ARTIFACT_ROOT/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json" \
  --final-cache-root "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_detector/features/train_5000_seed20261012_short672" \
  --mvr-cache-root "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/features/train5000_layers3_6_9_short672" \
  --output-dir "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/example_full_mvr" \
  --epochs 20 --seed 20261012 --device mps
```

Add `--resume` only when resuming the exact same configuration and output
directory.

### Evaluate matched frozen readouts

```bash
python scripts/evaluate_dinov3_mvr_fcos.py \
  --root "$LOGXY_ARTIFACT_ROOT/data" \
  --manifest "$LOGXY_ARTIFACT_ROOT/outputs/subsets/forestpersons_val_600_seed20261010_manifest.json" \
  --final-cache-root "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_detector/features/val_600_short672" \
  --mvr-cache-root "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/features/val600_layers3_6_9_short672" \
  --baseline-checkpoint "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/example_baseline/dinov3_head.pt" \
  --mvr-checkpoint "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/example_full_mvr/dinov3_head.pt" \
  --output-dir "$LOGXY_ARTIFACT_ROOT/outputs/dinov3_mvr/example_evaluation" \
  --device mps
```

The evaluator saves COCO AP/AP50/AP75/AR@100, confidence sweeps, visibility
recall, predictions, and matched-FP summaries. The canonical one-class
evaluation uses IoU 0.50 one-to-one matching for custom recall and FP/image;
COCO AP remains AP@[0.50:0.95].

For online LoRA+MVR continuation, use
`scripts/train_dinov3_lora_mvr_fcos.py`; its exact completed protocol and
commands are documented in
`docs/reports/dinov3_mvr_frozen_vs_lora_readiness_2026-10-23.md`.

## Reproducibility and Reports

- `docs/experiment_registry/experiments.csv` is the authoritative status and
  artifact index. Prepared scripts are not completed experiments.
- `docs/results/forestpersons_latest.json` is the lightweight current-results
  summary.
- `docs/reports/logxy_comprehensive_experiment_comparison_2026-10-24.md`
  consolidates verified findings and protocol limitations.
- Seeds, manifests, initialization paths, thresholds, and output paths are
  recorded in each experiment report and saved run configuration.
- Generated checkpoints and metrics are excluded from Git. A clone reproduces
  code and documentation, not private data or the local artifact store.

## Known Limitations

- val600 is a repeatedly inspected selected subset, not the official benchmark.
- Most comparisons use one seed; small differences are not significance claims.
- Published full-dataset results are not directly comparable to train5000
  pilots.
- The corrected train5000 annotation count is 6,415; one historical YOLO
  conversion artifact records 6,431 because of the earlier filename lookup.
- Holdout1200 is image-disjoint but not conclusively scene-disjoint.
- EfficientAD lacks verified normal-only ForestPersons training imagery.
- ANN2SNN spike activity is not a hardware-energy measurement.
- No project-wide code license has yet been approved.

## Reproducibility Rules

- Preserve official dataset splits and split-qualified image identities.
- Resolve annotations by `image_id`, verifying filenames where manifests exist.
- Keep val600 results labeled as exploratory selected-subset results.
- Do not infer completed experiments from prepared scripts; consult the registry.
- Never commit credentials, dataset files, feature caches, model weights, or
  generated prediction artifacts.
- Do not overwrite experiment outputs; use a new output directory.

## Licensing

No project-wide open-source license has yet been approved. See
[LICENSES.md](LICENSES.md) before sharing or reusing code. Dataset and model
licenses remain independent of this repository.
