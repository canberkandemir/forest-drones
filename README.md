# Finding People Under Dense Vegetation with DINOv3

Research code and reproducibility documentation for the LogXY internship on
detecting partially or heavily occluded people in UAV imagery. ForestPersons is
the primary benchmark; DINOv3 feature readouts, lightweight detectors,
multi-view/multimodal fusion, anomaly detection, and hybrid spiking inference
are investigated incrementally.

All ForestPersons `train5000`/`val600` numbers are selected-subset pilot results,
not official full-split benchmark scores.

## Current Results

### Frozen DINOv3 MVR + FCOS

Matched 20-epoch training on train5000, evaluated on val600:

| Model | AP | AP50 | AP75 | AR@100 |
|---|---:|---:|---:|---:|
| Frozen final-layer DINOv3 + FCOS | 0.3267 | 0.7915 | 0.2022 | 0.4550 |
| Frozen DINOv3 MVR + FCOS | **0.3623** | **0.8162** | **0.2675** | **0.4900** |

At the exploratory 0.15 FP/image budget, MVR recall/R20 were 0.7210/0.4885,
compared with 0.6916/0.4425 for the matched baseline.

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

See [the experiment registry](docs/experiment_registry/experiments.csv),
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
