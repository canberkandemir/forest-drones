# Published ForestPersons Baselines: CUDA Preparation

This directory prepares minimal CUDA-side experiments for the ForestPersons
published-baseline detector families requested in Davide's plan:

- Deformable Faster R-CNN with ResNet-50/FPN and deformable convolutions.
- Original DETR with ResNet-50.

Nothing in this directory is a completed GPU result. No MMDetection install,
checkpoint download, inference, or training was run on the local M4 Mac.

## Verified Paper Context

From the ForestPersons paper and the prior feasibility audit:

- Deformable R-CNN is described as Deformable Faster R-CNN with a ResNet-50
  backbone.
- DETR is the original DETR-family detector.
- Most baselines except YOLOv11, DINO, and CZDet were implemented with
  MMDetection.
- DINO in the paper is the DETR-family DINO detector implemented with detrex,
  not DINOv3.
- Reported hardware used NVIDIA RTX 3090 GPUs, with DETR-family training on
  NVIDIA A100/A6000-class GPUs.

Paper-reported training settings:

| Model | Optimizer | LR | Batch | Weight decay | Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| Deformable R-CNN | SGD | 2e-2 | 16 | 1e-4 | 12 |
| DETR | AdamW | 1e-4 | 16 | 1e-4 | 150 |

The paper does not provide exact config files or public ForestPersons-trained
checkpoints in our verified materials, so the configs here are faithful
approximations using official MMDetection model-zoo architectures.

## Official Architecture Sources

Recommended MMDetection line:

- `mmdetection==2.28.2`
- `mmcv-full==1.7.1`
- `torch==1.13.1`, `torchvision==0.14.1`, CUDA 11.7

This is chosen because it supports the classic MMDetection 2.x configs and
custom CUDA ops needed by DCN. It is not claimed to be the exact paper version.

Official COCO-pretrained sources:

| Baseline | Official config | Official COCO checkpoint | Meaning |
| --- | --- | --- | --- |
| Deformable Faster R-CNN R50-FPN dconv c3-c5 | `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py` | `https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth` | COCO-pretrained only |
| DETR R50 | `configs/detr/detr_r50_8x2_150e_coco.py` | `https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth` | COCO-pretrained only |

No ForestPersons-trained Deformable R-CNN or DETR checkpoints have been
verified. Generic checkpoint inference must be labeled as COCO-pretrained
transfer, not as reproduction of the published ForestPersons baseline.

## Prepared Dataset Exports

The CUDA experiments reuse our existing one-class COCO exports:

- Train:
  `outputs/baseline_feasibility/forestpersons_train5000_oneclass_coco.json`
  - 5,000 images.
  - 6,415 annotations.
  - Category: `{"id": 1, "name": "person"}`.
- Validation:
  `outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json`
  - 600 images.
  - 749 evaluated annotations.
  - 1 unknown-visibility annotation skipped.
  - Category: `{"id": 1, "name": "person"}`.

Important exporter fix:

- The official train JSON contains duplicate `file_name` values for some images.
- The exporter now prefers manifest `image_id`/`id` when present, avoiding
  accidental over-selection by filename.

## CUDA Setup

Run these commands on a Linux CUDA machine, not on the local Mac:

```bash
cd /path/to/LogXY

conda env create -f experiments/published_baselines_cuda/environment_cuda_mmdet2.yml
conda activate logxy-mmdet2

pip install -U openmim
mim install "mmcv-full==1.7.1"

mkdir -p third_party
git clone --branch v2.28.2 --depth 1 https://github.com/open-mmlab/mmdetection.git third_party/mmdetection
cd third_party/mmdetection
pip install -v -e .
cd /path/to/LogXY

export MMDET_ROOT=/path/to/LogXY/third_party/mmdetection
PYTHONPATH=/path/to/LogXY/src:/path/to/LogXY python scripts/verify_mmdet_cuda_env.py
```

Expected verification:

- `cuda_available: true`
- `mmdet: 2.28.2`
- `mmcv: 1.7.1`
- one NVIDIA CUDA device with enough memory.

## Checkpoint Downloads

Download only after approval on the CUDA machine:

```bash
mkdir -p experiments/published_baselines_cuda/checkpoints

wget -O experiments/published_baselines_cuda/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth \
  https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth

wget -O experiments/published_baselines_cuda/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth \
  https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth
```

These are COCO-pretrained checkpoints, not ForestPersons-trained checkpoints.

## Generic COCO-Pretrained Smoke Inference

Run 10-30 images first.

Deformable Faster R-CNN:

```bash
PYTHONPATH=/path/to/LogXY/src:/path/to/LogXY python scripts/mmdet_infer_forestpersons.py \
  --config "$MMDET_ROOT/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py" \
  --checkpoint experiments/published_baselines_cuda/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth \
  --annotation outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json \
  --image-root data/images \
  --output experiments/published_baselines_cuda/results/deformable_faster_rcnn_coco_val600_smoke30.json \
  --device cuda:0 \
  --max-images 30 \
  --score-thr 0.001 \
  --source-class-index 0
```

DETR:

```bash
PYTHONPATH=/path/to/LogXY/src:/path/to/LogXY python scripts/mmdet_infer_forestpersons.py \
  --config "$MMDET_ROOT/configs/detr/detr_r50_8x2_150e_coco.py" \
  --checkpoint experiments/published_baselines_cuda/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth \
  --annotation outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json \
  --image-root data/images \
  --output experiments/published_baselines_cuda/results/detr_r50_coco_val600_smoke30.json \
  --device cuda:0 \
  --max-images 30 \
  --score-thr 0.001 \
  --source-class-index 0
```

Evaluate smoke predictions with our verified external-results adapter:

```bash
PYTHONPATH=/path/to/LogXY/src python scripts/evaluate_external_coco_results.py \
  --root data \
  --split val \
  --manifest outputs/subsets/forestpersons_val_600_seed20261010_manifest.json \
  --predictions experiments/published_baselines_cuda/results/detr_r50_coco_val600_smoke30.json \
  --output-dir experiments/published_baselines_cuda/results/detr_r50_coco_val600_smoke30_eval
```

For all 600 images, remove `--max-images 30` and write to a separate output
file. Do not call the result a published baseline reproduction unless it uses
ForestPersons-trained weights.

## Fine-Tuning Configs

Prepared but not executed:

- `configs/forestpersons_deformable_faster_rcnn_r50_fpn_dconv_c3-c5_1x_train5000.py`
- `configs/forestpersons_detr_r50_8x2_150e_train5000.py`

Example Deformable Faster R-CNN fine-tuning command:

```bash
PYTHONPATH=/path/to/LogXY/src:/path/to/LogXY python "$MMDET_ROOT/tools/train.py" \
  experiments/published_baselines_cuda/configs/forestpersons_deformable_faster_rcnn_r50_fpn_dconv_c3-c5_1x_train5000.py \
  --work-dir experiments/published_baselines_cuda/work_dirs/deformable_faster_rcnn_r50_train5000 \
  --cfg-options load_from=experiments/published_baselines_cuda/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth
```

Example DETR fine-tuning command:

```bash
PYTHONPATH=/path/to/LogXY/src:/path/to/LogXY python "$MMDET_ROOT/tools/train.py" \
  experiments/published_baselines_cuda/configs/forestpersons_detr_r50_8x2_150e_train5000.py \
  --work-dir experiments/published_baselines_cuda/work_dirs/detr_r50_train5000 \
  --cfg-options load_from=experiments/published_baselines_cuda/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth
```

The configs change `num_classes` to 1 and use our one-class COCO train5000 and
val600 exports. Because the COCO classification head has 80 classes, verify in
the log that mismatched head weights are skipped/reinitialized as expected.

## Estimated GPU Requirements

These are approximate planning estimates, not measured local results:

| Model | Smoke inference | Train5000 fine-tuning | Notes |
| --- | --- | --- | --- |
| Deformable Faster R-CNN R50-FPN DCN | 6-10 GB VRAM should be comfortable for batch 1 inference | 12-16 GB VRAM for batch 2; 24 GB preferred for fewer compromises | DCN requires compiled CUDA/MMCV ops |
| DETR R50 | 8-12 GB VRAM for batch 1 inference | 16-24 GB VRAM for batch 1-2; 24-48 GB preferred | 150 epochs is much longer than Faster R-CNN |

Runtime assumptions:

- 30-image smoke inference should take minutes after the environment is working.
- Val600 inference should likely take tens of minutes or less on a modern CUDA
  GPU, depending on GPU and I/O.
- Deformable Faster R-CNN train5000, 12 epochs: likely 1-4 hours on a single
  RTX 3090/A5000-class GPU.
- DETR train5000, 150 epochs: likely many hours to more than a day on a single
  GPU; closer to paper conditions needs A100/A6000-class hardware.

## Remaining Blockers

- Exact ForestPersons paper configs and trained checkpoints are not available in
  the verified materials.
- CUDA/MMCV custom ops cannot be validated on the local M4 Mac.
- Generic COCO-pretrained inference is useful as a smoke test only.
- Full reproduction requires running these commands on a CUDA machine and
preserving logs, checkpoints, predictions, and evaluation summaries.

## Google Colab Free/T4 Workflow

Task-1 Colab preparation was added after the original CUDA package. The Colab
workflow is intentionally limited to environment setup and 30-image smoke
inference. It does not fine-tune models.

Files:

- `colab/LogXY_ForestPersons_MMdetection_Smoke.ipynb`
- `colab/setup_mmdet2_colab.sh`
- `colab/download_checkpoints_colab.sh`
- `colab/run_smoke30_colab.sh`

### Compatibility Choice

The least disruptive architecture-faithful solution is still MMDetection 2.x:

- `mmdetection==2.28.2`
- `mmcv-full==1.7.1`
- `torch==1.13.1`
- `torchvision==0.14.1`
- CUDA runtime `11.7`
- Python `3.10`

Reasoning:

- The official MMDetection model-zoo configs for both required smoke tests are
  from the MMDetection 2.x config API:
  - `configs/detr/detr_r50_8x2_150e_coco.py`
  - `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`
- The Deformable Faster R-CNN/DCN baseline depends on MMCV custom CUDA ops.
- Current Colab runtimes may use newer Python/PyTorch/CUDA combinations than
  old MMDetection 2.x wheels expect.
- To avoid downgrading or breaking Colab's main runtime, the workflow creates an
  isolated micromamba environment with Python 3.10 and the matching PyTorch /
  MMCV stack, then runs commands through:
  `micromamba run -n logxy-mmdet2 ...`

This preserves the published architecture choices better than migrating to a
different detector or replacing DCN with plain Faster R-CNN. If the old
`mmcv-full==1.7.1` wheel is unavailable for the Colab image at runtime, the
workflow should stop and report the install failure rather than silently
substitute another model.

### What To Upload To Google Drive

For the 30-image smoke test, the recommended minimal upload folder is prepared
locally with:

```bash
PYTHONPATH="$PWD/src:$PWD" \
  .venv-dinov3/bin/python \
  scripts/prepare_colab_smoke_bundle.py \
  --project-root "$PWD" \
  --output-dir outputs/colab_upload/LogXY
```

Upload the resulting folder:

`outputs/colab_upload/LogXY`

to Google Drive as:

`/content/drive/MyDrive/LogXY`

This bundle is about the 30-image smoke test only. It intentionally excludes
virtual environments, cached DINOv3 features, old training outputs, full
train5000 images, and the full ForestPersons dataset.

Required minimum files/directories inside the Drive folder:

- `data/images/`
  - For the smoke bundle, this contains exactly the first 30 images listed in
    `forestpersons_val600_oneclass_coco.json`.
- `data/annotations/val.json`
- `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`
- `outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json`
- `outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json`
- `scripts/mmdet_infer_forestpersons.py`
- `scripts/verify_mmdet_cuda_env.py`
- `scripts/evaluate_external_coco_results.py`
- `src/forestpersons/`
- `experiments/published_baselines_cuda/colab/`

If you want to run all 600 validation images later, upload the full val600 image
subset and switch evaluation back to the full val600 manifest.

### Colab Execution Steps

1. Open `colab/LogXY_ForestPersons_MMdetection_Smoke.ipynb` in Google Colab.
2. Select `Runtime > Change runtime type > GPU`.
3. Confirm that `nvidia-smi` shows an NVIDIA GPU, ideally T4 on Colab Free.
4. Mount Google Drive in the first notebook cell.
5. Set:
   `LOGXY_ROOT=/content/drive/MyDrive/LogXY`
6. Run the setup cell:
   `bash $LOGXY_ROOT/experiments/published_baselines_cuda/colab/setup_mmdet2_colab.sh`
7. Run the verification cell:
   `micromamba run -n logxy-mmdet2 python scripts/verify_mmdet_cuda_env.py`
8. Download COCO-pretrained checkpoints to Drive:
   `bash $LOGXY_ROOT/experiments/published_baselines_cuda/colab/download_checkpoints_colab.sh`
9. Run smoke inference and evaluation:
   `bash $LOGXY_ROOT/experiments/published_baselines_cuda/colab/run_smoke30_colab.sh`

Persistent outputs:

- Checkpoints:
  `experiments/published_baselines_cuda/checkpoints/`
- Smoke predictions and metadata:
  `experiments/published_baselines_cuda/results/colab_smoke30/`
- Smoke evaluation summaries:
  `experiments/published_baselines_cuda/results/colab_smoke30/*_eval/`

### Architecture Support Status

- DETR R50:
  - Faithfully uses the official MMDetection 2.x original DETR R50 COCO config
    and checkpoint for smoke inference.
  - Smoke inference is generic COCO-pretrained transfer, not ForestPersons
    training.
- Deformable Faster R-CNN R50-FPN/DCN:
  - Faithfully uses the official MMDetection 2.x Faster R-CNN R50-FPN dconv
    c3-c5 COCO config and checkpoint.
  - It preserves DCN; it does not silently fall back to plain Faster R-CNN.
  - Requires MMCV DCN CUDA ops to install and load correctly.

Remaining blockers before actual smoke results:

- The notebook has not been run on Colab yet.
- Availability of compatible `mmcv-full==1.7.1` wheels must be confirmed by the
  setup cell in the live Colab runtime.
- Google Drive must contain the required val600 images at the paths referenced
  by `outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json`.
