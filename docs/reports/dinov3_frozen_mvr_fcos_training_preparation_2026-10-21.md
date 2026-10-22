# Frozen DINOv3 MVR + FCOS Training Preparation

Date: 2026-10-21  
Status: implemented and validated; no extraction, training, or evaluation run

## Controlled comparison

The experiment compares two fresh matched heads on ForestPersons train5000 and
val600:

1. Frozen DINOv3 final layer (block 12) + canonical stride-16 FCOS.
2. Frozen DINOv3 blocks 3/6/9/12 + residual multi-layer/multi-resolution
   readout + the same canonical stride-16 FCOS.

Both FCOS heads are initialized from the exact same deterministic state and
use the established frozen-baseline budget: 20 epochs, AdamW LR 3e-4, weight
decay 1e-4, seed 20261012, batch size 1, focal alpha 0.75/gamma 2 with the
historical mean reduction, Smooth L1 log-distance regression weighted 2.0,
centerness BCE, and gradient clipping 5. The backbone is never loaded by the
trainer and cannot receive gradients.

## MVR readout

The existing block-12 cache is reused. A new cache stores only normalized
blocks 3/6/9, avoiding duplication of the final layer. Expected additional
storage is about 37 GB based on the existing 12.3 GB train/validation cache;
approximately 101 GiB was free during preparation.

Each intermediate layer is projected from 384 to 64 channels. Their
concatenation is processed at native resolution, 2x resolution followed by
2x average pooling, and half resolution followed by nearest upsampling. A 1x1
convolution maps the concatenated contexts to a 384-channel residual over the
normalized final layer. The residual convolution is zero-initialized, so the
MVR arm presents exactly the baseline feature to its identically initialized
FCOS head at step zero.

The readout's trainable graph uses Conv2d, ReLU, average pooling, nearest-neighbor
resize, concatenation, and residual addition. The readout and FCOS are separate
checkpoint namespaces, providing a clear ANN2SNN conversion boundary. No
conversion is implemented here. Full-detector conversion will still need a
controlled treatment of the canonical FCOS GroupNorm/GELU operations rather
than silently replacing them in this comparison.

| Arm | Trainable parameters |
|---|---:|
| Baseline FCOS | 524,646 |
| MVR readout | 398,976 |
| MVR + FCOS | 923,622 |

## Checkpointing

Each trainer saves `epoch_NNN.pt`, atomically updates `last.pt`, and writes
`dinov3_head.pt` after epoch 20. Checkpoints include model/optimizer state,
history, completed epoch, runtime, peak MPS memory, configuration, and a digest
of the shared FCOS initialization. `--resume` validates arm, initialization,
and all settings except an allowed extension of the final epoch count.

## Exact commands

Run sequentially from the repository root.

### 1. Cache intermediate train features

```bash
mkdir -p outputs/dinov3_mvr
caffeinate -i env PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/cache_dinov3_mvr_features.py \
  --manifest outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json \
  --image-root data/images \
  --final-cache-root outputs/dinov3_detector/features/train_5000_seed20261012_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/train5000_layers3_6_9_short672 \
  --layers 3,6,9 --short-side 672 --device mps \
  2>&1 | tee outputs/dinov3_mvr/cache_train5000.log
```

### 2. Cache intermediate validation features

```bash
caffeinate -i env PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/cache_dinov3_mvr_features.py \
  --manifest outputs/subsets/forestpersons_val_600_seed20261010_manifest.json \
  --image-root data/images \
  --final-cache-root outputs/dinov3_detector/features/val_600_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/val600_layers3_6_9_short672 \
  --layers 3,6,9 --short-side 672 --device mps \
  2>&1 | tee outputs/dinov3_mvr/cache_val600.log
```

Both cache commands are restart-safe: existing files are skipped unless
`--force` is explicitly supplied.

### 3. Train the matched baseline

```bash
caffeinate -i env PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dinov3_mvr_fcos.py \
  --arm baseline \
  --manifest outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json \
  --final-cache-root outputs/dinov3_detector/features/train_5000_seed20261012_short672 \
  --output-dir outputs/dinov3_mvr/train5000_baseline_fcos_e20_seed20261012 \
  --epochs 20 --lr 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  2>&1 | tee outputs/dinov3_mvr/train5000_baseline_fcos.log
```

### 4. Train MVR + FCOS

```bash
caffeinate -i env PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dinov3_mvr_fcos.py \
  --arm mvr \
  --manifest outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json \
  --final-cache-root outputs/dinov3_detector/features/train_5000_seed20261012_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/train5000_layers3_6_9_short672 \
  --output-dir outputs/dinov3_mvr/train5000_mvr_fcos_e20_seed20261012 \
  --epochs 20 --lr 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  2>&1 | tee outputs/dinov3_mvr/train5000_mvr_fcos.log
```

To resume either trainer, rerun its exact command with `--resume` and change
`tee` to `tee -a`.

### 5. Evaluate both checkpoints on val600

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/evaluate_dinov3_mvr_fcos.py \
  --manifest outputs/subsets/forestpersons_val_600_seed20261010_manifest.json \
  --final-cache-root outputs/dinov3_detector/features/val_600_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/val600_layers3_6_9_short672 \
  --baseline-checkpoint outputs/dinov3_mvr/train5000_baseline_fcos_e20_seed20261012/dinov3_head.pt \
  --mvr-checkpoint outputs/dinov3_mvr/train5000_mvr_fcos_e20_seed20261012/dinov3_head.pt \
  --output-dir outputs/dinov3_mvr/evaluation_val600_e20_seed20261012 \
  --device mps
```

Evaluation saves COCO-style AP/AP50/AP75/AR@100, confidence sweeps, recall and
R20 at FP/image <= 0.425, runtime, predictions, and a comparison CSV.

## Validation

Six focused MVR/dataset tests passed. They cover exact baseline fallback,
matched FCOS outputs at initialization, gradients, cache round-trip, supported
module types, and dataset integrity. All four implementation files compile and
their CLIs load. No new feature cache, checkpoint, or experiment result was
created.

