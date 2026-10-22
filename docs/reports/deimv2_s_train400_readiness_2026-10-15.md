# DEIMv2-S ForestPersons Train400 Pilot Readiness

Date: 2026-10-15

## Status and decision gate

The controlled DEIMv2-S full-model train400 pilot is technically ready but has
not been started. The DEIMv2 source and weights are licensed for
non-commercial research only. Because this work is performed in a company
internship, the full command is intentionally gated by both
`--noncommercial-research-confirmed` and `--supervisor-confirmed`. Dr. Davide
Baroli or the appropriate LogXY representative must confirm that this pilot is
permitted non-commercial research before those flags are used.

The official COCO checkpoint remains unchanged. All new files use separate
`outputs/deimv2_finetune/` paths.

## Reproducible configuration

The machine-readable configuration is
`experiments/deimv2_local/configs/forestpersons_train400_fullmodel_seed20261015.json`.

| Setting | Prepared value |
| --- | --- |
| Source | Official `Intellindust/DEIMv2_DINOv3_S_COCO` weights |
| Model | Full one-class DEIMv2-S; 9,688,433 trainable parameters |
| Train data | Corrected train400 manifest; 400 images, 524 annotations |
| Calibration | Existing train-derived calibration100, disjoint from train400 and val600 |
| Validation | Existing val600; final epoch only, no val-based checkpoint selection |
| Input | Fixed 640 x 640, ImageNet normalization |
| Augmentation | None for this controlled two-epoch pilot |
| Precision/device | Float32, Apple MPS, explicit CPU fallback for deformable-attention backward |
| Epochs | 2 |
| Physical/effective batch | 1 / 4 through accumulation 4 |
| Seed | 20261015 |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8 |
| DINOv3-distilled backbone LR | 2.5e-5 |
| STA/encoder/decoder LR | 5e-4 |
| Weight decay | 1e-4; zero for normalization, BN, and bias parameters |
| Gradient clipping | Global norm 0.1 |
| Losses | Official MAL 1, L1 5, GIoU 2, FGL 0.15, DDF 1.5; reg_max 32 |
| Scheduler | Pilot-scaled official flat-cosine: 20-step quadratic warmup, flat through step 100, cosine to 0.5x by step 200 |
| Optimizer steps | 200 |
| Model selection | No val600 selection; evaluate final epoch |

The original S schedule has 2,000 warm-up iterations and 132 epochs, which is
not meaningful in a 200-step pilot. The prepared scheduler preserves the
upstream quadratic-warmup/flat/cosine functional form while scaling its phases
to the explicitly recorded 200-step budget. This is a training-schedule
adaptation, not an architecture or loss change.

## Data and initialization checks

- Every train400 sample is resolved by `image_id`; the manifest filename must
  agree with `train.json` before the image is opened.
- Duplicate filenames do not merge annotations: a regression test verifies
  that distinct image IDs retain their own annotations.
- COCO xywh boxes are independently clipped against image width and height,
  then converted to normalized cxcywh. Degenerate boxes are excluded.
- All ForestPersons pose/visibility categories map to internal class label 0.
- The COCO internal person row 0 initializes every one-class encoder and
  decoder classification head and the denoising class embedding. Eleven
  class-dependent tensors are copied; the denoising padding row is zeroed.
- All shape-compatible official pretrained parameters are loaded unchanged.
- Calibration100 has zero filename overlap with train400 and is not used for
  gradient updates.

## Training safety and artifacts

The trainer provides:

- full optimizer, scheduler, model, RNG, history, and resume-position state;
- atomic `latest.pt` replacement every 25 optimizer steps and at each epoch;
- immutable `epoch_001.pt`, `epoch_002.pt`, and final checkpoints;
- JSONL per-image losses, per-update LR/gradient/timing records, epoch means,
  and a final JSON summary;
- finite-loss and finite-gradient checks;
- MPS allocated/driver-memory monitoring;
- refusal to overwrite a non-empty output directory;
- a strict technical-validation cap of 20 images and five optimizer steps;
- refusal to start the full pilot without the supervisor-confirmation flag.

Each resumable checkpoint is approximately 112 MiB because it includes AdamW
state. The expected epoch/latest/final set needs about 450 MiB.

## Twenty-image timing validation

A separate technical validation processed exactly 20 train400 images and five
accumulated optimizer steps. This was not the two-epoch experiment and its
checkpoint is not a result model.

| Measurement | Result |
| --- | ---: |
| Physical images | 20 |
| Optimizer steps | 5 |
| Mean total loss | 34.3155 |
| Finite losses/gradients | Yes |
| Total measured training-loop time | 13.98 s |
| Warm steady-state accumulated-step median | 1.906 s |
| Warm steady-state accumulated-step mean | 2.181 s |
| Peak observed MPS allocated memory | 0.217 GiB |
| Peak observed MPS driver memory | 2.583 GiB |
| CPU fallback | Confirmed for `aten::grid_sampler_2d_backward` |
| Projected 200-step model-loop time | 6.35 min |
| Practical two-epoch estimate | 8-12 min plus calibration/val600 evaluation |

The first two updates included graph/fallback warm-up. The final three update
times were 1.871, 1.942, and 1.858 seconds. One expected PyTorch warning
confirmed CPU fallback; there were no memory or numerical failures.

The timing checkpoint was also loaded through the fine-tuned evaluation path
and evaluated on two calibration images. This validates serialization,
one-class postprocessing, COCO export, local COCO metrics, and pycocotools
integration. The two-image metrics are deliberately not reported as model
performance.

## Evaluation protocol

After training, inference first runs on calibration100 with `--split train`.
`select_deimv2_calibration_threshold.py` refuses validation summaries and
locks the highest-recall measured point satisfying FP/image <= 0.425. That
threshold is then passed unchanged to val600.

Val600 reporting will include:

- local and pycocotools AP, AP50, AP75, and AR@100;
- calibration-locked recall, precision, FP/image, and R20/R40/R70/R100;
- recall-versus-FP/image curve;
- a separate exploratory val600 matched-budget point, clearly identified as
  validation-selected rather than independent evidence.

## Exact post-approval training command

Run only after supervisor confirmation of permitted non-commercial research:

```bash
cd /path/to/logxy-dinov3
mkdir -p outputs/deimv2_finetune
set -o pipefail
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src \
  .venv-deimv2/bin/python scripts/train_deimv2_forestpersons.py \
  --root data \
  --manifest outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json \
  --calibration-manifest outputs/subsets/forestpersons_train_quality_calibration_100_seed20261015_manifest.json \
  --official-repo experiments/deimv2_local/official_DEIMv2 \
  --checkpoint-dir experiments/deimv2_local/checkpoints/DEIMv2_DINOv3_S_COCO \
  --output-dir outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2 \
  --device mps --epochs 2 --batch-size 1 --gradient-accumulation 4 \
  --seed 20261015 --backbone-lr 2.5e-5 --detector-lr 5e-4 \
  --weight-decay 1e-4 --gradient-clip 0.1 \
  --warmup-steps 20 --flat-steps 100 --min-lr-ratio 0.5 \
  --checkpoint-every-steps 25 --log-every-steps 5 \
  --noncommercial-research-confirmed --supervisor-confirmed \
  2>&1 | tee outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2_run.log
```

The prepared output directory does not currently exist; therefore this command
cannot overwrite an older experiment.

## Prepared post-training commands

Calibration:

```bash
PYTHONPATH=src .venv-deimv2/bin/python scripts/evaluate_deimv2_forestpersons.py \
  --root data --split train \
  --manifest outputs/subsets/forestpersons_train_quality_calibration_100_seed20261015_manifest.json \
  --official-repo experiments/deimv2_local/official_DEIMv2 \
  --checkpoint-dir experiments/deimv2_local/checkpoints/DEIMv2_DINOv3_S_COCO \
  --fine-tuned-checkpoint outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/final.pt \
  --output-dir outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/eval_calibration100 \
  --device mps --fp-budget 0.425

THRESHOLD=$(PYTHONPATH=src .venv-deimv2/bin/python \
  scripts/select_deimv2_calibration_threshold.py \
  --summary outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/eval_calibration100/summary.json \
  --fp-budget 0.425 \
  --output outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/selected_operating_point.json \
  --print-threshold)
```

Val600 with the locked threshold:

```bash
PYTHONPATH=src .venv-deimv2/bin/python scripts/evaluate_deimv2_forestpersons.py \
  --root data --split val \
  --manifest outputs/subsets/forestpersons_val_600_seed20261010_manifest.json \
  --official-repo experiments/deimv2_local/official_DEIMv2 \
  --checkpoint-dir experiments/deimv2_local/checkpoints/DEIMv2_DINOv3_S_COCO \
  --fine-tuned-checkpoint outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/final.pt \
  --output-dir outputs/deimv2_finetune/train400_fullmodel_seed20261015_e2/eval_val600 \
  --device mps --fp-budget 0.425 --operating-threshold "$THRESHOLD"
```

## Remaining limitations

- This is a reduced train400 pilot, not reproduction of the official COCO
  schedule or a full ForestPersons benchmark.
- Heavy upstream Mosaic/Mixup/CopyBlend augmentation and EMA are intentionally
  omitted to isolate basic domain adaptation over two epochs.
- CPU fallback makes runtime dependent on MPS/CPU synchronization and the
  installed PyTorch version.
- The calibration and val600 subsets are selected pilot subsets. Final claims
  require broader held-out evaluation.
- The license gate must be resolved before the full command is run.

## Validation

`PYTHONPATH=src .venv-deimv2/bin/python -m pytest -q tests` completed with
111 passing tests in 4.00 seconds. Python compilation, JSON configuration
validation, checkpoint reload, two-image evaluation smoke, and the full-run
supervisor gate all passed. The gate test confirmed that a full command without
`--supervisor-confirmed` exits before creating its output directory.

