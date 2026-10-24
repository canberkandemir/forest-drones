# DINOv3 Full MVR: frozen versus LoRA readiness

## Decision and controlled protocol

Experiment 3 is prepared but has not been executed. Both arms strict-load the
same strongest completed Full MVR + FCOS checkpoint from Experiment 1:

`outputs/dinov3_mvr/train5000_mvr_fcos_e20_20261021_213321/dinov3_head.pt`

That checkpoint achieved AP 0.3623, AP50 0.8162, AP75 0.2675, and AR@100
0.4900 on the reused val600 pilot. The Experiment 2 train5000 checkpoint is
preserved but is not selected because its different deterministic image order
produced AP 0.3353, demonstrating a material order effect.

Both arms use the original train5000 manifest and receive two matched online
epochs at short side 672, batch size 1, accumulation 4, seed 20261012, identical
image order, FCOS targets, losses, clipping, and MVR/FCOS learning rate.

| Setting | Frozen control | LoRA experiment |
|---|---:|---:|
| DINOv3 | ViT-S/16 frozen | ViT-S/16 original weights frozen |
| Adapter | none | fresh rank-4 q/v LoRA, alpha 16, dropout 0.05 |
| MVR + FCOS | trainable; shared checkpoint init | trainable; same checkpoint init |
| Trainable parameters | 923,622 | 997,350 (923,622 head + 73,728 LoRA) |
| LR | MVR/FCOS 3e-4 | MVR/FCOS 3e-4; LoRA 1e-4 |
| Optimizer | AdamW, weight decay 1e-4 | same |
| Scheduler | none | none |
| Loss | mean focal, 2x Smooth L1, centerness BCE | same |
| Optimizer steps | 2,500 expected | 2,500 expected |

The frozen arm also runs online rather than reading detached caches. This makes
preprocessing, selected block outputs (3/6/9/12), image order, and head updates
identical. The LoRA arm alone retains gradients through those block outputs.
Fresh LoRA has a zero-initialized B matrix, so both arms begin with identical
detector predictions.

## Implemented safeguards

- Online differentiable extraction uses the same one-based Hugging Face hidden
  states and normalization as the frozen MVR cache extractor.
- Original backbone parameters are rejected if trainable in either arm; only
  q/v LoRA parameters are accepted in the adapted backbone.
- Output directories refuse overwrite. Atomic epoch checkpoints contain model,
  adapter, optimizer, history, manifest hash, initialization hash, image-order
  hash, runtime, optimizer steps, and peak MPS driver memory.
- Resume verifies arm, manifest, and initial head hashes.
- Evaluation refuses comparison unless initialization, manifest, epoch count,
  optimizer steps, shared settings, and per-epoch image orders match.
- Evaluation uses online features for both arms and the established val600
  protocol: COCO AP/AP50/AP75/AR@100 plus IoU 0.50 one-to-one confidence sweeps
  and matched FP/image budgets 0.15/0.25/0.35/0.425.

## Execution commands

Run from the repository. The two training arms should run sequentially.

```bash
cd "${LOGXY_REPO:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
ROOT="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
PY="$ROOT/.venv-dinov3/bin/python"
INIT="$ROOT/outputs/dinov3_mvr/train5000_mvr_fcos_e20_20261021_213321/dinov3_head.pt"
TRAIN="$ROOT/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json"
VAL="$ROOT/outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"
```

Frozen control:

```bash
$PY scripts/train_dinov3_lora_mvr_fcos.py \
  --arm frozen --root "$ROOT/data" --manifest "$TRAIN" \
  --init-mvr-checkpoint "$INIT" \
  --output-dir "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/train5000_frozen_mvr_cont_e2_seed20261012" \
  --epochs 2 --gradient-accumulation 4 --short-side 672 \
  --lr-head 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  2>&1 | tee "$ROOT/outputs/dinov3_mvr/frozen_vs_lora_frozen_train.log"
```

Fresh-LoRA arm:

```bash
$PY scripts/train_dinov3_lora_mvr_fcos.py \
  --arm lora --root "$ROOT/data" --manifest "$TRAIN" \
  --init-mvr-checkpoint "$INIT" \
  --output-dir "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/train5000_lora_r4_qv_mvr_cont_e2_seed20261012" \
  --epochs 2 --gradient-accumulation 4 --short-side 672 \
  --lr-lora 1e-4 --lr-head 3e-4 --weight-decay 1e-4 \
  --lora-rank 4 --lora-alpha 16 --lora-dropout 0.05 \
  --seed 20261012 --device mps \
  2>&1 | tee "$ROOT/outputs/dinov3_mvr/frozen_vs_lora_lora_train.log"
```

Evaluation, only after both `final.pt` files exist:

```bash
$PY scripts/evaluate_dinov3_lora_mvr_fcos.py \
  --root "$ROOT/data" --manifest "$VAL" \
  --frozen-checkpoint "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/train5000_frozen_mvr_cont_e2_seed20261012/final.pt" \
  --lora-checkpoint "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/train5000_lora_r4_qv_mvr_cont_e2_seed20261012/final.pt" \
  --output-dir "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/evaluation_val600_seed20261012" \
  --device mps \
  2>&1 | tee "$ROOT/outputs/dinov3_mvr/frozen_vs_lora_evaluation.log"
```

To resume an interrupted arm, repeat its command with `--resume`; do not change
the output directory or shared configuration.

## Runtime, memory, and limitations

The prior online ViT-S LoRA FCOS train5000 run required 3.31 hours and about
10.6 GiB peak MPS driver memory. Full MVR retains four block outputs and adds a
923,622-parameter trainable readout/head. Conservative estimates are 2-3 hours
for the frozen online arm, 3.5-4.5 hours for LoRA, 1-2 hours for two online
val600 passes, and 6.5-9.5 hours total. LoRA peak memory is expected around
10.5-12.5 GiB; sustained pressure or non-finite loss should stop the run.

This is continuation from a 20-epoch trained head, not training from scratch.
It isolates the incremental value of fresh q/v LoRA over an equally continued
frozen control. Val600 is repeatedly inspected, matched-budget thresholds are
exploratory, and a gain below 0.01 AP/AR requires additional seeds before a
research claim. The frozen control is computationally more expensive than cache
training by design, but it is the cleanest matched online comparison.

## Validation

The actual selected checkpoint strict-loaded as Full MVR with 923,622 head
parameters. The train5000 and val600 manifests contain 5,000 and 600 unique
image IDs/files, all 5,600 files exist, and their filename overlap is zero.
The planned output paths did not exist. Focused LoRA/MVR and label-efficiency
tests passed 25/25; no training, inference, cache generation, checkpoint write,
commit, or push was performed.
