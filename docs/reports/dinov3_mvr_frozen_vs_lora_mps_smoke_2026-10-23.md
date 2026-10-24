# Frozen versus LoRA Full MVR: MPS smoke validation

## Scope

Two bounded real-data smoke runs used the complete train5000 manifest and the
same Full MVR initialization intended for Experiment 3. Each run processed the
first eight images in the deterministic epoch-1 order, performed two real
optimizer steps with accumulation 4, then exited through the smoke-only path.
No `latest.pt`, `final.pt`, full epoch, evaluation, commit, or push was created.

## Results

| Arm | Images | Steps | Mean loss | Positive cells | Max grad norm | Clipped | Runtime | Peak MPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 8 | 2 | 0.632151 | 80 | 0.762267 | 0 | 6.51 s | 1.69 GiB |
| Fresh LoRA | 8 | 2 | 0.635843 | 80 | 0.903228 | 0 | 11.25 s | 5.56 GiB |

Both arms used sample-order SHA-256
`8313d3f9f9917db96bb400c7c07fe0f46aee7a39d44f760ab6fbb82b8f2d84d9`.
All losses, gradients, and gradient norms were finite.

- Frozen: all 32 MVR/FCOS gradient tensors were nonzero at both steps.
- LoRA step 1: all 48 adapter tensors had gradients and 24 were nonzero. This
  is expected because each `lora_b` starts at zero, initially blocking the
  corresponding `lora_a` gradient.
- LoRA step 2: all 48 adapter gradients were nonzero; all 32 MVR/FCOS gradients
  were nonzero at both steps.
- Post-step verification found all 32 head tensors changed in each arm and all
  24 LoRA B matrices became nonzero.

Artifacts:

- `outputs/dinov3_mvr/frozen_vs_lora/smoke_frozen_steps2_20261023/`
- `outputs/dinov3_mvr/frozen_vs_lora/smoke_lora_steps2_20261023/`
- Adjacent `.log` files preserve terminal output.

The smoke peak does not bound full-run peak memory because allocator behavior,
image geometry, and accumulated runtime vary across 5,000 images. The prior
10.6 GiB LoRA run remains the more conservative planning reference.

## Exact smoke commands

These are the commands that produced the verified artifacts. The paths now
exist and the trainer will correctly refuse to overwrite them. Use a new suffix
such as `_rerun1` for an intentional repeat.

```bash
cd "${LOGXY_REPO:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
ROOT="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
PY="$ROOT/.venv-dinov3/bin/python"
INIT="$ROOT/outputs/dinov3_mvr/train5000_mvr_fcos_e20_20261021_213321/dinov3_head.pt"
TRAIN="$ROOT/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json"
```

```bash
$PY scripts/train_dinov3_lora_mvr_fcos.py \
  --arm frozen --root "$ROOT/data" --manifest "$TRAIN" \
  --init-mvr-checkpoint "$INIT" \
  --output-dir "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/smoke_frozen_steps2_20261023" \
  --epochs 2 --gradient-accumulation 4 --short-side 672 \
  --lr-head 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  --smoke-optimizer-steps 2 \
  2>&1 | tee "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/smoke_frozen_steps2_20261023.log"
```

```bash
$PY scripts/train_dinov3_lora_mvr_fcos.py \
  --arm lora --root "$ROOT/data" --manifest "$TRAIN" \
  --init-mvr-checkpoint "$INIT" \
  --output-dir "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/smoke_lora_steps2_20261023" \
  --epochs 2 --gradient-accumulation 4 --short-side 672 \
  --lr-lora 1e-4 --lr-head 3e-4 --weight-decay 1e-4 \
  --lora-rank 4 --lora-alpha 16 --lora-dropout 0.05 \
  --seed 20261012 --device mps --smoke-optimizer-steps 2 \
  2>&1 | tee "$ROOT/outputs/dinov3_mvr/frozen_vs_lora/smoke_lora_steps2_20261023.log"
```

The full-run commands remain unchanged in
`docs/reports/dinov3_mvr_frozen_vs_lora_readiness_2026-10-23.md`.
