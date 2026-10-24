# DINOv3 MVR Native Readout Ablation - Implementation Readiness

Date: 2026-10-22
Status: implemented and smoke-tested; full training not started

## Purpose

The completed MVR experiment changed intermediate-layer and multi-resolution
readout together. The new `mvr_native` arm isolates a native-grid multi-layer
readout between the existing final-layer baseline and full MVR:

1. `baseline`: final block 12 features directly into FCOS.
2. `mvr_native`: blocks 3/6/9/12 projected and concatenated at native stride 16.
3. `mvr`: unchanged blocks 3/6/9/12 with native/fine/coarse context.

## Implementation

`DINOv3MVRNativeReadout` uses four independent 1x1 Conv/ReLU projections from
384 to 64 channels. Their 256-channel native-grid concatenation is mapped by a
zero-initialized 1x1 convolution to a 384-channel residual over the normalized
final-layer feature. No upsampling, pooling, fine branch, or coarse branch is
present.

The native readout has 197,248 parameters. With the unchanged 524,646-parameter
FCOS head, the arm has 721,894 trainable parameters. Existing baseline and
full-MVR models remain 524,646 and 923,622 parameters respectively.

The existing `DINOv3MVRReadout` class and all its state-dict names are unchanged.
The trainer records `arm=mvr_native`, `head=mvr_native_fcos`, layer indices
3/6/9/12, and an explicit readout description. Resume validation remains strict.
The evaluator includes the native checkpoint only when
`--mvr-native-checkpoint` is supplied, so historical two-arm invocations remain
valid. Evaluation now also emits R20/R40/R70/R100 and includes the 0.425
FP/image budget in addition to 0.15/0.25/0.35.

## Compatibility and tests

- Existing epoch-20 baseline checkpoint strict-loaded successfully.
- Existing epoch-20 full-MVR checkpoint strict-loaded successfully with its old
  state dictionary.
- Recomputed matched FCOS initialization SHA-256 exactly equals the completed
  full-MVR checkpoint: `5ef06d7af762954d4e9cf8bc89a6c5545136d4e7f39488bf40ab9a346ff7a2d5`.
- Focused MVR/ANN2SNN tests: 14 passed.
- Maintained suite: 190 passed, 3 skipped.
- Both updated CLIs load and show the new optional mode.

## Real-cache MPS smoke

Two train5000 samples were loaded from the existing final-layer and blocks
3/6/9 caches. Two AdamW updates used the established FCOS focal, Smooth L1, and
centerness losses.

| Measurement | Result |
|---|---:|
| Device | Apple MPS |
| Losses | 1.25248, 1.04025 |
| Native projection gradient after first update | finite and nonzero |
| Runtime for two samples | 0.927 s |
| Observed MPS driver memory | 1.043 GiB |
| Checkpoint/output written | no |

This validates the training path but is not a performance experiment.

## Exact training command - not executed

The GitHub-ready source tree and local artifact tree are separate. The command
below runs the new source against the existing local dataset/caches and writes a
new non-overwriting experiment directory.

```bash
REPO="${LOGXY_REPO:-$PWD}"
WORK="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
cd "$REPO"

caffeinate -i env PYTHONPATH="$REPO/src:$REPO" \
  "$WORK/.venv-dinov3/bin/python" scripts/train_dinov3_mvr_fcos.py \
  --arm mvr_native \
  --root "$WORK/data" \
  --manifest "$WORK/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json" \
  --final-cache-root "$WORK/outputs/dinov3_detector/features/train_5000_seed20261012_short672" \
  --mvr-cache-root "$WORK/outputs/dinov3_mvr/features/train5000_layers3_6_9_short672" \
  --output-dir "$WORK/outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012" \
  --epochs 20 --lr 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  2>&1 | tee "$WORK/outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012.log"
```

Resume after an interruption by repeating the exact command with `--resume`
and changing `tee` to `tee -a`.

## Exact three-arm evaluation command - not executed

```bash
REPO="${LOGXY_REPO:-$PWD}"
WORK="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
cd "$REPO"

PYTHONPATH="$REPO/src:$REPO" "$WORK/.venv-dinov3/bin/python" \
  scripts/evaluate_dinov3_mvr_fcos.py \
  --root "$WORK/data" \
  --manifest "$WORK/outputs/subsets/forestpersons_val_600_seed20261010_manifest.json" \
  --final-cache-root "$WORK/outputs/dinov3_detector/features/val_600_short672" \
  --mvr-cache-root "$WORK/outputs/dinov3_mvr/features/val600_layers3_6_9_short672" \
  --baseline-checkpoint "$WORK/outputs/dinov3_mvr/train5000_baseline_fcos_e20_seed20261012/dinov3_head.pt" \
  --mvr-native-checkpoint "$WORK/outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012/dinov3_head.pt" \
  --mvr-checkpoint "$WORK/outputs/dinov3_mvr/train5000_mvr_fcos_e20_20261021_213321/dinov3_head.pt" \
  --output-dir "$WORK/outputs/dinov3_mvr/evaluation_val600_readout_ablation_seed20261012" \
  --device mps
```

## Decision gate

The arm is ready for the controlled 20-epoch cached-feature run. Approval is
still required before executing it. Existing checkpoints and result directories
must remain unchanged.
