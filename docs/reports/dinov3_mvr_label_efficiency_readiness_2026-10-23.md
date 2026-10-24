# DINOv3 Full MVR Label-Efficiency Study - Execution Readiness

Date: 2026-10-23
Status: prepared and validated; no training started

## Objective

Measure label efficiency for the completed frozen-DINOv3 Full MVR + FCOS
architecture using exact nested ForestPersons train100/train400/train1000/
train5000 subsets and the unchanged val600 protocol.

## Audit findings

- The historical train400 manifest is not nested in train5000: 397/400 image
  IDs overlap and three images differ.
- Prefixes of the existing train5000 manifest are unusable: its first 100
  images are all target visibility 20, and its first 1000 contain 700 target-
  visibility-20 images.
- A new deterministic order was therefore created from exactly the existing
  train5000 image pool. Within each target-visibility stratum, images are
  round-robined across sequence folders, then the four visibility streams are
  interleaved. Every requested size is an exact prefix of the next.
- ForestPersons numeric image IDs are split-local: 45 values occur in both the
  train5000 and val600 manifests. They are intentionally ignored for cross-split
  identity. Split-qualified filenames and sequence folders have zero overlap.
- All 5,000 source images, final-layer caches, and blocks-3/6/9 MVR caches are
  present and non-empty. Val600 caches were already verified by the completed
  three-arm evaluation.

## Frozen subsets

| Images | GT annotations | Target 20/40/70/100 | Annotation 20/40/70/100 | Sequence folders | Max images/folder |
|---:|---:|---:|---:|---:|---:|
| 100 | 141 | 25 / 25 / 25 / 25 | 30 / 30 / 37 / 44 | 89 | 2 |
| 400 | 519 | 100 / 100 / 100 / 100 | 116 / 118 / 126 / 159 | 222 | 4 |
| 1000 | 1300 | 250 / 250 / 250 / 250 | 285 / 296 / 331 / 388 | 258 | 7 |
| 5000 | 6415 | 1250 / 1250 / 1250 / 1250 | 1423 / 1440 / 1595 / 1957 | 258 | 34 |

Target-image balancing does not imply equal annotation counts because images
may contain multiple people with different visibility levels.

The manifests and audit live under
`outputs/subsets/label_efficiency/`. Selection seed is 20261023. The full
train5000 set is preserved exactly; only its order changes.

## Matched training protocol

All four runs use Full MVR, fresh deterministic initialization, 20 epochs,
AdamW, learning rate `3e-4`, weight decay `1e-4`, focal alpha 0.75, focal gamma
2, box-loss weight 2, gradient clipping 5, and training seed 20261012. The
trainer records and resume-validates the manifest SHA-256, sample count, FCOS
initialization SHA-256, and complete initial model SHA-256.

A no-training four-process simulation verified identical initialization:

- FCOS initialization: `5ef06d7af762954d4e9cf8bc89a6c5545136d4e7f39488bf40ab9a346ff7a2d5`
- Full MVR initialization: `346a513dd0453e645c06002340c9c062e871e9b7d750646f17211dff25125ef6`

The number of optimizer steps necessarily grows with subset size because the
protocol fixes epochs rather than total updates. This tests the practical
effect of labeled-set size under a standard fixed-epoch budget; it does not
isolate labels from optimization-step count.

The previous Full MVR train5000 result remains a useful architectural reference
but is not the matched 5000 endpoint because it used the old manifest ordering.
All four label-efficiency arms must be trained fresh. Estimated MPS runtimes are
approximately 3, 11, 27, and 136 minutes.

## Exact preparation verification

The manifests have already been created. This command is documented for
reproducibility and will refuse to overwrite them:

```bash
REPO="${LOGXY_REPO:-$PWD}"
WORK="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
cd "$REPO"

PYTHONPATH="$REPO/src:$REPO" "$WORK/.venv-dinov3/bin/python" \
  scripts/prepare_dinov3_mvr_label_efficiency.py \
  --source-manifest "$WORK/outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json" \
  --val-manifest "$WORK/outputs/subsets/forestpersons_val_600_seed20261010_manifest.json" \
  --output-dir "$WORK/outputs/subsets/label_efficiency" \
  --seed 20261023 --sizes 100 400 1000 5000 \
  --image-root "$WORK/data/images" \
  --final-cache-root "$WORK/outputs/dinov3_detector/features/train_5000_seed20261012_short672" \
  --mvr-cache-root "$WORK/outputs/dinov3_mvr/features/train5000_layers3_6_9_short672"
```

## Exact training commands - not executed

Run sequentially to avoid MPS contention:

```bash
REPO="${LOGXY_REPO:-$PWD}"
WORK="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
cd "$REPO"

for N in 100 400 1000 5000; do
  OUT="$WORK/outputs/dinov3_mvr/label_efficiency/train${N}_mvr_e20_seed20261012_nested20261023"
  caffeinate -i env PYTHONPATH="$REPO/src:$REPO" \
    "$WORK/.venv-dinov3/bin/python" scripts/train_dinov3_mvr_fcos.py \
    --arm mvr \
    --root "$WORK/data" \
    --manifest "$WORK/outputs/subsets/label_efficiency/forestpersons_train_nested_${N}_seed20261023_manifest.json" \
    --final-cache-root "$WORK/outputs/dinov3_detector/features/train_5000_seed20261012_short672" \
    --mvr-cache-root "$WORK/outputs/dinov3_mvr/features/train5000_layers3_6_9_short672" \
    --output-dir "$OUT" \
    --expected-samples "$N" \
    --epochs 20 --lr 3e-4 --weight-decay 1e-4 \
    --seed 20261012 --device mps \
    2>&1 | tee "${OUT}.log"
done
```

For a genuine resume, repeat the corresponding single training command with
`--resume` and use `tee -a`. Resume rejects a changed manifest, sample count,
initialization, arm, or training configuration.

## Exact evaluation command - not executed

```bash
REPO="${LOGXY_REPO:-$PWD}"
WORK="${LOGXY_ARTIFACT_ROOT:?Set LOGXY_ARTIFACT_ROOT}"
cd "$REPO"

PYTHONPATH="$REPO/src:$REPO" "$WORK/.venv-dinov3/bin/python" \
  scripts/evaluate_dinov3_mvr_label_efficiency.py \
  --root "$WORK/data" \
  --manifest "$WORK/outputs/subsets/forestpersons_val_600_seed20261010_manifest.json" \
  --final-cache-root "$WORK/outputs/dinov3_detector/features/val_600_short672" \
  --mvr-cache-root "$WORK/outputs/dinov3_mvr/features/val600_layers3_6_9_short672" \
  --checkpoint "100=$WORK/outputs/dinov3_mvr/label_efficiency/train100_mvr_e20_seed20261012_nested20261023/dinov3_head.pt" \
  --checkpoint "400=$WORK/outputs/dinov3_mvr/label_efficiency/train400_mvr_e20_seed20261012_nested20261023/dinov3_head.pt" \
  --checkpoint "1000=$WORK/outputs/dinov3_mvr/label_efficiency/train1000_mvr_e20_seed20261012_nested20261023/dinov3_head.pt" \
  --checkpoint "5000=$WORK/outputs/dinov3_mvr/label_efficiency/train5000_mvr_e20_seed20261012_nested20261023/dinov3_head.pt" \
  --output-dir "$WORK/outputs/dinov3_mvr/label_efficiency/evaluation_val600_seed20261012_nested20261023" \
  --device mps
```

The evaluator reuses the canonical IoU 0.50 one-to-one matching, COCO AP/AP50/
AP75/AR@100, visibility recall, confidence sweep, and matched FP/image budgets
0.15/0.25/0.35/0.425. It rejects mislabeled subset sizes or mixed model
initializations.

## Readiness decision

Experiment 2 is ready for sequential execution. No training, inference, feature
generation, download, checkpoint creation, or existing-output modification was
performed during preparation.
