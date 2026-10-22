# DTSS ForestPersons train5000/val600 Training Readiness

Date: 2026-10-20  
Status: implemented and validated; no training run

## Protocol

The existing lightweight DTSS patch detector is unchanged: 128 x 128 RGB
input, 16-pixel patches, 32-dimensional tokens, one Spiking Transformer block,
four attention heads, and `T_max=4`. Two separately launched arms use the same
seed, deterministic epoch-specific image order, architecture, initialization,
train5000 data, optimizer, and 20-epoch budget:

- `fixed_t4`: all timestep masks remain fixed at four active steps.
- `learned_dtss`: timestep parameters are trainable and retain the established
  normalized mask penalty (`0.02`).

The established pilot settings remain unchanged: AdamW LR 0.002, weight decay
0, physical batch size 4, box-loss weight 5, global gradient clipping 5, fixed
confidence 0.05, and NMS IoU 0.50. This is a scaling run, not a new model.

Every epoch writes `epoch_NNN.pt` and atomically updates `last.pt`. Checkpoints
contain model and optimizer state, completed epoch, history, deterministic
initial-state digest, active timestep counts, runtime, and MPS memory. `--resume`
loads `last.pt`, validates every shared setting except the allowed final epoch
extension, and resumes at the next epoch with the same precomputed order.

After epoch 20, each command evaluates val600 and saves AP50, AR@100,
recall/precision at IoU 0.50 and confidence 0.05, recall at visibility 20,
FP/image, active timesteps, runtime, and peak observed MPS driver memory in
`evaluation.json`.

## Data verification

| Split | Images | Evaluable annotations | Excluded | Identity overlap |
|---|---:|---:|---:|---:|
| train5000 | 5,000 | 6,415 | 0 | 0 |
| val600 | 600 | 749 | 1 `unknown_0` | 0 |

Samples and annotations are resolved by `image_id`, with filename verification.
This prevents duplicate filenames from attaching annotations from another
image ID. All referenced files are present locally.

## Launch commands

Run the arms sequentially on MPS from the repository root; do not run
both simultaneously on the 16 GB Mac.

### Fixed T=4

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dtss_forestpersons_full.py \
  --arm fixed_t4 \
  --epochs 20 \
  --device mps \
  --output-dir outputs/dtss_detection/train5000_val600_fixed_t4_e20_seed20261020 \
  2>&1 | tee outputs/dtss_detection/train5000_val600_fixed_t4_e20_seed20261020.log
```

### Learned DTSS

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dtss_forestpersons_full.py \
  --arm learned_dtss \
  --epochs 20 \
  --device mps \
  --output-dir outputs/dtss_detection/train5000_val600_learned_dtss_e20_seed20261020 \
  2>&1 | tee outputs/dtss_detection/train5000_val600_learned_dtss_e20_seed20261020.log
```

### Resume after interruption

Repeat the corresponding command with `--resume` and append to the log. For
example:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dtss_forestpersons_full.py \
  --arm learned_dtss \
  --epochs 20 \
  --device mps \
  --resume \
  --output-dir outputs/dtss_detection/train5000_val600_learned_dtss_e20_seed20261020 \
  2>&1 | tee -a outputs/dtss_detection/train5000_val600_learned_dtss_e20_seed20261020.log
```

The same pattern applies to `fixed_t4` by changing the arm and output path.

## Validation

- 16 focused DTSS/detector tests passed.
- Full runner and CLI passed Python compilation/import checks.
- Actual manifests, image paths, annotation totals, visibility filtering, and
  split-qualified overlap were validated without loading or training a model.
- No checkpoints or experiment outputs were created by this preparation task.

DTSS masks remain an algorithmic timestep-selection measurement: the current
eager PyTorch implementation computes all four temporal tensors before masking,
so fewer active steps do not yet imply proportional wall-clock savings.

