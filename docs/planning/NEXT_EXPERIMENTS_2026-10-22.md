# LogXY Next Experiments - 2026-10-22

## Ranked experiments

### 1. MVR readout ablation - execute next

**Hypothesis.** Intermediate DINOv3 layers provide most of the current MVR
gain, while fine/coarse context contributes an additional localization benefit.

**Why first.** This directly explains the strongest recent DINOv3 result using
existing caches. It requires one new lightweight arm rather than another
backbone run.

**Arms.**

1. Existing final-layer baseline FCOS.
2. New selected-layer native-only residual readout using blocks 3/6/9/12.
3. Existing full selected-layer native/fine/coarse MVR.

The new arm should reuse four 384-to-64 projections, concatenate 256 channels,
and map a native-resolution residual back to 384 channels. Its final residual
convolution must be zero-initialized. The same deterministic fresh FCOS state,
training order, loss, optimizer, and 20-epoch budget used by the completed
baseline/full-MVR experiment must be retained.

**Reusable code and data.**

- `src/forestpersons/dinov3_detector.py`
- `scripts/train_dinov3_mvr_fcos.py`
- `scripts/evaluate_dinov3_mvr_fcos.py`
- `tests/test_dinov3_mvr.py`
- Existing blocks 3/6/9 and block-12 train5000/val600 caches
- Existing baseline and full-MVR checkpoints

**Implementation status.** The approved minimal extension is complete. The CLI
accepts `baseline`, `mvr_native`, and `mvr`; the evaluator accepts the native
checkpoint optionally, preserving legacy two-arm commands. Zeroing trained
multi-resolution branches after training is not used because it would not be a
fair ablation.

**Protocol.** Train the new arm on train5000 for 20 epochs, seed 20261012,
AdamW LR 3e-4, weight decay 1e-4, batch 1, focal alpha 0.75/gamma 2, Smooth L1
box weight 2, centerness BCE, and gradient clip 5. Evaluate unchanged val600
with AP/AP50/AP75/AR@100 and recall/R20/R40/R70/R100 at FP/image budgets
0.15/0.25/0.35/0.425. Keep val600 selection explicitly exploratory.

**Success criteria.**

- Multi-layer evidence: native-only improves AP or AR@100 by at least 0.01
  over the final-layer baseline without reducing R20 at matched FP/image.
- Multi-resolution evidence: full MVR improves AP or AR@100 by at least 0.01,
  or R20 by at least 0.02, over native-only.
- If differences are smaller, repeat all arms over three seeds before claiming
  a component effect.

**Estimated cost.** One cached-feature arm should take approximately 1-2 hours
on MPS. Evaluation should take minutes. No new feature extraction is needed.

**Prepared command.** The implementation and smoke validation are complete,
but this full command has not been executed:

```bash
cd /path/to/LogXY
caffeinate -i env PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/train_dinov3_mvr_fcos.py \
  --arm mvr_native \
  --manifest outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json \
  --final-cache-root outputs/dinov3_detector/features/train_5000_seed20261012_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/train5000_layers3_6_9_short672 \
  --output-dir outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012 \
  --epochs 20 --lr 3e-4 --weight-decay 1e-4 --seed 20261012 --device mps \
  2>&1 | tee outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012.log
```

The evaluator must then compare that checkpoint with the existing baseline and
full-MVR checkpoints. No existing checkpoint should be retrained or overwritten.
The intended post-extension evaluation command is:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/evaluate_dinov3_mvr_fcos.py \
  --manifest outputs/subsets/forestpersons_val_600_seed20261010_manifest.json \
  --final-cache-root outputs/dinov3_detector/features/val_600_short672 \
  --mvr-cache-root outputs/dinov3_mvr/features/val600_layers3_6_9_short672 \
  --baseline-checkpoint outputs/dinov3_mvr/train5000_baseline_fcos_e20_seed20261012/dinov3_head.pt \
  --mvr-native-checkpoint outputs/dinov3_mvr/train5000_mvr_native_fcos_e20_seed20261012/dinov3_head.pt \
  --mvr-checkpoint outputs/dinov3_mvr/train5000_mvr_fcos_e20_20261021_213321/dinov3_head.pt \
  --output-dir outputs/dinov3_mvr/evaluation_val600_readout_ablation_seed20261012 \
  --device mps
```

`--mvr-native-checkpoint` is optional, so the historical baseline/full-MVR
evaluation command remains valid.

### 2. Label-efficiency study

**Hypothesis.** The selected frozen MVR readout retains more of its train5000
performance than YOLO when labels are reduced.

**Dataset.** Nested train100/train400/train1000/train5000 manifests derived
from train5000 using sequence-aware sampling, balanced visibility targets, and
three deterministic seeds. Do not combine existing unmatched train400 metrics
with the new curve.

**Protocol.** Same model initialization, 20 epochs, losses, resolution, and
validation protocol for every subset. Report per-seed and mean/dispersion.
Run a matched YOLO curve only if Davide wants a direct label-efficiency claim.

**Hardware/cost.** Cached MVR heads are Mac-feasible. A three-seed MVR curve is
estimated at about 9 hours total. Matched YOLO adds substantial compute.

**Metrics.** AP, AP50, AP75, AR@100, R20/R40, recall at fixed FP/image, learning
curves, and performance as a fraction of train5000.

**Success.** MVR should preserve a larger fraction of its 5000-image AP/AR or
R20 than the comparator at 100/400/1000 labels. Absolute metrics still matter;
fractional retention alone is insufficient.

**Risks.** Sequence correlation, multiple annotations per selected image, and
subset composition can dominate small-data results. Three seeds are required.

### 3. Frozen vs LoRA MVR

**Hypothesis.** Rank-4 q/v LoRA adds in-domain occlusion sensitivity beyond the
best frozen MVR readout.

**Protocol.** Start both arms from the same trained MVR readout/FCOS checkpoint.
Continue a frozen-backbone control online for two epochs and compare it with a
fresh-LoRA arm using identical order, accumulation, optimizer steps, and losses.
Do not reuse detached caches for LoRA optimization.

**Hardware/cost.** MPS is feasible for ViT-S. Prior LoRA train5000 required
3.31 hours for two epochs and about 10.6 GiB MPS driver memory. Two sequential
arms are expected to require 6-8 hours.

**Metrics/success.** Same detector metrics plus trainable parameters, memory,
runtime, and three-seed confirmation if AP/AR gain is below 0.01. LoRA should
improve R20/R40 at matched FP/image without reducing AP/AR.

**Risk.** MVR intermediate features must be extracted online with gradients;
checkpoint initialization and normalization must exactly match the frozen arm.

### 4. Hybrid ANN2SNN optimization

**Hypothesis.** Better calibrated conversion boundaries can reach T8-level
latency/activity with T16-level accuracy.

**Protocol.** Keep checkpoint and train-only calibration fixed. Measure
activation reconstruction error at each converted site, then test predeclared
site groups and T4/T8/T16. Evaluate all 600 images once per configuration.

**Metrics.** AP50, AP75 where supported, recall, R20/R40, FP/image, latency,
peak memory, activity, and approximation error. Energy remains unmeasured.

**Hardware.** Mac CPU/MPS supports methodology validation. A real energy claim
requires neuromorphic or target edge hardware. Spikify can be assessed as an
input encoding utility, but it is not an ANN-to-SNN detector converter.

**Success.** At least 99% ANN AP50 and no more than 0.01 absolute overall recall
loss at T8 or lower, followed by actual runtime/power benefit on target hardware.

### 5. Geometry-aware AOS multi-view fusion

**Hypothesis.** Official virtual-camera projection can align static scene
features more reliably than image homographies and avoid ForestPersons fusion
failures.

**Blocker.** The local release lacks the scene DEM/OBJ, its transform into the
COLMAP frame, and exact virtual-camera/focal-surface configuration. Raw RGB
frames also lack directly released person boxes.

**Next valid action.** Request those files and reproduce one official training
scene projection. Do not train or report raw-frame recall before that gate.

**If unblocked.** Compare single-view, geometry-warped feature fusion, and
official integral images on scene-disjoint splits with reference-view labels.
Use shared frozen DINOv3 and the same FCOS head.

### 6. RGB/thermal fusion

**Hypothesis.** Thermal-dominant fusion can retain the strong thermal signal
while using RGB only where it is complementary.

**Existing evidence.** On the small AOS canonical test, thermal detected 26/57
people, RGB 0/57, and mean fusion 5/57 at validation-locked thresholds. Mean
fusion lowered candidate confidence rather than adding complementary evidence.

**Prepared experiment.** Fresh matched FCOS heads for thermal-only, alpha 0.75,
and alpha 0.90 cached-feature fusion; O6 validation chooses the arm/threshold,
and canonical test remains untouched until final evaluation.

**Limitation.** AOS TRAIN/VALID augmentations represent only 37/5 underlying
people. This is a feasibility study, not a broad multimodal benchmark. A
stronger claim requires a paired dataset such as WiSARD/RGBTDronePerson with
verified calibration, labels, access, and license.

## Rigorous evaluation protocol

Every detector experiment should record:

- COCO AP@[0.50:0.95], AP50, AP75, AR@1/10/100.
- IoU-0.50 one-to-one recall, precision, TP/FP/FN, and FP/image.
- R20/R40/R70/R100 and GT counts.
- Recall at predeclared FP/image budgets, selecting thresholds only on
  calibration/development data.
- Localization IoU and center/scale errors by visibility and box size.
- Parameters, trainable parameters, peak memory, and steady latency.
- Exact manifest, seed, preprocessing, checkpoint hashes, and code revision.
- Per-seed results and uncertainty for controlled claims.

Training, calibration, development validation, and final test identities must
be explicit. No threshold may be selected on the final test set.

## Three highest-priority experiments

1. MVR readout ablation.
2. Nested label-efficiency curve.
3. Matched frozen-vs-LoRA MVR comparison.

## Questions for Davide

1. Is the immediate scientific claim MVR component attribution, label
   efficiency against YOLO, or maximum ForestPersons recall?
2. Should the label-efficiency study include costly matched YOLO retraining, or
   focus first on DINOv3-only scaling?
3. Can LogXY contact the AOS authors for the missing geometry and original 3D
   person contours?
4. Is non-commercial DEIMv2 use acceptable inside the internship, and should a
   license request be initiated before fine-tuning?
5. Which edge target is available: a specific Jetson, CPU, or neuromorphic
   platform? What latency, memory, and power budgets matter?
6. Is access to a CUDA GPU possible for full-resolution ViT-L satellite and
   published DETR-family work, or should those remain deferred?
7. Which dataset can serve as the final untouched, scene-disjoint evaluation
   set for the December report?
