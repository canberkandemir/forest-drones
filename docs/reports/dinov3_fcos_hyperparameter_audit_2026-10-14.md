# DINOv3 LoRA FCOS Hyperparameter Audit

Date: 2026-10-14

## Scope

This is a read-only audit of the best ForestPersons DINOv3 ViT-S/16 + LoRA +
single-scale stride-16 FCOS detector. No training, inference, checkpoint update,
or dataset download was performed. The only repository changes are this report
and a verified progress note.

Canonical artifacts:

- Training script: `scripts/train_dinov3_lora_fcos.py`
- Detector and target code: `src/forestpersons/dinov3_detector.py`
- LoRA code: `src/forestpersons/dinov3_lora.py`
- Best checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`
- Training summary:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/summary.json`
- Canonical val600 evaluation:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/summary.json`
- Frozen-feature FCOS initialization:
  `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`

The selected val600 reference is AP 0.3587, AP50 0.8139, AP75 0.2379,
AR@100 0.4704, and IoU-0.50 recall 0.8465 at confidence 0.55 and 0.4167
FP/image. Visibility-20 recall is 0.6667. These are selected-subset pilot
results, not official full-validation benchmark results.

## Verified Parameter Table

| Area | Verified value | Evidence | Candidate change |
| --- | --- | --- | --- |
| Backbone | `facebook/dinov3-vits16-pretrain-lvd1689m`; ViT-S/16; hidden size 384; 12 layers; 6 heads | Checkpoint `backbone_model_id`; local HF config; `src/forestpersons/dinov3_detector.py:19` | None for the next experiment |
| Frozen/trainable backbone | Original DINOv3 parameters frozen; q/v LoRA only is trainable | `src/forestpersons/dinov3_lora.py:93-115` | None |
| LoRA targets | `attention.q_proj` and `attention.v_proj` in all 12 blocks, 24 modules | Checkpoint target list and settings | None |
| LoRA rank/scaling | Rank 4, alpha 16, scale 4.0 | Checkpoint `lora_config`; `src/forestpersons/dinov3_lora.py:28-35,48-50` | None |
| LoRA dropout | 0.05 on adapter input | `src/forestpersons/dinov3_lora.py:51,64-68` | Keep fixed |
| Trainable parameters | 73,728 LoRA + 524,646 FCOS = 598,374 | Checkpoint `parameter_summary` | None |
| FCOS head | 384-to-160 1x1 projection; three GroupNorm/GELU stages with two 3x3 convolutions; separate class, box, and centerness outputs | `src/forestpersons/dinov3_detector.py:205-231` | None |
| Head activations | GELU; no ReLU or SiLU; no head dropout | Same source lines | None |
| Backbone activations/dropout | HF config `hidden_act='gelu'`, `drop_path_rate=0.0`; inspected model has zero `nn.Dropout` modules | Local cached HF config and instantiated model inspection | None |
| Optimizer | AdamW, separate LoRA/head groups | `scripts/train_dinov3_lora_fcos.py:240-246,430-433` | Keep fixed in first diagnostic |
| LoRA learning rate | 1e-4, constant | Saved settings | Keep fixed initially |
| Head learning rate | 3e-4, constant | Saved settings | Keep fixed initially |
| AdamW betas | `(0.9, 0.999)` | PyTorch 2.9.0 AdamW defaults; no override in trainer | None |
| AdamW epsilon | `1e-8` | PyTorch 2.9.0 AdamW defaults; no override | None |
| AdamW momentum | No SGD momentum; beta1 0.9 is the first-moment coefficient | Optimizer construction | None |
| Weight decay | 1e-4, applied through AdamW to both groups | Saved settings and optimizer construction | Keep fixed |
| Scheduler | None; learning rates remain fixed | No scheduler object or step in trainer | Do not combine scheduler changes with the first loss test |
| Warmup | None | Trainer and saved settings | Do not add in the first loss test |
| Gradient clipping | Global norm, max 5.0, immediately before each optimizer step | `scripts/train_dinov3_lora_fcos.py:541-553` | Keep fixed; log pre-clip norm in future runs |
| Physical batch | 1 image; variable patch grids | `scripts/train_dinov3_lora_fcos.py:71-74,423-429` | None |
| Gradient accumulation | 4 images | Saved settings; `scripts/train_dinov3_lora_fcos.py:528,541-553` | None |
| Effective batch | 4 images per optimizer update | Batch 1 x accumulation 4 | None |
| Online LoRA epochs/steps | 2 epochs, 1,250 steps/epoch, 2,500 total optimizer steps | Saved history | See evidence assessment below |
| Head pretraining | 20 cached-feature epochs, batch/effective batch 1, 5,000 steps/epoch, 100,000 optimizer steps | Frozen-head checkpoint/settings/history | Already substantial; do not describe the head as trained for only two epochs |
| Resume state | Model and history only; no optimizer or scheduler state in either source or LoRA checkpoint | Inspected checkpoint keys; `src/forestpersons/dinov3_lora.py:182-208` | Any continuation is weight initialization plus a fresh optimizer, not a true resume |
| Classification loss | Binary sigmoid focal loss; alpha 0.75, gamma 2.0 | Saved settings; `scripts/train_dinov3_lora_fcos.py:77-87` | Test positive-count normalization only |
| Classification normalization | Elementwise focal loss averaged over every grid cell with `.mean()` | `scripts/train_dinov3_lora_fcos.py:83-87` | Replace with sum divided by positive count in the controlled arm |
| Box prediction | Positive L/T/R/B distances produced by `softplus` | `scripts/train_dinov3_lora_fcos.py:144-160` | None |
| Box loss | Smooth L1 on `log1p(pred)` versus `log1p(target)`, beta 0.25 | `scripts/train_dinov3_lora_fcos.py:153-160` | Keep fixed; GIoU already failed to improve AP75 |
| Box weight | 2.0 | Saved settings; `scripts/train_dinov3_lora_fcos.py:314` | None |
| GIoU weight | 0.0 in the best run | Best run predates the optional field; trainer default is 0.0 | Keep 0.0 |
| Centerness loss | BCE-with-logits on positive cells only, mean reduction | `scripts/train_dinov3_lora_fcos.py:307-313` | Keep fixed |
| Centerness weight | 1.0 implicit | Total loss at `scripts/train_dinov3_lora_fcos.py:314` | Keep fixed |
| Total loss | `focal + 2 * SmoothL1 + centerness` | Same source line | Change only focal normalization in the proposed arm |
| Assignment grid | Single stride-16 patch grid; center at `(index + 0.5) * 16` | `src/forestpersons/dinov3_detector.py:492-518` | None in the next experiment |
| Center sampling | Radius 1.5 strides = 24 resized pixels | `src/forestpersons/dinov3_detector.py:495,539-545` | Defer |
| Overlap ownership | Smaller-area GT replaces larger-area GT at shared cells | `src/forestpersons/dinov3_detector.py:508,553-570` | Defer |
| Small-object fallback | Nearest grid center when no center lies in the GT candidate region | `src/forestpersons/dinov3_detector.py:547-551` | None |
| Regression units | L/T/R/B divided by stride 16 | `src/forestpersons/dinov3_detector.py:559-571` | None |
| Centerness target | Square root of balanced left/right and top/bottom distance ratios | `src/forestpersons/dinov3_detector.py:563-571` | None |
| Input resizing | Short side 672; width and height independently rounded to multiples of 16; bicubic; no padding | `src/forestpersons/dinov3_detector.py:56-70` | None |
| Coordinate scaling | Separate `scale_x` and `scale_y` for GT and decoded boxes | `src/forestpersons/dinov3_detector.py:33-39,520-524,894-899` | Verified correct |
| Normalization | Processor mean/std, locally verified as ImageNet mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)` | `src/forestpersons/dinov3_detector.py:73-78` and cached processor config | None |
| Patch features | Last patch tokens reshaped to grid and L2-normalized over channels | `src/forestpersons/dinov3_lora.py:164-179` | None |
| Data augmentation | None: RGB load, deterministic resize, tensor conversion, normalization only | Online dataset and preprocessing paths | Add only after the loss-normalization question is resolved |
| Sampling | DataLoader shuffle with seed 20261012; original best run did not store image-order digests | Saved settings; current trainer records digests at line 570, but that was added later | Future comparisons must record matching order hashes |
| Inference score | `sqrt(sigmoid(class) * sigmoid(centerness))` | `src/forestpersons/dinov3_detector.py:846-858` | Keep fixed; prior calibration did not improve R20 under the FP budget |
| Candidate/NMS limits | Confidence filtering, top 300 per image, NMS IoU 0.50 | `src/forestpersons/dinov3_detector.py:851-892`; evaluation settings | Keep fixed |

## Training and Loss Evidence

### Training duration and loss trajectory

The FCOS head was not trained for only two epochs. It first received 20 epochs
and 100,000 updates on cached, frozen DINOv3 features. Its loss continued to
decline from 0.6477 at epoch 15 to 0.6321 at epoch 20. The best online run then
trained LoRA and the head jointly for two epochs and 2,500 updates:

| Online epoch | Total | Classification | Box before x2 weight | Centerness | Time | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.626874 | 0.000637 | 0.016233 | 0.593770 | 5,751 s | 10.53 GiB |
| 2 | 0.621289 | 0.000613 | 0.014137 | 0.592403 | 6,172 s | 10.62 GiB |

The online loss decreased by 0.9%, with a larger relative decrease in box loss
than centerness. This is compatible with remaining optimization headroom, but it
does not prove undertraining because no per-epoch val600 metrics were recorded.
There is also no train/validation loss comparison from which to diagnose
overfitting.

A later same-learning-rate stride-16 control continued the best weights on
train400 for two epochs with a fresh optimizer. It obtained AP 0.3503,
AR@100 0.4714, matched-budget recall 0.8438, and R20 0.6724. Relative to the
best checkpoint this is a small R20/AR increase alongside lower AP and overall
recall. It is evidence against blindly repeating unchanged training.

### Classification-loss scaling

The train5000 images produce 15,955,800 stride-16 grid locations and 50,600
positive locations, or 315.3 grid locations per positive. The classification
loss averages over all locations, while box and centerness average only over
positive locations. At online epoch 2, the scalar objective is approximately:

```text
classification:             0.000613  (0.10% of total)
2 * box:                    0.028273  (4.55% of total)
centerness:                 0.592403 (95.35% of total)
```

Scalar contribution is not identical to gradient contribution, so this does
not prove that classification gradients are always negligible. It does prove
that the current reduction makes classification scale depend on total grid
size and positive density, and that it differs from the loss normalization in
the FCOS and focal-loss papers.

The [FCOS paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf)
normalizes the summed classification loss by the number of positive samples.
The [Focal Loss paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
likewise sums over dense anchors and normalizes by assigned anchors rather than
all anchors. The papers do not establish that this change will improve
ForestPersons, but they make it the most evidence-backed untested optimization
change in the current implementation.

The current alpha 0.75 intentionally weights positive labels more than
background. The focal-loss paper reports alpha 0.25 with gamma 2 for RetinaNet,
but alpha interacts with gamma and detector conventions. Alpha should remain
0.75 in the first normalization test so that only one factor changes.

## Assignment and Visibility Audit

The manifest was designed for 1,250 target images per visibility group, but
image-level balancing does not imply equal GT or equal positive-cell counts.

A fresh read-only replay of the production loader and stride-16 assignment gave:

| Visibility | GT used by trainer | Mean final positives/GT | Median | Center fallbacks | Final zero-positive GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 1,422 | 7.080 | 9 | 4 | 0 |
| 40 | 1,448 | 7.756 | 9 | 1 | 0 |
| 70 | 1,593 | 8.189 | 9 | 0 | 1 |
| 100 | 1,968 | 8.261 | 9 | 2 | 0 |
| Total | 6,431 | 7.868 | 9 | 7 | 1 |

Visibility-20 people receive about 14.3% fewer final positive locations per GT
than visibility-100 people. This is a real supervision imbalance, but it is not
an assignment collapse: every visibility-20 GT receives at least one final
positive. The one zero-positive visibility-70 GT results from final ownership
after overlapping-GT resolution.

The [ATSS paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.pdf)
shows that positive/negative selection can materially affect detector quality.
For this project, however, the stride-8 test increased positive locations and
still reduced every primary metric. Adaptive assignment remains a plausible
later experiment, not the first optimization to try.

Occlusion-aware pedestrian work shows that visible-part attention can help when
full-person evidence is missing, including
[guided channel attention](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.html)
and [mask-guided attention](https://openaccess.thecvf.com/content_ICCV_2019/html/Pang_Mask-Guided_Attention_Network_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.html).
That literature supports the broader hypothesis that 20%-visible people need
different supervision or representation, but those methods are architectural
and are not evidence that a particular ForestPersons hyperparameter will work.

## Confirmed Reproducibility Issue

The official train JSON has 182 duplicated `file_name` values with distinct
image IDs. `OnlineForestPersonsDataset` constructs a dictionary keyed only by
filename at `scripts/train_dinov3_lora_fcos.py:54-55`. For 18 rows in the
train5000 manifest, the dictionary selects the later duplicate image record and
its annotations instead of the manifest's specified image ID.

Consequences verified in this audit:

- Intended manifest annotation counts total 6,415:
  20=1,423, 40=1,440, 70=1,595, 100=1,957.
- The actual trainer-resolved records total 6,431:
  20=1,422, 40=1,448, 70=1,593, 100=1,968.
- The 18 mismatches add a net 16 annotations.
- The train400 manifest has 3 analogous mismatches.
- The val split has no duplicate filenames and val600 has zero mismatches, so
  the saved val600 metric computation is not affected by this lookup issue.

This does not erase the existing checkpoint or its measured val600 results,
but it means its exact training set differs slightly from the manifest's image
ID definition. Future training must resolve manifest rows by `image_id` and
assert that `file_name` also matches. That correction is a prerequisite, not an
experimental variable.

## Prior Negative or Inconclusive Changes

Do not immediately repeat these tests:

| Change | Verified outcome |
| --- | --- |
| Smooth L1 + GIoU | Small train400 recall/R20 increase, no AP75 improvement |
| FPN-style head | AP 0.3062, AP75 0.1622, recall 0.8024, R20 0.6149; worse than reference |
| Centerness score calibration | Class-only rule recovered a few TPs but violated the FP budget; no R20 gain under budget |
| YOLO-style DFL and quality branch | Best AP 0.1870 and AP75 0.0287; substantially worse |
| RoI box refinement | AP75 improved, but matched recall and R20 gains were only about one GT each; rescoring hurt |
| Learned stride-8 adapter | AP 0.2188, AR 0.3943, matched recall 0.6662, R20 0.4598; worse despite more positives |
| Unchanged same-LR train400 continuation | Slight R20/AR movement but lower AP/overall recall; not evidence for blind longer training |

## Evidence Assessment

### Confirmed

- No LR schedule or warmup was used.
- No optimizer state exists for a true continuation.
- The classification term is normalized over all grid cells and is only 0.10%
  of the recorded epoch-2 scalar objective.
- 20%-visibility GT receives fewer positive locations per instance.
- No data augmentation is used.
- The filename-only lookup changes 18 intended train5000 records.
- Previous architectural and scoring changes did not materially improve the
  primary recall objective.

### Hypotheses

- Positive-count-normalized focal loss may improve class/background separation
  and recover some of the 56 confidence misses, including 26 at visibility 20,
  without the indiscriminate FP increase seen in post-hoc score calibration.
- The model may benefit from more online LoRA optimization, but two epochs of
  decreasing training loss alone are insufficient evidence.
- No augmentation may encourage scene/camera overfitting because train5000 is
  dominated by summer, forest, and See3CAM imagery; train-vs-val evidence is
  unavailable.
- Visibility-aware weighting may improve R20, but it can also increase false
  positives and overfit the selected visibility-balanced pilot.

## Ranked Optimization Sequence

| Rank | Controlled change | Expected recall value | Compute | Main risk |
| ---: | --- | --- | --- | --- |
| 1 | Positive-count-normalized focal classification loss, with image-ID-safe lookup | High relative to cost; directly addresses an untested 315x reduction mismatch and confidence misses | Two train400 arms, about 30-35 min total based on measured stride-16 runtime | Larger class gradients may increase FP or destabilize early steps |
| 2 | Scale the winning normalization to train5000 with matched initialization | High if rank 1 succeeds | About 3.3 h per two-epoch arm on this Mac | Expensive; fresh optimizer prevents exact resume |
| 3 | Visibility-aware positive-loss weighting for 20/40 GT | Directly targets R20 | Train400 A/B first | Uses training metadata and may overfit visibility or increase FP |
| 4 | Horizontal flip plus mild photometric augmentation | Medium generalization value | Small runtime overhead | Can damage subtle person cues; needs a clean A/B |
| 5 | Lower-LR extended LoRA/head training with a recorded schedule | Low-to-medium given current evidence | 1.65 h per train5000 epoch | Diminishing returns; no optimizer state; unchanged continuation already weak |
| 6 | Adaptive assignment such as an ATSS-inspired rule | Scientifically plausible | Higher implementation/testing cost | Assignment confound; extra positives alone already failed at stride 8 |

## Single Best Next Experiment

Run a **controlled train400 focal-normalization A/B** from the same best
train5000 LoRA+FCOS weights.

Prerequisite for both arms:

- Change dataset resolution to key manifest rows by `image_id`, verify the
  paired `file_name`, and fail on ambiguity. This fixes the three train400
  manifest mismatches identically in both arms.

Matched protocol:

- Arm A: unchanged grid-mean focal loss.
- Arm B: identical focal elements, but compute
  `sum(focal_loss_over_all_locations) / max(number_of_positive_locations, 1)`.
- Same train400 manifest, image order digest, seed, short side 672, LoRA/head
  checkpoint initialization, AdamW groups, LRs, weight decay, batch 1,
  accumulation 4, clipping 5, Smooth L1, centerness, two epochs, inference, and
  val600 evaluation.
- Keep alpha 0.75 and gamma 2.0 fixed. Do not add warmup, a scheduler,
  visibility weighting, augmentation, or GIoU in this experiment.
- Before the full arms, run a 10-image forward/backward check and log per-loss
  gradient norms to ensure the new normalization remains finite under clipping.

Primary decision metrics:

1. Recall and R20 at FP/image <= 0.425.
2. AR@100.
3. FP/image and precision at the locked fixed threshold.
4. AP/AP50/AP75 as secondary diagnostics.

Scale the change to train5000 only if Arm B improves recall or R20 at the same
FP budget without a material AR/AP collapse. This is the highest-value next
experiment because it tests a documented mismatch with standard dense-detector
loss normalization, costs far less than another train5000 run, and does not
change architecture or post-hoc thresholds.

## Validation

- Documentation whitespace check: passed.
- Maintained test suite: 87 passed in 4.14 seconds with
  `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests -q`.
- An initial invocation without `PYTHONPATH=src` failed during collection
  because the local package was not importable; rerunning with the repository's
  required source path passed all tests.
