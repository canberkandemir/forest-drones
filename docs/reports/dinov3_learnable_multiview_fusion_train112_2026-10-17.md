# DINOv3 Learnable Multi-View Fusion Pilot

## Scope

This was one exploratory ForestPersons pilot of a trainable feature-fusion
module. It used 112 three-view training groups from 67 sequences and 33
validation groups from 20 disjoint sequences. The validation groups were held
out from fusion-gate training, but their images were part of the train5000 data
used to train the original DINOv3 LoRA + FCOS detector. Results therefore do
not establish independent generalization.

No backbone, LoRA, or FCOS parameter was updated. No data were downloaded.

## Architecture and training

Each of two geometrically warped source maps is compared locally with the
reference map using normalized reference features, normalized source features,
their absolute difference, cosine agreement, and the geometric validity mask.
A shared `1x1 conv -> GroupNorm -> SiLU -> depthwise 3x3 conv -> SiLU -> 1x1
conv` predicts one source logit per patch.

- Trainable parameters: **37,377**.
- Source weights: sigmoid logits multiplied by strict validity masks.
- Combined source weight: capped at **0.50** relative to reference weight 1.0.
- Invalid correspondence: exact reference-feature fallback.
- Initialization: final weight `N(0, 0.001)`, bias `-4.6`, close to reference-only.
- Frozen model: `facebook/dinov3-vits16-pretrain-lvd1689m`, rank-4 q/v LoRA,
  and the trained stride-16 FCOS head.
- Input: short side 672, exact trained-LoRA preprocessing.
- Optimizer: AdamW, LR `3e-4`, weight decay `1e-4`, batch 1, accumulation 4,
  gradient clipping 5.0, seed 20261017.
- Training: 10 epochs, 28 optimizer steps per epoch (280 total).
- Losses: unchanged FCOS focal loss (`alpha=0.75`, `gamma=2`, all-grid mean),
  weighted Smooth L1 box loss, and positive-only centerness BCE.
- Best checkpoint rule, declared before training: highest AR@100, then recall at
  confidence 0.55, then lower FP/image.

The 435 trained-LoRA feature maps were cached in float32 (1.98 GiB) before gate
training. This is valid because the encoder and LoRA were frozen.

## Smoke validation

The required one-group optimizer smoke passed:

- finite loss: 0.632370;
- finite fusion gradients and changed fusion weights;
- zero FCOS parameter gradients and unchanged FCOS state digest;
- exact invalid-mask reference fallback;
- 0.86 s for the first measured step;
- 1.56 GiB observed MPS driver memory in the first smoke process.

## Validation comparison

All methods use the same 33 reference images, 48 GT people, checkpoint,
confidence 0.55, NMS IoU 0.50, and GT matching IoU 0.50. COCO values are from
pycocotools with maxDets `[1,10,100]`.

| Method | TP | FP | FN | Recall | R20 | R40 | R70 | R100 | Precision | FP/img | AP | AP50 | AP75 | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A single view | 41 | 6 | 7 | 0.8542 | 0.8182 | 0.7273 | 0.8462 | 1.0000 | 0.8723 | 0.1818 | 0.6134 | 0.9451 | 0.7742 | 0.7021 |
| B equal mean | 32 | 12 | 16 | 0.6667 | 0.4545 | 0.5455 | 0.6923 | 0.9231 | 0.7273 | 0.3636 | 0.3246 | 0.7158 | 0.2284 | 0.5125 |
| C fixed selective | 39 | 6 | 9 | 0.8125 | 0.7273 | 0.6364 | 0.8462 | 1.0000 | 0.8667 | 0.1818 | 0.5918 | 0.9006 | 0.6923 | 0.7000 |
| D learned selective | 41 | 6 | 7 | 0.8542 | 0.8182 | 0.7273 | 0.8462 | 1.0000 | 0.8723 | 0.1818 | 0.6132 | 0.9440 | 0.7734 | 0.7021 |

Epoch 1 was selected. Learned fusion recovered **0** GT and lost **0** GT
relative to single view. Its tiny metric differences are score/localization
perturbations, not additional detections. Equal averaging lost nine TPs and
added six FPs relative to single view; fixed selective fusion still lost two
TPs.

## Optimization behavior and resources

Training loss decreased from 0.620139 to 0.619815. Validation loss reached its
minimum at epoch 8 (0.621349), while AR@100 and fixed-threshold recall were
unchanged at displayed precision across all epochs. Mean validation source
weight decreased from 0.01677 to 0.01447. The gate neither hit the formal
reference-only collapse threshold (0.005) nor excessive-source threshold
(0.45), but it converged to a functionally near-reference solution.

- Gate training time: **33.11 s** total for 10 epochs.
- Peak MPS driver memory during gate training: **1.10 GiB**.
- Peak MPS driver memory while extracting/caching features: **2.26 GiB**.
- Mean feature extraction: 0.180 s/image.
- Mean geometric alignment: 0.447 s/source.
- Cached-feature head latency: single 0.0059 s/group; learned 0.0127 s/group.
- Approximate online learned-fusion cost from measured components: three
  feature extractions plus two alignments plus fusion/head, about 1.45 s/group.

## Interpretation and decision

The learned gate successfully avoided the destructive behavior of equal
averaging, but it found no demonstrated extra information that improved person
detection. It exactly matched single-view TP/FP/FN and all visibility recalls.
This is a useful negative result: with global background homographies and no
person-specific geometry, minimizing the frozen detector loss favors preserving
the reference representation.

The 33-group validation is small and not independent of detector pretraining,
so it cannot prove that learned fusion never helps. It does show that further
ForestPersons gate tuning is not justified. Per the predeclared decision rule,
stop ForestPersons multi-view fusion experiments and move next to official AOS
data with camera pose/calibration, rectification, confirmed-person ground truth,
and a genuinely scene-disjoint evaluation split.

## Artifacts

- Code: `src/forestpersons/multiview.py`
- Runner: `scripts/train_dinov3_learnable_multiview_fusion.py`
- Summarizer: `scripts/summarize_dinov3_learnable_multiview_fusion.py`
- Experiment: `outputs/multiview/forestpersons_learnable_fusion_train112_seed20261017/`
- Best fusion-only checkpoint: `best_fusion.pt`
- Metrics: `summary.json`, `comparison.csv`, `per_group_comparison.csv`
- Figures: `training_curves.png`, `visual_examples/`

