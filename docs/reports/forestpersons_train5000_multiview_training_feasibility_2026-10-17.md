# ForestPersons Train5000 Multi-View Training Feasibility

Date: 2026-10-17

## Decision

ForestPersons train5000 contains enough registration-qualified material for a
small, controlled **fusion-module feasibility pilot**, but not enough verified
geometry or identity information for a definitive multi-view result.

The recommended pilot uses 112 training groups and 33 sequence-disjoint
validation groups. It should be treated as exploratory. ForestPersons has no
camera calibration, poses, depth, person track IDs, or global physical-site
identifier, so sequence-disjointness is verifiable but true site-disjointness
and same-person correspondence are not.

## Audit protocol

Group discovery used only:

- split-qualified filenames (`train::<relative file_name>`), never bare
  numeric COCO IDs as global identity;
- sequence folder and parsed frame number;
- fixed offsets -10, -8, -5, -3 and +3, +5, +8, +10; and
- the unchanged benchmark49 ORB/RANSAC registration thresholds.

Annotations and model predictions were not read during group qualification.
The SHA-256 of the 228 qualified groups was frozen before `train.json` was
opened. Person annotations were attached afterward for statistics and split
stratification. No detector output influenced membership.

Registration required at least 40 ratio-test matches, 30 RANSAC inliers, 0.50
inlier ratio, at most 2 registration-scale pixels median reprojection error,
0.70 reference overlap, 0.05 inlier spread in both images, and projected area
ratio between 0.50 and 1.50.

## Dataset statistics

| Quantity | Count |
|---|---:|
| Train5000 images present/readable | 5,000 / 5,000 |
| Sequence folders | 258 |
| Metadata-eligible reference frames | 2,302 |
| Metadata triplet combinations | 12,022 |
| Sequences with a metadata candidate | 197 |
| Registration-qualified references | 228 |
| Sequences with a qualified group | 87 |
| Frame-disjoint three-view groups | 145 |
| ORB/RANSAC source attempts | 9,641 |

The 145 frame-disjoint groups contain 435 unique images and 208 official
reference-view person annotations:

| Visibility | People |
|---|---:|
| 20% | 49 |
| 40% | 47 |
| 70% | 55 |
| 100% | 57 |

The main registration rejection was too few ratio-test matches (7,523 source
attempts), followed by too few RANSAC inliers (314), low inlier ratio (263),
and low projected reference overlap (195). This confirms that temporal
proximity alone is not treated as alignment.

## Physical-region and person correspondence

Passing registration confirms substantial correspondence in the static image
content, not alignment of a moving or depth-varying person. A post-selection
diagnostic projected source annotations through the already frozen
homographies. It did not remove or select groups.

Across the 208 reference people:

- 144 have projected annotation IoU >= 0.50 in at least one source;
- 76 have projected IoU >= 0.50 in both sources;
- 64 have no source annotation overlap at IoU 0.50;
- median minimum-source IoU is 0.332; and
- median maximum-source IoU is 0.662.

There are no person track IDs. These are spatial-overlap diagnostics, not proof
that annotations represent the same physical person. The result agrees with
benchmark49: static-background registration is real, but target correspondence
is unreliable.

## Leakage-safe proposed split

Eligibility was frozen first. A deterministic 100,000-trial sequence-level
stratification then balanced reference visibility without using predictions.

| Split | Groups | Sequences | People | V20 | V40 | V70 | V100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 112 | 67 | 160 | 38 | 36 | 42 | 44 |
| Validation | 33 | 20 | 48 | 11 | 11 | 13 | 13 |

Verified overlap:

- training/validation sequence overlap: 0;
- training/validation split-qualified image overlap: 0;
- train5000/benchmark49 sequence overlap: 0; and
- train5000/pilot8 sequence overlap: 0.

Target correspondence in the proposed training split is present in at least
one source for 112/160 people and in both sources for 64/160. Validation has
at least one-source correspondence for 32/48 and both-source correspondence
for 12/48.

Benchmark49 remains an inspected exploratory development benchmark. It may be
reported as an external exploratory check after training, but not as an
untouched final test.

## Proposed lightweight architecture

```text
3 RGB views
  -> shared frozen DINOv3 ViT-S/16 + frozen rank-4 q/v LoRA
  -> fixed ORB/RANSAC feature-grid warping into the reference view
  -> shared local source-gating module
  -> reference-preserving capped weighted fusion
  -> frozen existing stride-16 FCOS head
  -> reference-view person boxes
```

For each source patch, concatenate reference feature, aligned source feature,
absolute feature difference, cosine agreement, and validity: `3*384 + 2 =
1,154` channels. A shared gate uses:

1. 1x1 convolution, 1,154 to 32 channels;
2. GroupNorm + SiLU;
3. depthwise 3x3 convolution for local context + SiLU; and
4. 1x1 convolution to one source-weight logit.

This has approximately **37,377 trainable parameters**. The same module is
shared by both sources. Validity masks force unsupported weights to zero;
combined source weight is capped at 0.50; and no valid source gives an exact
reference-only fallback. Initialize the output bias near -4.6 so training
starts close to the existing single-view detector instead of equal averaging.

The fusion output is a convex, normalized mixture dominated by the reference
feature. GT boxes and existing FCOS target assignment remain in reference
coordinates.

## Controlled pilot design

Stage 1 trains only the fusion module:

- initialize DINOv3, LoRA, and FCOS from
  `train5000_seed20261012_r4_qv_fcosinit_e2_rerun1`;
- freeze all DINOv3, LoRA, and FCOS parameters;
- use the 112/33 frozen sequence split above;
- short side 672, batch size 1, gradient accumulation 4;
- AdamW, fusion LR 3e-4, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8;
- global gradient clipping at 5.0;
- seed 20261017 and 10 epochs (28 optimizer steps/epoch, 280 total); and
- retain the checkpoint selected by internal validation AR@100, with recall,
  R20 and FP/image reported but not used for repeated tuning.

Use the unchanged FCOS objective from the source checkpoint: focal loss with
alpha 0.75 and gamma 2 using the legacy all-location mean, Smooth L1 on
log-distances with beta 0.25 and weight 2.0, plus centerness BCE. Losses
backpropagate through the frozen FCOS operations into the fusion output, but
FCOS parameters receive no updates.

Compare on identical validation references:

1. single-view reference features;
2. equal feature averaging;
3. fixed selective fusion; and
4. learned fusion.

Report AP/AP50/AP75, AR@100, recall and visibility recall at fixed confidence
0.55, FP/image, rescued/lost GT, latency, and peak memory. Use paired
sequence-cluster uncertainty. Do not tune a threshold on the 33 validation
groups and then present it as independent.

Only if Stage 1 improves paired recall/AR without harmful FP growth should a
separate Stage 2 unfreeze the 524,646-parameter FCOS head at a lower LR. LoRA
and the backbone should remain frozen for that experiment.

## Mac feasibility

The 435 group images are frame-disjoint. Extracting the frozen LoRA features
once should take roughly 3-5 minutes based on benchmark49 throughput. Expected
compressed cache size is about 1 GiB (roughly 2.1 GiB uncompressed float32).
The existing frozen-DINO feature cache cannot be reused because it predates the
trained LoRA adapters.

Fusion-only training from cached features should require well below the 16 GB
limit; the prior three-view inference peak was 2.26 GiB. A conservative
expected pilot runtime is 5-20 minutes after feature extraction. Online
three-view encoding every epoch would instead be approximately 30-45 minutes
and adds no benefit while the encoder is frozen.

These are estimates, not measured training results. A 2-4 group MPS
forward/backward smoke must precede the pilot.

## Limitations and next decision

The audit supports implementation of a small fusion-only pilot, not a claim
that ForestPersons is a reliable multi-view dataset. Risks are:

- only 87 registration-qualified sequences;
- 48 people in the internal validation split;
- no camera geometry, depth, timestamps, tracks, or site IDs;
- 30% of proposed training GT lacks even one projected source overlap at IoU
  0.50; and
- benchmark49 has already influenced algorithm design.

If the learned gate merely reproduces reference-only performance, stop
ForestPersons fusion work. Even if exploratory metrics improve, final
confirmation should move to official AOS geometry and camera metadata rather
than more ForestPersons gate tuning.

## Artifacts

- `scripts/audit_forestpersons_train5000_multiview.py`
- `scripts/analyze_forestpersons_train5000_multiview_correspondence.py`
- `scripts/design_forestpersons_multiview_sequence_split.py`
- `outputs/multiview/forestpersons_train5000_multiview_audit_seed20261017/`

