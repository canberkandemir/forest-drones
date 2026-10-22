# LogXY Code Review Guide

This guide describes the maintained execution path for ForestPersons DINOv3
person detection and the current AOS multi-view work. Historical experiments
remain indexed in `docs/experiment_registry/experiments.csv`; they should not
be mistaken for the canonical pipeline.

## Repository Map

| Path | Purpose | Status |
| --- | --- | --- |
| `src/forestpersons/` | Datasets, models, geometry, fusion, and metrics | Active library |
| `scripts/` | Training, evaluation, analysis, and preparation entry points | Active plus preserved experiments |
| `tests/` | Unit/regression tests organized by source component | Active |
| `data/` | Local ForestPersons, AOS, SARD, and Weitefeld data | Git-ignored; README tracked |
| `models/` | Local pretrained weights | Git-ignored; README tracked |
| `outputs/` | Local checkpoints, predictions, metrics, figures, and manifests | Git-ignored; summarized in docs |
| `experiments/` | External implementations and CUDA/Colab preparation | Historical/specialized |
| `docs/reports/` | Technical conclusions for individual experiments | Active record |
| `docs/experiment_registry/` | Experiment status and artifact index | Result provenance |
| `runs/detect/` | Ultralytics smoke-run output | Legacy; preserve |

The flat `scripts/` layout is deliberate backward compatibility. Reports,
commands, and outputs refer to these paths, so grouping scripts physically now
would break reproducibility for little practical gain.

Run entry points from the repository root with
`PYTHONPATH="$PWD/src:$PWD"`. The root component is required by the learnable
fusion trainer because it reuses the canonical FCOS loss implementation from
`scripts/train_dinov3_lora_fcos.py`.

## Canonical Execution Flow

1. `ForestPersonsDataset` reads COCO JSON, indexes annotations by `image_id`,
   preserves visibility/pose metadata, and returns a PIL RGB image plus target.
2. `resize_to_patch_grid` scales the short side and rounds both dimensions to
   multiples of 16. `image_to_tensor` applies the checkpoint processor's mean
   and standard deviation.
3. Hugging Face DINOv3 receives `[1, 3, H, W]`. Final patch tokens become
   `[H/16, W/16, C]` (`C=384` for ViT-S/16) and are L2-normalized.
4. `LoRALinear` wraps frozen attention `q_proj` and `v_proj` layers. The base
   matrices stay frozen; rank-4 A/B residual matrices are trainable.
5. `DINOv3FCOSHead` converts `[Gh, Gw, 384]` into class logits `[Gh, Gw]`, box
   distances `[Gh, Gw, 4]`, and centerness logits `[Gh, Gw]`.
6. `build_fcos_targets` assigns patch centers inside each GT center region,
   resolves overlap in favor of the smaller box, and encodes L/T/R/B in stride
   units. Training combines focal classification, Smooth L1 regression, and
   centerness BCE; experimental terms remain configuration-gated.
7. `decode_fcos_predictions` computes `sqrt(class * centerness)`, filters and
   top-k ranks candidates, decodes L/T/R/B around patch centers, applies NMS,
   then divides x and y by their independent resize scales.
8. Evaluation creates typed prediction records, applies one-to-one IoU
   matching, and reports COCO AP/AR plus fixed-threshold recall, FP/image, and
   visibility-specific recall.

## MVR And ANN2SNN Flow

`DINOv3MVRReadout` consumes frozen blocks 3/6/9/12, projects each layer, adds
native/fine/coarse context, and learns a residual over the final-layer feature.
`DINOv3MVRFCOSHead` feeds that normalized readout into the canonical FCOS head.

The hybrid conversion in `dinov3_ann2snn.py` calibrates six nonnegative MVR
ReLU sites from train features and replaces them with deterministic soft-reset
integrate-and-fire rate coding. Signed DINOv3 features, residual normalization,
the GroupNorm/GELU FCOS tower, and box decoding stay analog. `T=4/8/16` changes
rate approximation fidelity; spike activity is not a hardware energy result.

## Multi-View Flow

Each view is encoded by the same shared DINOv3+LoRA model.
`resized_homography` converts original-pixel homographies to resized image
coordinates. `warp_patch_features` inverse-warps source patch features onto the
reference grid with `grid_sample` and returns a validity mask.

- `fuse_aligned_features`: equal average of valid views.
- `fuse_selective_aligned_features`: non-trainable local cosine gate with a
  reference-preserving fallback.
- `TrainableReferencePreservingFusion`: lightweight learned gate; encoder and
  detector can remain frozen.

The unchanged FCOS head predicts in the reference image coordinate system.
Current AOS work verified camera poses and image overlap, but exact official
virtual-camera projection is blocked by missing F8 terrain/focal-surface
configuration; see `docs/reports/aos_geometry_feasibility_f8_2026-10-19.md`.

## Study Order

1. `src/forestpersons/dataset.py`
   - `validate_coco_annotations`, `ForestPersonsDataset`.
   - Learn how image IDs, metadata, and file paths remain consistent.
2. `src/forestpersons/dinov3_detector.py`
   - `PreprocessInfo`, `resize_to_patch_grid`, `extract_patch_features`,
     `DINOv3FCOSHead`, `build_fcos_targets`, `decode_fcos_predictions`.
   - This is the central tensor and coordinate path.
3. `src/forestpersons/dinov3_lora.py`
   - `LoRALinear`, `apply_lora_adapters`, `extract_lora_patch_features`, and
     checkpoint save/load functions.
4. `scripts/train_dinov3_lora_fcos.py`
   - `OnlineForestPersonsDataset`, `compute_fcos_losses`, optimizer parameter
     groups, and the training/checkpoint loop.
5. `src/forestpersons/evaluation.py` and `src/forestpersons/detection.py`
   - COCO formatting, AP/AR, confidence sweeps, and one-to-one matching.
6. `src/forestpersons/multiview.py`
   - `resized_homography`, `warp_patch_features`, and three fusion strategies.
7. `scripts/evaluate_dinov3_multiview_benchmark.py`
   - Shared encoding, warping, fusion, FCOS decoding, and paired analysis.
8. `scripts/train_dinov3_learnable_multiview_fusion.py`
   - Feature caching, fusion-only optimization, checkpointing, and validation.
9. `scripts/validate_aos_f8_geometry.py`
   - Current AOS metadata and three-view geometry feasibility check.
10. `src/forestpersons/dinov3_ann2snn.py`
    - IF rate coding, train-feature threshold calibration, spike activity, and
      the explicit analog/spiking conversion boundary.
11. `scripts/evaluate_dinov3_mvr_ann2snn.py`
    - ANN/T4/T8/T16 comparison using cached features and unchanged FCOS decode.

Supporting entry points:

- Train LoRA+FCOS: `scripts/train_dinov3_lora_fcos.py`
- Evaluate models: `scripts/evaluate_forestpersons_framework.py`
- Prepare image-only view groups: `scripts/prepare_forestpersons_multiview_benchmark.py`
- Evaluate equal fusion: `scripts/evaluate_dinov3_multiview_benchmark.py`
- Evaluate selective fusion: `scripts/evaluate_dinov3_selective_multiview_fusion.py`
- Visualize annotations: `scripts/visualize_forestpersons.py`
- Audit AOS Stage 1: `scripts/audit_aos_stage1.py`

## Five-Day Study Plan

### Day 1: Data and coordinates (45-60 min)

Read `dataset.py` and preprocessing in `dinov3_detector.py`. Trace one COCO
`xywh` annotation into original `xyxy`, resized pixels, and patch-grid units.

### Day 2: Encoder and LoRA (45-60 min)

Read `dinov3_lora.py`. Write down token shape, patch stride, LoRA equation,
target modules, frozen parameters, and checkpoint contents.

### Day 3: FCOS training and inference (60 min)

Read the FCOS head, target assignment, loss computation, and decoder. Explain
centerness, L/T/R/B distances, score ranking, top-k, and NMS without the code.

### Day 4: Evaluation (45-60 min)

Read `detection.py` and `evaluation.py`. Contrast AP, AP50, AP75, AR@100,
fixed-threshold recall, maximum recall, and recall under an FP/image budget.

### Day 5: MVR, ANN2SNN, and multi-view (60 min)

Read the MVR readout, `dinov3_ann2snn.py`, then `multiview.py`. Explain why only
MVR ReLUs were converted, what increasing timesteps changes, and why spike
activity is not energy. Then trace a source patch through homography resizing,
inverse sampling, gating, reference fallback, and FCOS prediction.

## Questions To Answer For Davide

1. Why are DINOv3 patch features useful under occlusion, and why are they not
   themselves detections?
2. Which DINOv3 parameters does LoRA change, and why do detached feature caches
   not work for LoRA training?
3. How are FCOS positives, L/T/R/B targets, centerness, and final scores built?
4. Why must x and y resize scales be handled independently?
5. What is the difference between COCO AR@100 and recall at a chosen threshold
   or FP/image budget?
6. Why did global feature averaging hurt, and how does reference-preserving
   selective fusion limit that failure?
7. Why does image-plane homography not guarantee alignment of people above the
   ground or moving between views?
8. Which AOS files are still required for faithful virtual-camera projection?
9. Which results are pilot, cross-dataset, or non-deployable oracle analyses?
10. What evidence would prove multi-view fusion adds information over a strong
    single-view detector?
11. Why is the current ANN2SNN result hybrid, and which signed/normalized layers
    block a faithful end-to-end conversion without additional methodology?

