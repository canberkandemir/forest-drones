# AOS Frozen DINOv3 RGB-Thermal Correspondence: Implementation Note

Date: 2026-10-20  
Status: implemented and statically validated; experiment not executed

## Protocol

The script discovers exactly one canonical, no-AHE RGB/thermal integral-image
pair for each of the 18 official AOS scenes. It verifies split membership,
512 x 512 dimensions, matching stems, and byte-identical YOLO labels before
loading the model.

It loads the existing ForestPersons DINOv3 ViT-S/16 rank-4 LoRA checkpoint,
freezes every backbone parameter, and does not use the FCOS head. RGB images
use the checkpoint's normal preprocessing. Thermal integral images are
converted to grayscale and replicated across three channels before the same
preprocessing. Both modalities use short side 672, producing matching patch
grids.

Per scene, the experiment records:

- same-location patch cosine similarity;
- best cosine within a fixed 3 x 3 neighborhood and its displacement;
- global RGB-to-thermal patch retrieval accuracy at exact position and within
  one patch;
- person-box versus background-patch similarity using the released common-frame
  labels;
- a predeclared wrong-scene control using the next thermal scene in fixed
  split-then-scene order.

Visualizations are generated for F0, F8, and O6. They contain the labeled RGB
integral, thermal integral, same-location cosine map, local-best cosine map,
and local displacement map.

## Alignment and interpretation

The canonical integral products share official virtual coordinates, dimensions,
stems, and labels. No learned or image-fitted warp is applied. The local search
is reported separately to expose small residual rendering or modality-specific
misalignment; it does not redefine the primary same-location metric.

These measurements test frozen-feature correspondence only. They are not
detection AP/recall, do not establish that fusion helps, and must not be used to
select a fusion rule on the official test scenes.

## Command

From the repository root:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv-dinov3/bin/python \
  scripts/evaluate_aos_dinov3_rgb_thermal_correspondence.py \
  --integral-root "data/aos/integral_images/Integral Images/data" \
  --checkpoint outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt \
  --output-dir outputs/aos/dinov3_rgb_thermal_correspondence_20261020 \
  --device mps \
  --short-side 672 \
  --local-radius 1 \
  --visual-scenes F0,F8,O6
```

The output directory must be absent or empty. Expected artifacts are
`metrics.json`, `per_scene_metrics.csv`, and three visualization PNG files.

## Validation

Three synthetic tests passed for YOLO-box patch masks, local shifted
correspondence, and global spatial retrieval. Both implementation files passed
`py_compile`. No checkpoint was loaded and no AOS feature inference was run.
