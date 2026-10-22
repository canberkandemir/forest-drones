# DINOv3 Detector Outputs

This directory contains frozen-DINOv3 feature caches, detector-head checkpoints,
evaluation sweeps, plots, and analysis outputs for ForestPersons pilot
experiments.

The current ForestPersons research priority is DINOv3 development, especially
LoRA adaptation, lightweight detection heads, and recall under heavy occlusion.
Existing FCOS-style outputs should be preserved as prior pilot work, but not
developed further unless explicitly requested.

For the experiment-level status map, see:

- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

## Canonical Artifacts

- `features/`: cached frozen DINOv3 ViT-S/16 patch features for approved
  subsets. This is large and should not be moved casually.
- `head_v3_fcos_train5000_seed20261012/`: completed DINOv3 FCOS-style pilot
  trained on `train5000`.
- `head_linear_patch_train5000_seed20261013_rerun/`: completed simple linear
  patch-head experiment trained on `train5000`.
- `eval_val600_head_v3_fcos_train5000_*`: threshold sweeps for the train5000
  FCOS-style pilot.

## Historical Or Prototype Artifacts

- `head_v1/`: first frozen-DINOv3 detection head prototype.
- `head_v2_grid_focal/`: grid/focal prototype.
- `head_v3_fcos/`: FCOS-style prototype trained on the smaller `train400`
  subset.
- `eval_val600_head_v1_*`, `eval_val600_head_v1_recheck_*`,
  `eval_val600_head_v2_grid_focal_*`, and `eval_val600_head_v3_fcos_*`:
  historical sweeps that should be preserved for provenance.

## Interrupted Or Smoke-Only Artifacts

- `head_linear_patch_train5000_seed20261013/`: interrupted or superseded linear
  patch-head run. Use the `_rerun` directory for the completed result.
- `head_linear_patch_train5000_smoke/`: smoke-only training run.

## Preservation Rules

Do not delete cached features, checkpoints, threshold sweeps, plots, or
historical rechecks. New DINOv3 experiments should use new directories with
explicit names and should update the experiment registry.
