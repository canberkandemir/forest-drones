# Data Directory

This directory holds local datasets used by the LogXY research project.
Dataset files are intentionally excluded from Git; only directory
documentation is published.

The official work plan is organized around ForestPersons first. Weitefeld, AOS,
WiSARD, and other datasets should be added only after the ForestPersons stage is
complete.

## Contents

- `annotations/`: canonical official ForestPersons COCO-style annotation JSON
  files:
  - `train.json`
  - `val.json`
  - `test.json`
- `images/`: approved local ForestPersons image subsets used for pilot
  experiments.

GitHub clones contain no annotations or images. Obtain datasets directly from
their official sources under the applicable access and licensing terms.

## Canonical Use

Use `data/annotations/` as the source of truth for official train, validation,
and test splits. Use subset manifests under `outputs/subsets/` to determine
which locally downloaded images belong to `train5000`, `val600`, smoke tests, or
other selected subsets.

For the experiment-level map, see:

- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

## Preservation Rules

Do not rename, move, delete, or rewrite files in this directory as part of normal
experimentation. Keep dataset access, subset construction, and image downloads
reproducible through scripts and manifests.
