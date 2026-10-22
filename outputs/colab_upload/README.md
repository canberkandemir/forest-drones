# Colab Upload Bundles

This directory contains prepared Google Colab upload bundles for the published
ForestPersons detector baseline workflow.

These bundles belong to Phase 2 of the official work plan: published and
reproduced baselines, specifically DETR R50 and Deformable Faster R-CNN
R50-FPN/DCN using the CUDA/MMDetection workflow.

For the canonical experiment status, see:

- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

## Contents

- `LogXY/`: prepared minimal Colab bundle.
- `LogXY 2/`: duplicate-looking prepared bundle from a previous sync attempt.
- `LogXY_colab_smoke30.zip`: zipped upload bundle when present.

## Status

The Colab workflow has been prepared locally, but DETR and Deformable Faster
R-CNN GPU inference has not been verified successfully yet. Do not treat these
bundles as completed baseline results.

The bundles should contain only files required for Colab smoke testing, such as
the notebook, setup scripts, exported COCO subsets, and the 30-image smoke
subset. They should not contain the full ForestPersons dataset, local virtual
environments, DINOv3 feature caches, or old training outputs.

## Preservation Rules

Do not overwrite these bundles unless the Colab workflow is being intentionally
refreshed. When preparing a new bundle, prefer a dated or clearly versioned
folder name so old setup attempts remain inspectable.
