# Project Guidelines

## Scope

This repository supports a LogXY research internship on DINOv3-based detection
of partially or heavily occluded people in forest imagery. Work incrementally.

The project now contains multiple pilot experiments beyond Phase 1. Before
adding new experiments or reorganizing files, read:

- `PROGRESS.md`
- `docs/audit/repository_audit_2026-10-12.md`
- `docs/experiment_registry/README.md`
- `docs/experiment_registry/experiments.csv`

Treat the experiment registry as the current path-level map of what has run,
what is prepared, and what remains unexecuted.

## Research Background

- The user has already completed a literature review before implementation.
- Treat DINOv3, DINOv2, DINO-WM, ForestPersons, Weitefeld, and the other papers
  referenced in Dr. Davide Baroli's internship PDF as already studied by the
  user.
- Do not restart theoretical research from scratch unless the user asks for it.
- Do not assume access to the user's personal study notes or the full paper
  contents unless they are provided in the session.
- Use Dr. Davide Baroli's internship PDF as the primary project specification,
  and use the relevant papers as technical references when implementation
  choices require them.
- Current priority is translating the research into working code, experiments,
  and evaluation, especially person-detection recall under different vegetation
  occlusion levels.

## Research Rules

- Do not fabricate images, annotations, metadata, metrics, or results.
- Do not train YOLO, DETR-family models, anomaly detectors, SAM, or DINOv3 unless
  the relevant phase is explicitly requested.
- Preserve official train, validation, and test splits.
- Preserve ForestPersons metadata fields such as visibility, pose, difficulty,
  and any additional fields present in the official annotations.
- Treat gated dataset terms seriously. Do not download the full dataset without
  explicit permission.
- Report blockers plainly, especially authentication, missing files, licensing,
  or unavailable dependencies.

## Coding Rules

- Keep `test_dinov3.py` working unless the user explicitly asks to refactor it.
- Prefer small, readable Python modules over notebook-only workflows.
- Avoid heavy dependencies for dataset loading; parse COCO JSON directly unless
  a later phase clearly needs `pycocotools`.
- Add tests for loader behavior using synthetic fixtures when real data is not
  available.
- Use reproducible commands and document anything that depends on local paths or
  credentials.

## Data Layout

Use this layout for local ForestPersons files after approved access:

```text
data/
  images/
  annotations/
    train.json
    val.json
    test.json
```

Large datasets and model weights should stay out of version control.

## Repository Layout

Keep the project organized as follows:

```text
docs/specs/              Internship specifications from LogXY.
docs/papers/             Research papers and dataset references.
assets/sample_images/    Small scratch/demo images used by exploration scripts.
models/                  Local pretrained weights such as YOLO11 checkpoints.
data/annotations/        Official ForestPersons COCO JSON annotations.
data/images/             Approved local ForestPersons image subsets.
src/forestpersons/       Reusable dataset/loading code.
scripts/                 Reproducible project scripts.
tests/                   Lightweight validation tests.
outputs/                 Generated figures, manifests, and evaluation results.
```

Root-level `test_dinov3.py` and `test_yolo.py` are exploratory scripts. Keep
them working, but put new reusable implementation in `src/` or `scripts/`.

## Experiment Status Discipline

- Clearly label selected-subset results as pilot results, not official
  ForestPersons benchmark scores.
- Distinguish completed runs from prepared, interrupted, smoke-only, or
  feasibility-only work.
- Preserve existing checkpoints, metrics, cached features, and output
  directories unless the user explicitly approves a migration.
- Do not move or delete large experiment artifacts as part of ordinary coding
  work. Prefer adding registry entries or README notes first.
- For published baselines, do not claim reproduction unless the architecture,
  weights, environment, and evaluation protocol match the documented setup.
