# Experiment Registry

This directory is the navigation index for LogXY ForestPersons experiments.

Use `experiments.csv` to identify:

- which phase of Davide's work plan an experiment belongs to;
- whether the experiment completed, was only prepared, or was only a feasibility
  assessment;
- the main entry-point script;
- the dataset subset used;
- the relevant checkpoints or outputs;
- the metrics that were actually produced.

The registry is intentionally descriptive. It does not move or rewrite any
experiment artifacts. Existing paths remain valid so older commands, reports,
and tests stay reproducible.

Phase-level consolidation reports live under `docs/reports/`. The current
ForestPersons Phase 1-2 report is
`docs/reports/forestpersons_phase1_phase2_2026-10-12.md`.

Status labels:

- `completed`: experiment was executed and produced usable outputs.
- `prepared`: code/configuration exists, but the experiment has not successfully
  run yet.
- `feasibility`: methodology/dependency analysis exists, but no model output was
  produced.
- `not_started`: no implemented experiment was found.
- `legacy`: exploratory output that should be preserved but not treated as a
  current result.
- `interrupted`: run appears incomplete or superseded by a later rerun.
