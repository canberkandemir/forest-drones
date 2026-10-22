# Cleanup Candidates Awaiting Approval

No files listed here have been deleted.

## Demonstrably Regenerable

- All repository `.DS_Store` files (Finder metadata).
- `.pytest_cache/` (pytest cache).
- Repository `__pycache__/` directories and `*.pyc` files outside virtual
  environments (Python bytecode). This includes one stale
  `dinov3_detr.cpython-313.pyc`; its unfinished DETR source was already removed.

These items contain no datasets, source code, checkpoints, metrics, or reports.

## Preserve Or Review Manually

- `runs/detect/` (37 MiB): legacy YOLO smoke checkpoint/results referenced by
  the experiment registry; preserve unless its reproducibility value is no
  longer needed.
- Root PDFs (`Internship_DINOv3_DIC.pdf`,
  `Internship_DINOv3_People_Under_Vegetation.pdf`, `logxy1.pdf`): hashes differ,
  so none is a proven duplicate. They could later move under `docs/specs/`, but
  only with link/path updates.
- Interrupted/superseded outputs: preserve until each is checked against the
  experiment registry and reports; names alone are not deletion evidence.

