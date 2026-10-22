# DTSS Real-Image Pilot: Label Feasibility Gate

Date: 2026-10-20

## Decision

The requested person-present versus person-absent pilot was **not run**. The
locally available ForestPersons data does not contain a verified negative
class, so constructing binary labels would not be scientifically valid.

## Verified evidence

- The official train, validation, and test COCO files contain 67,686, 18,243,
  and 10,553 image records, respectively.
- Thirteen train images and seven test images have no bounding-box annotation;
  validation has none. Their image records contain only `id`, `file_name`,
  `width`, and `height`, with no exhaustive-annotation or person-absence flag.
  They therefore cannot safely be relabeled as negatives.
- All 13 annotation categories have the `person` supercategory; category 13 is
  `unknown_0`, not a background class.
- Appendix C.5 of the [ForestPersons paper](https://arxiv.org/html/2603.02541v1)
  states that the primary dataset was designed around images containing a
  missing person. It separately identifies 193 verified no-human images in
  `379_FPV_No_Person_summer_forest`.
- A local filesystem check found zero files and no directory matching
  `379_FPV_No_Person_summer_forest` under `data/`.

## Consequence

No image patch embedding, classifier, training run, checkpoint, or performance
metric was created. This follows the experiment's predeclared stop condition
and avoids treating incomplete annotations as evidence of absence.

## Smallest valid continuation

After separately approving access to the official 193-image no-person set,
verify its provenance and checksum, then construct a deterministic,
image-disjoint balanced pilot. Because all negatives come from one sequence,
results must also be labeled vulnerable to scene/background shortcuts; a
sequence-independent negative set would be preferable for a defensible model
comparison.
