# LogXY Research Roadmap - 2026-10-22 to 2026-12-20

## Planning principles

- Prioritize a small number of controlled questions over additional heads.
- Treat val600 as development data and establish a new sequence-disjoint final
  evaluation set before the last comparison.
- Reuse cached DINOv3 features when the backbone is frozen.
- Never compare paper full-split values directly with train5000/val600 pilots.
- Do not claim energy savings without device-level latency and power evidence.
- Stop a line of work when its predeclared success gate fails.

## Priority order

1. Explain the existing MVR gain with a readout ablation.
2. Measure label efficiency with nested training subsets.
3. Test whether LoRA adds value to the best MVR readout.
4. Optimize and characterize the hybrid ANN2SNN boundary.
5. Continue AOS geometry or RGB/thermal work only when its validity gate is met.
6. Perform final independent evaluation and deployment feasibility.

## Schedule

### Oct 22-27: MVR readout ablation

- Add one native-only multi-layer MVR mode without changing existing modes.
- Reuse train5000/val600 caches and existing baseline/full-MVR checkpoints.
- Train only the missing native multi-layer arm for 20 epochs.
- Compare final-layer baseline, multi-layer native readout, and full
  multi-layer/multi-resolution MVR.
- Gate: proceed only if the new arm clarifies a reproducible AP/AR/R20 effect.

Expected local cost: approximately 1-2 hours for the new cached-feature arm,
plus less than 20 minutes for tests and evaluation. The observed endpoints are
46 minutes for baseline and 136 minutes for full MVR.

### Oct 28-Nov 8: Label-efficiency study

- Freeze the winning readout design from the ablation.
- Create nested, sequence-aware train100/train400/train1000/train5000 manifests
  from train5000; retain visibility balance and image-ID-safe lookup.
- Run three seeds if the first seed shows a meaningful trend.
- Use identical initialization, 20 epochs per subset, and fixed evaluation.
- Compare against matched YOLO11s subset training only if Davide confirms the
  extra compute is worth the direct label-efficiency claim.

Expected cached-head cost per seed is roughly 3, 11, 27, and 136 minutes for
100/400/1000/5000 MVR examples, based on linear scaling from the measured
train5000 run. Three complete MVR seeds are approximately 9 hours total.

### Nov 9-18: Frozen vs LoRA MVR

- Initialize both arms from the same selected MVR checkpoint.
- Control: continue readout/FCOS with the encoder frozen.
- Experimental: add fresh rank-4 q/v LoRA and train online with the same image
  order, two epochs, optimizer steps, and loss definitions.
- Use batch 1, accumulation 4, and strict MPS memory monitoring.

Expected local cost: about 3.3 hours per online train5000 two-epoch arm, or
6-8 hours sequentially. Peak driver memory is expected near the prior 10.6 GiB
ViT-S LoRA observation. Stop on non-finite loss or sustained memory pressure.

### Nov 19-28: ANN2SNN optimization

- Use the winning frozen MVR checkpoint and unchanged train-only calibration.
- Measure per-site activation approximation error for T4/T8/T16.
- Test conversion boundaries separately rather than converting unsupported
  signed/normalized layers silently.
- Compare dense MPS and CPU latency/memory; report spike activity only as an
  activity proxy.
- Evaluate whether Spikify is useful for input encoding. It is an encoder
  library, not evidence of faithful detector conversion or energy reduction.

Gate: retain ANN2SNN only if T8 or lower preserves the predeclared accuracy
tolerance, or if actual target hardware becomes available for a power study.

### Nov 29-Dec 7: AOS geometry and RGB/thermal decision

- In parallel from Oct 22, request the missing AOS DEM/OBJ, COLMAP-frame
  transform, exact virtual camera, and focal-surface parameters from the
  authors.
- If supplied, reproduce one training-scene integral projection before any
  detector comparison.
- If not supplied, do not invent geometry. Run only the already prepared
  thermal-weighted integral-feature comparison (thermal, alpha 0.75, alpha
  0.90), selected on O6 validation and evaluated once on canonical test.
- Treat the 37/5 unique train/validation people as a feasibility limitation.

### Dec 8-13: Cross-domain and final split preparation

- Freeze model and operating points.
- Construct a new sequence-disjoint ForestPersons validation/test protocol from
  locally complete images, excluding every development image and calibration
  image. If this is impossible, document the dependence and use the official
  validation split without claiming scene independence.
- Resolve SARD license and annotation-exhaustiveness questions before expanding
  its pilot.
- Keep Weitefeld excluded from person-recall claims unless annotations are
  independently adjudicated.

### Dec 14-20: Final evaluation and deployment feasibility

- Evaluate only the selected YOLO, DINOv3 baseline, MVR, optional LoRA-MVR,
  and optional hybrid ANN2SNN modes.
- Report AP/AP50/AP75/AR@100, visibility recall, matched FP/image, localization,
  parameter count, memory, and latency.
- Perform quantization/export feasibility only for the selected model. Without
  a Jetson-class target, report Mac measurements and an explicit hardware test
  plan rather than projected energy savings.
- Consolidate negative results and prepare the final technical narrative.

## Work that can run in parallel

| Parallel task | Can start | Dependency |
|---|---|---|
| Ask AOS authors for missing geometry | Immediately | Davide approval/contact route |
| Confirm licensing for ForestPersons, DEIMv2, SARD | Immediately | Supervisor/company review |
| Design untouched sequence-disjoint evaluation | During MVR ablation | Local image availability |
| Prepare nested label manifests | During MVR training | Frozen selection rules |
| Identify edge target and measurement protocol | During label study | Davide hardware decision |

Do not run multiple MPS training jobs concurrently on the 16 GB Mac.

## GPU-dependent work

- Full-resolution ViT-L web-vs-satellite training.
- Faithful DETR and Deformable R-CNN reproduction with legacy CUDA operators.
- Full official ForestPersons online-backbone training or multi-seed LoRA at
  scale.

These are not on the critical path unless Davide explicitly prioritizes them.

## Optional extensions

- DINO-WM/temporal prediction only after obtaining reliable track identities,
  timestamps, and sequence-disjoint evaluation.
- WiSARD/RGBTDronePerson only after access, license, paired calibration, and
  confirmed exhaustive person labels are verified.
- DEIMv2-S train400 after non-commercial internship use is approved.
- Full DTSS training only after a bounded pilot demonstrates masks crossing a
  timestep boundary and genuine skipped computation is implemented.

## Final decision gates

1. MVR must show a component-level benefit, not just a combined-head gain.
2. Label efficiency must be reported across seeds with nested subsets.
3. LoRA must beat a matched continuation, not an older differently initialized
   frozen model.
4. ANN2SNN must preserve accuracy and be measured on appropriate hardware
   before any efficiency claim.
5. Multi-view/thermal claims require geometry and labels tied to the same
   physical people and scenes.
