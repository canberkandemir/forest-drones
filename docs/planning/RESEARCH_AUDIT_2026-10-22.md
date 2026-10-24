# LogXY Research Audit - 2026-10-22

## Scope and evidence

This audit covers the internship project "Finding People Under Dense Vegetation
with DINOv3" through 2026-10-22. It is a planning audit, not a new experiment.
No model was trained, no inference was run, and no dataset or checkpoint was
changed.

Evidence was ranked in this order:

1. Saved metrics, checkpoints, manifests, and prediction artifacts indexed by
   the experiment registry.
2. Completed experiment reports and the chronological `PROGRESS.md` record.
3. Implemented scripts and tests.
4. Prepared-only code, which is never treated as a completed experiment.

The original one-page internship specification defines five broad work areas:
data, published/anomaly baselines, DINOv3 heads and adaptation, multi-view and
thermal fusion, and edge feasibility. ForestPersons remains the primary
under-canopy benchmark. All `train5000`/`val600` numbers below are selected
subset pilot results, not official full-split benchmark results.

The older 2026-10-12 repository audit is historically useful but no longer
describes the current implementation status. LoRA, MVR, ANN2SNN, cross-dataset,
and fusion work was completed after that audit.

## Dataset state

### ForestPersons

- Official annotations: 67,686 train images / 145,816 annotations, 18,243
  validation images / 37,395 annotations, and 10,553 test images / 20,867
  annotations.
- Official categories encode pose and visibility; the project maps them to one
  `person` class while retaining visibility metadata.
- Verified pilot training set: 5,000 images, 6,431 annotations in the original
  YOLO conversion summary. The later image-ID-safe audit counted 6,415 usable
  visibility-labelled annotations for current cached-feature training.
- Verified pilot validation set: 600 images, 749 evaluated people, with
  visibility 20/40/70/100 counts of 174/170/193/212. One `unknown_0`
  annotation is excluded from visibility evaluation.
- Train5000 and val600 have zero exact-image overlap. Duplicate filenames with
  distinct image IDs were found historically; current maintained loaders use
  `image_id` and verify filenames.
- The last full-download verification recorded 96,292 unique official image
  paths with 901 local files missing. This does not affect train5000 or val600,
  which are complete, but it blocks a claim that the full local dataset is
  complete.

### Other local datasets

- AOS Stage 1 is available: 6,271 raw low-resolution RGB/thermal frame pairs,
  poses/calibration, and 9,612 paired RGB/thermal integral products per
  modality. Canonical scene labels contain 37 train, 5 validation, and 57 test
  people. Augmentations repeat those underlying people.
- SARD has a 200-frame, 305-person mountain pilot. Frames are sequence
  correlated, visibility labels are absent, annotation exhaustiveness and the
  concrete dataset license remain unresolved.
- Weitefeld has only the audited 200-image subset. Its 155 class-3 regions are
  not confirmed-person ground truth: 134 have explicit non-person/no-abnormality
  evidence and 21 remain uncertain.
- No verified local WiSARD, RGBTDronePerson, NOMAD, or HERIDAL benchmark is
  ready for a controlled experiment.

## Experiment status by research direction

| Direction | Actual status | Best verified evidence | Main limitation |
|---|---|---|---|
| Dataset and evaluation | Completed for pilot scope | COCOeval-verified evaluator; train5000/val600 manifests and image-ID-safe loading | No official full-split result; val600 is repeatedly inspected |
| YOLO11 baseline | Completed | Fine-tuned YOLO11s: AP 0.5541, AR@100 0.6339, recall 0.9039, R20 0.7701 at 0.3467 FP/image | 47-epoch train5000 pilot, not paper reproduction |
| DETR / Deformable R-CNN | Prepared only | CUDA/MMDetection configs and Colab packaging | No successful detector inference or ForestPersons training; CUDA-dependent legacy stack |
| DEIMv2-S | Zero-shot completed; fine-tuning prepared only | AP 0.3408, AP75 0.3544, AR@100 0.6016; local MPS gradient smoke passed | No ForestPersons fine-tuning; research-only license needs supervisor review |
| RX anomaly | Completed | Patch AUROC 0.5787, AP 0.0314 | Weak RGB-statistics baseline; not box detection |
| EfficientAD | Blocked feasibility | No executed model | ForestPersons does not provide reliable normal-only training imagery |
| DINOv3 patch-kNN | Completed | Patch AUROC 0.8037, AP 0.1338 | Patch anomaly metrics are not detector AP/AR |
| Frozen DINOv3 FCOS | Completed | AP 0.3113, AR@100 0.4368, R20 0.5632 at the reported operating point | Selected subset; weak localization and score-quality mismatch |
| DINOv3 LoRA FCOS | Completed | AP 0.3587, AP75 0.2379, AR@100 0.4704; recall 0.8465 and R20 0.6667 near 0.425 FP/image | Initialized from a trained FCOS head, so not scratch-matched to frozen FCOS |
| Alternative DINOv3 heads | Completed pilots | Linear, FPN, stride-8, YOLO/DFL, quality, GIoU, and RoI refinement results preserved | None clearly surpassed the original LoRA stride-16 FCOS recall; do not repeat without new evidence |
| Web vs satellite DINOv3 | Reduced pilot completed | Matched ViT-L/16 train400 at short side 336; web AP 0.1409 vs satellite AP 0.0656 | Short side 672 caused MPS OOM; reduced pilot cannot isolate domain pretraining conclusively |
| Frozen MVR + FCOS | Completed | Baseline AP/AR 0.3267/0.4550 vs MVR 0.3623/0.4900; R20 0.4425 vs 0.4885 at 0.15 FP/image | One seed, val600 development set, and combined multi-layer/multi-resolution change |
| Hybrid ANN2SNN MVR | Completed | T16 retained 99.0% of ANN AP50 and matched ANN R20 | Hybrid only; 3.77x slower on dense MPS; spike activity is not measured energy |
| ForestPersons multi-view | Completed and stopped | Equal averaging failed; selective/learned gates preserved single-view but added no robust new detections | Homographies align background, not necessarily people; no site/person identity |
| AOS geometry-aware multi-view | Feasibility blocked | Poses reproduce COLMAP transforms; RGB overlaps register in image space | Missing DEM/OBJ, frame transform, exact virtual camera and focal-surface configuration |
| AOS RGB/thermal detection | Small experiment completed | Thermal 26/57 TP at 0.2 FP/image; RGB 0/57; mean fusion 5/57 at validation-locked thresholds | Only 37/5 underlying train/validation people; mean fusion suppresses thermal confidence |
| Thermal-weighted AOS fusion | Prepared only | Alpha 0.75/0.90 workflow and tests | Not executed; scientific power remains small |
| Cross-dataset generalization | Completed pilots | SARD DINO recall 0.1311 vs YOLO 0.0787; Weitefeld deemed invalid for person recall | Severe domain/scale shift and dataset annotation limitations |
| DTSS Spiking Transformer | Tiny pilot completed; full run prepared | Forward/backward valid; learned masks stayed 4/4 active | Pilot detector unusable; masking did not skip computation or reduce timesteps |
| Edge deployment | Not started | MPS latency/memory observations only | No Jetson-class device, quantization study, power measurement, or deployment runtime |

## Controlled MVR and ANN2SNN findings

The strongest newly completed controlled experiment trains fresh matched FCOS
heads for 20 epochs on the same train5000 cached features:

| Readout | AP | AP50 | AP75 | AR@100 | Recall at 0.15 FP/image | R20 at 0.15 FP/image |
|---|---:|---:|---:|---:|---:|---:|
| Final DINOv3 layer | 0.3267 | 0.7915 | 0.2022 | 0.4550 | 0.6916 | 0.4425 |
| Current MVR | 0.3623 | 0.8162 | 0.2675 | 0.4900 | 0.7210 | 0.4885 |

The current detector is inspired by DINO-MVR, not a faithful reproduction of
the medical segmentation paper. The paper reads the final three DINOv3 blocks
with scale-specific lightweight probes and performs multi-resolution/TTA
probability fusion. This repository uses blocks 3/6/9/12, a convolutional
native/fine/coarse residual readout, and an FCOS detector. The gain is real for
this controlled pilot, but the responsible component is not yet isolated.

The full val600 hybrid conversion gives:

| Mode | AP50 | Recall | R20 | FP/image | ms/image |
|---|---:|---:|---:|---:|---:|
| ANN | 0.8162 | 0.8077 | 0.5862 | 0.3183 | 13.0 |
| SNN T=4 | 0.7622 | 0.7156 | 0.4425 | 0.2700 | 23.2 |
| SNN T=8 | 0.8002 | 0.7891 | 0.5632 | 0.2967 | 31.1 |
| SNN T=16 | 0.8083 | 0.7997 | 0.5862 | 0.2933 | 49.1 |

Only six nonnegative MVR ReLU sites are converted. Frozen signed DINOv3
features, residual normalization, the FCOS GroupNorm/GELU tower, score heads,
box regression, decoding, and NMS remain analog.

## Scientific gaps

### A. Does MVR improve visibility-20/40 detection?

Partly answered on val600. MVR improves R20 at a matched exploratory FP budget,
but only one seed and a repeatedly inspected validation subset exist. R40 also
needs a standard matched-budget table in the ablation. Independent confirmation
is missing.

### B. Does MVR outperform standard DINOv3 under controlled training?

Yes for the current combined readout on train5000/val600. AP, AP50, AP75,
AR@100, recall, and R20 all improved. What remains unknown is whether the gain
comes from intermediate layers, multi-resolution context, parameter count, or
their interaction.

### C. How much labeled data does DINOv3 need relative to YOLO?

Unanswered. Existing train400 and train5000 runs have different initialization,
epoch histories, and sometimes architectures. They are not a label-efficiency
curve. A nested 100/400/1000/5000 study is required for both methods.

### D. Does LoRA improve MVR enough to justify cost?

Unanswered. LoRA improves the older FCOS pipeline, but no matched frozen-vs-LoRA
MVR experiment exists. Cached detached features cannot train LoRA, so this
requires online DINOv3 forward/backward passes.

### E. Can ANN2SNN preserve accuracy while reducing cost?

Accuracy preservation is partly answered: T16 is close to ANN. Cost reduction
is not demonstrated. Dense MPS simulation is slower, masks do not establish
event-driven execution, and no neuromorphic or power measurement exists.

### F. Can aligned multi-view or RGB/thermal data rescue misses?

Not established. ForestPersons homography fusion did not add robust detections.
AOS mean RGB/thermal fusion was much worse than thermal alone. Exact AOS raw
multi-view geometry remains blocked by missing release parameters.

### G. How well do models generalize across canopy regimes?

Poorly, based on SARD and AOS zero-shot pilots, but current pilots are small and
dataset-limited. Weitefeld cannot support confirmed-person recall. No rigorous
under-canopy to above-canopy scene-disjoint benchmark is complete.

### H. Which methods are realistic for edge deployment?

Unanswered. YOLO and DEIMv2 have useful local latency observations; MVR and
hybrid ANN2SNN have software latency only. Model size, quantization accuracy,
Jetson latency, power, and memory must be measured after selecting a final
detector.

## Reproducibility and leakage risks

1. `val600` has been used repeatedly for architecture analysis and threshold
   sweeps. It is a development set, not a final unbiased test.
2. The independent1200 fusion holdout is exact-image and hash disjoint, but all
   50 sequence folders overlap protected manifests. It is not scene-disjoint.
3. Historical filename-based loading mishandled duplicate names. Current
   image-ID-safe loading is required for every future arm.
4. LoRA-FCOS, frozen FCOS, and MVR results have different initialization and
   training histories unless explicitly described as matched.
5. AOS augmentations greatly outnumber unique people. Statistical units should
   be people/scenes, not augmented images.
6. Patch anomaly AUROC/AP must not be compared directly with box AP/AR.
7. Paper baselines use the full official split. Internal train5000/val600
   results cannot claim paper-benchmark superiority.
8. The registry still contains historical prepared rows alongside later
   completed rows. The latest dated completed row/report takes precedence.

## Resource feasibility

| Work | Mac M4 16 GB | CUDA / special hardware |
|---|---|---|
| Cached-feature FCOS/MVR heads | Feasible; baseline 20 epochs took 46 min, full MVR 136 min | Not required |
| ViT-S LoRA online training | Feasible but near memory limit; 2 train5000 epochs took 3.31 h and about 10.6 GiB driver memory | Helpful for repetitions |
| ViT-L short side 672 | Not feasible locally; observed MPS OOM near 19.36 GiB | CUDA GPU recommended |
| DEIMv2-S inference/backward | Feasible with documented CPU fallback | CUDA helpful for larger runs |
| DETR/Deformable R-CNN legacy reproduction | Not reliable on current MPS stack | CUDA environment required |
| ANN2SNN software simulation | Feasible but slower than ANN | Neuromorphic/edge hardware required for energy claims |
| Full official ForestPersons training | Possible only for selected lightweight cached heads locally; online backbones are long | CUDA strongly preferred |

## Audit conclusion

The project has moved beyond feasibility: it has a verified evaluation stack,
strong YOLO baseline, multiple DINOv3 detectors, controlled MVR gain, hybrid
ANN2SNN conversion, anomaly baselines, and negative fusion/cross-domain results.
The largest scientific weakness is no longer lack of implementations. It is the
absence of a compact ablation and independent evaluation design that explains
why MVR helps and how label-efficient that help is.

The highest-value next action is therefore the MVR readout ablation defined in
`NEXT_EXPERIMENTS_2026-10-22.md`, not another detection architecture.

## Primary references

- Davide Baroli, `Finding People Under Dense Vegetation with DINOv3`, local
  internship specification, 2026-09-23.
- Jiang et al., [DINO-MVR: Multi-View Readout of Frozen DINOv3 for
  Annotation-Efficient Medical Segmentation](https://arxiv.org/abs/2605.07221),
  2026. This is a medical segmentation method, not the detector implemented
  here.
- Kim et al., ForestPersons, arXiv:2603.02541 / ICLR 2026, as consolidated in
  `docs/reports/forestpersons_phase1_phase2_2026-10-12.md`.
- [neuromorphic-polito/spikify](https://github.com/neuromorphic-polito/spikify),
  an input signal-to-spike encoding library; it is not an end-to-end detector
  ANN-to-SNN conversion result.
