# Progress

## Completed

- Read the LogXY internship PDF and extracted the ForestPersons-specific scope.
- Inspected the existing project directory.
- Confirmed `test_dinov3.py` exists and left it unchanged.
- Verified that the directory is not currently a Git repository.
- Checked Python environments:
  - `python3` points to framework Python 3.13.7 and lacks Torch/Transformers.
  - `python` points to Anaconda Python 3.13.5 and has Torch 2.9.0, torchvision
    0.24.0, Pillow, matplotlib, NumPy, pandas, and pytest.
- Verified from Hugging Face documentation that `etri/ForestPersons` is gated,
  CC BY-NC-SA 4.0 for non-commercial research, and approximately 47.6 GB.
- Added a lightweight COCO-style ForestPersons dataset loader.
- Added a ground-truth visualization script that requires real local dataset
  files.
- Added synthetic tests for loader behavior and validation.
- Ran `PYTHONPATH=src python -m pytest tests -q`: 2 tests passed.
- Ran `PYTHONPATH=src python -m py_compile src/forestpersons/*.py scripts/visualize_forestpersons.py`.
- Inspected official ForestPersons annotation JSON files located at
  `data/annotations/train.json`, `data/annotations/val.json`, and
  `data/annotations/test.json`.
- Confirmed official COCO-style structure:
  - Top-level keys: `info`, `licenses`, `images`, `annotations`, `categories`.
  - Image records: `id`, `file_name`, `width`, `height`.
  - Annotation records: `id`, `image_id`, `category_id`, `bbox`, `iscrowd`,
    `area`, `attributes`.
  - ForestPersons metadata is nested in `attributes`: `camera`, `weather`,
    `season`, `place`, `pose`, `visibility_level`.
- Updated the loader to recognize categories via `supercategory=person` and to
  flatten official `attributes` into per-annotation `metadata` while preserving
  the original `attributes` dictionary.
- Updated the visualization script default root to `data` and labels to include
  `visibility_level`.
- Added real-annotation tests for the official train, validation, and test JSON
  files.
- Ran `PYTHONPATH=src python -m pytest tests -q`: 5 tests passed.
- Ran `PYTHONPATH=src python -m py_compile src/forestpersons/*.py scripts/visualize_forestpersons.py tests/*.py`.
- Recorded the user's clarification that the literature review is already
  complete. The user has studied DINOv3, DINOv2, DINO-WM, ForestPersons,
  Weitefeld, and the other relevant papers/datasets from the internship PDF;
  future work should focus on implementation and evaluation rather than
  restarting theoretical research.
- Updated `scripts/visualize_forestpersons.py` so it samples only locally
  available image files by default, which supports small approved image subsets.
- Inspected the official Hugging Face image directory structure via metadata:
  `images/<sequence_folder>/<sequence_folder>-####.jpg`.
- Selected and downloaded an approved 32-image validation subset into
  `data/images/`, preserving official relative paths.
- The selected subset targets 8 validation images each for visibility levels 20,
  40, 70, and 100.
- Downloaded/verified image bytes: 15,364,484 bytes, approximately 14.65 MiB.
- Saved the subset manifest to
  `outputs/visualizations/forestpersons_val_subset_manifest.json`.
- Verified all 32 downloaded files with PIL image loading, expected byte sizes,
  RGB mode, image dimensions from `val.json`, and annotation matching.
- Verified the local subset contains 48 matched validation annotations, with
  annotation visibility counts: 20 = 10, 40 = 11, 70 = 11, 100 = 16.
- Ran the real-image visualization script on 10 locally available validation
  images and saved the result to
  `outputs/visualizations/forestpersons_val_gt_preview.png`.
- Ran `PYTHONPATH=src python -m pytest tests -q`: 5 tests passed.
- Created isolated YOLO environment `.venv-yolo` with `--system-site-packages`
  to reuse existing Torch, then installed `ultralytics` and `opencv-python`.
- Verified YOLO environment imports: Torch 2.9.0, torchvision 0.24.0,
  Ultralytics 8.4.175, OpenCV 5.0.0, NumPy 2.5.3, Pillow 11.1.0, matplotlib
  3.10.0.
- Added `scripts/evaluate_yolo11_forestpersons.py`, a reproducible pretrained
  YOLO11 evaluator for the local ForestPersons subset.
- Ran pretrained YOLO11n smoke test on the 32 downloaded validation images with
  confidence threshold 0.25, IoU matching threshold 0.50, image size 960, and CPU
  inference.
- Saved YOLO11n results under
  `outputs/yolo11_baseline/yolo11n_conf025_iou050_img960/`.
- YOLO11n smoke-test metrics:
  - Overall: 48 GT, 18 TP, 4 FP, 30 FN, recall 0.3750, precision 0.8182.
  - Visibility 20: 10 GT, 0 TP, 2 FP, 10 FN, recall 0.0000, precision 0.0000.
  - Visibility 40: 11 GT, 2 TP, 1 FP, 9 FN, recall 0.1818, precision 0.6667.
  - Visibility 70: 11 GT, 3 TP, 0 FP, 8 FN, recall 0.2727, precision 1.0000.
  - Visibility 100: 16 GT, 13 TP, 1 FP, 3 FN, recall 0.8125, precision 0.9286.
- Saved YOLO11 visual comparisons for 16 images plus a contact sheet at
  `outputs/yolo11_baseline/yolo11n_conf025_iou050_img960/comparison_contact_sheet.jpg`.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m py_compile src/forestpersons/*.py scripts/visualize_forestpersons.py tests/*.py`.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m pytest tests -q`: 5 tests passed.
- Reorganized the project layout:
  - Internship specification PDFs moved to `docs/specs/`.
  - Research papers moved to `docs/papers/`.
  - Demo images moved to `assets/sample_images/`.
  - YOLO weights moved to `models/`.
  - DINOv3 exploratory output images moved to `outputs/dinov3_exploration/`.
  - Legacy YOLO exploratory runs moved to `outputs/yolo_exploration/`.
- Updated `test_dinov3.py`, `test_yolo.py`, and
  `scripts/evaluate_yolo11_forestpersons.py` to use the organized paths.
- Added `scripts/create_forestpersons_val_subset.py` for reproducible
  validation subset manifest generation.
- Added `scripts/download_forestpersons_subset.py` for approved manifest-based
  image downloads with size, RGB mode, and dimension verification.
- Added `scripts/run_yolo11_validation_grid.py` to run YOLO11n/YOLO11s over
  confidence thresholds 0.05, 0.10, and 0.25 using the same manifest.
- Updated `scripts/evaluate_yolo11_forestpersons.py` to report inference time
  and false positives per image.
- Prepared a proposed 600-image ForestPersons validation subset manifest at
  `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`.
  Selection seed: 20261010. Target: 150 images per visibility level.
  The resulting image-level targets are exactly 150 each for visibility 20, 40,
  70, and 100. Actual annotation counts are 20 = 174, 40 = 170, 70 = 193,
  100 = 212, because images can contain multiple persons.
- Proposed subset diversity summary: 55 sequence folders, poses represented by
  image counts sitting = 437, lying = 156, standing = 75; seasons summer = 491,
  fall = 62, winter = 47; cameras See3CAM = 473, GoPro = 92, Sony a57 = 35;
  scenes forest = 572, road = 15, valley = 13; weather clear = 509, rain = 91.
- Estimated proposed subset size from Hugging Face `Content-Length` metadata:
  292,958,727 bytes total, approximately 279.39 MiB. Since 20 selected images
  already exist locally, the additional download would be 283,389,216 bytes,
  approximately 270.26 MiB.
- Downloaded and verified the approved 600-image validation subset. The first
  serial attempt was interrupted after partial progress because it was too slow;
  the downloader was then updated to use parallel verified downloads with
  retries. Final verification: 600/600 manifest images present, RGB, expected
  dimensions, and matching byte sizes.
- Ran the YOLO validation grid on the 600-image subset using YOLO11n and YOLO11s
  at confidence thresholds 0.05, 0.10, and 0.25; all runs used IoU matching
  threshold 0.50, image size 960, CPU inference, person-class filtering, and
  one-to-one greedy IoU matching.
- Saved grid outputs under
  `outputs/yolo11_validation_grid/forestpersons_val_600_seed20261010/`,
  including `summary_metrics.csv`, `README.md`, per-run `metrics.json`,
  per-run `matches.csv`, comparison images, plots, and false-negative examples.
- YOLO11 validation grid summary on 600 images / 749 GT persons:
  - YOLO11n conf 0.05: recall 0.3645, precision 0.5220, FP/image 0.417,
    seconds/image 0.0535; recall by visibility 20 = 0.1494, 40 = 0.2118,
    70 = 0.3575, 100 = 0.6698.
  - YOLO11n conf 0.10: recall 0.3351, precision 0.6765, FP/image 0.200,
    seconds/image 0.0572; recall by visibility 20 = 0.1264, 40 = 0.2000,
    70 = 0.3161, 100 = 0.6321.
  - YOLO11n conf 0.25: recall 0.2603, precision 0.8058, FP/image 0.078,
    seconds/image 0.0522; recall by visibility 20 = 0.0517, 40 = 0.1294,
    70 = 0.2539, 100 = 0.5425.
  - YOLO11s conf 0.05: recall 0.4673, precision 0.5785, FP/image 0.425,
    seconds/image 0.0971; recall by visibility 20 = 0.1782, 40 = 0.3353,
    70 = 0.5181, 100 = 0.7642.
  - YOLO11s conf 0.10: recall 0.4326, precision 0.7431, FP/image 0.187,
    seconds/image 0.0979; recall by visibility 20 = 0.1379, 40 = 0.2941,
    70 = 0.4922, 100 = 0.7311.
  - YOLO11s conf 0.25: recall 0.3618, precision 0.8658, FP/image 0.070,
    seconds/image 0.0936; recall by visibility 20 = 0.0920, 40 = 0.2176,
    70 = 0.3886, 100 = 0.6745.
- Best recall in this grid was YOLO11s at confidence 0.05: overall recall
  0.4673 and precision 0.5785.
- Saved false-negative examples at
  `outputs/yolo11_validation_grid/forestpersons_val_600_seed20261010/false_negative_examples_contact_sheet.jpg`.
- Created isolated DINOv3 environment `.venv-dinov3` with
  `--system-site-packages` to reuse the existing Anaconda Torch installation,
  then installed `transformers`, `huggingface-hub`, `safetensors`, and
  `accelerate`.
- Verified DINOv3 environment imports:
  - Torch 2.9.0.
  - Transformers 5.19.0.
  - Hugging Face Hub 2.2.0.
  - Safetensors 0.8.0.
  - Apple MPS is available and built.
- Verified the DINOv3 model used by the existing `test_dinov3.py`
  (`facebook/dinov3-vits16-pretrain-lvd1689m`) loads successfully from the
  local Hugging Face cache in `.venv-dinov3`; no additional model download was
  required for this check. The existing working DINOv3 script was not modified.
- Prepared and approved a 400-image ForestPersons train subset manifest at
  `outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json`.
  Selection seed: 20261011. Target: 100 image-level samples per visibility
  level. Actual annotation counts are 20 = 114, 40 = 119, 70 = 131,
  100 = 160 because images can contain multiple persons.
- Downloaded and verified the approved 400-image train subset into
  `data/images/`, preserving official ForestPersons relative paths.
- Train subset download result:
  - Downloaded images: 400.
  - Verified images: 400.
  - Downloaded bytes: 196,234,816 bytes, approximately 187.14 MiB.
  - All files are RGB, match manifest dimensions, and match expected byte sizes.
- Verified the 400-image train subset against `data/annotations/train.json`:
  - Manifest images: 400.
  - Unique image files: 400.
  - Missing files: 0.
  - Bad dimensions or image modes: 0.
  - Images without train annotations: 0.
  - Files overlapping the 600-image validation manifest: 0.
  - Annotation visibility counts: 20 = 114, 40 = 119, 70 = 131, 100 = 160.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m pytest tests -q`: 5 tests
  passed.
- Implemented the first frozen-DINOv3 detector experiment without modifying
  `test_dinov3.py` or existing YOLO11 result folders.
- Added shared detection utilities in `src/forestpersons/detection.py` for
  IoU, one-to-one matching, metric summaries, and visual comparisons.
- Added `src/forestpersons/dinov3_detector.py` with:
  - DINOv3 ViT-S/16 model loading.
  - MPS/CPU device selection.
  - Short-side 672 resizing without cropping.
  - Patch-grid feature extraction.
  - Compressed feature-cache read/write helpers.
  - Center-patch GT assignment.
  - A lightweight objectness and box-regression head.
  - Confidence filtering and NMS decoding.
- Added DINOv3 detector scripts:
  - `scripts/benchmark_dinov3_features.py`.
  - `scripts/cache_dinov3_features.py`.
  - `scripts/train_dinov3_detector_head.py`.
  - `scripts/evaluate_dinov3_detector.py`.
- Ran corrected MPS-synchronized DINOv3 feature benchmark on 20 approved train
  images at short-side 672. Result: 0.455 seconds/image on average. Typical
  1920x1080 image becomes 1200x672, producing a 75 x 42 patch grid with
  feature shape 42 x 75 x 384. Benchmark saved to
  `outputs/dinov3_detector/benchmark_train20_short672_sync.json`.
- Cached frozen DINOv3 features for the approved 400-image train subset under
  `outputs/dinov3_detector/features/train_400_short672/`.
- Cached frozen DINOv3 features for the existing 600-image validation subset
  under `outputs/dinov3_detector/features/val_600_short672/`. Validation cache
  extraction summary: 600/600 extracted, 0 skipped, 269.71 seconds total,
  0.450 seconds/image.
- Trained only the lightweight DINOv3 detection head for 12 epochs on cached
  train features. The DINOv3 backbone remained frozen. Training output saved to
  `outputs/dinov3_detector/head_v1/`.
- DINOv3 head assignment/training details:
  - GT boxes are resized by the same image scale used for feature extraction.
  - Each GT person is assigned to the patch containing the resized box center.
  - Box targets are center offsets relative to the patch center plus log width
    and log height in patch-size units.
  - If multiple GT centers collide in one patch cell, the smaller-area box is
    kept; the 400-image train subset had 1 such collision.
  - Objectness loss: weighted binary cross entropy with logits.
  - Box loss: Smooth L1 on positive cells.
  - Total loss: objectness loss + 5.0 * box loss.
  - Final epoch loss: 0.3181.
- Evaluated DINOv3 head v1 on the same 600-image validation subset used for the
  YOLO11 baseline, with IoU 0.50, one-to-one matching, and no validation data
  used for training.
- DINOv3 head v1 threshold sweep on 600 images / 749 GT persons:
  - Conf 0.05: recall 0.8104, precision 0.0229, FP/image 43.227; recall by
    visibility 20 = 0.7069, 40 = 0.8176, 70 = 0.8031, 100 = 0.8962.
  - Conf 0.10: recall 0.7917, precision 0.0421, FP/image 22.462; recall by
    visibility 20 = 0.6609, 40 = 0.8000, 70 = 0.7876, 100 = 0.8962.
  - Conf 0.25: recall 0.7543, precision 0.0826, FP/image 10.460; recall by
    visibility 20 = 0.5977, 40 = 0.7412, 70 = 0.7617, 100 = 0.8868.
  - Conf 0.50: recall 0.6996, precision 0.1192, FP/image 6.453; recall by
    visibility 20 = 0.4770, 40 = 0.6588, 70 = 0.7409, 100 = 0.8774.
  - Conf 0.75: recall 0.6368, precision 0.1546, FP/image 4.348; recall by
    visibility 20 = 0.4023, 40 = 0.5588, 70 = 0.6891, 100 = 0.8443.
  - Conf 0.90: recall 0.4486, precision 0.1920, FP/image 2.357; recall by
    visibility 20 = 0.2241, 40 = 0.4412, 70 = 0.5233, 100 = 0.5708.
- Compared DINOv3 head v1 with the existing best YOLO11 baseline:
  - YOLO11s conf 0.05: recall 0.4673, precision 0.5785, FP/image 0.425,
    visibility-20 recall 0.1782.
  - DINOv3 head v1 conf 0.90: recall 0.4486, precision 0.1920, FP/image 2.357,
    visibility-20 recall 0.2241.
  - DINOv3 head v1 is not competitive overall because false positives are much
    higher, but it shows useful low-visibility signal in frozen DINOv3 features.
- Saved DINOv3 detector report at `outputs/dinov3_detector/README.md`, summary
  CSV at `outputs/dinov3_detector/head_v1_threshold_summary.csv`, plots under
  `outputs/dinov3_detector/plots/`, and visual comparisons under
  `outputs/dinov3_detector/eval_val600_head_v1_conf090/comparisons/`.
- Inspected head v1 architecture, target assignment, losses, NMS, and
  evaluation. The main geometry issue found was that v1 used one width-derived
  scale for both axes; future code now derives separate `scale_x` and `scale_y`
  from cached resized dimensions. This is a small rounding correction and does
  not modify existing head v1 artifacts.
- Added `scripts/analyze_dinov3_assignments.py` to inspect GT center-patch
  assignment.
- Assignment analysis for cached DINOv3 features:
  - Train 400 subset: 525 annotations, 1 collided patch cell, 2 annotations in
    the collided cell, 1 annotation lost by the one-target-per-cell rule.
    The collision involved two sitting annotations at 70% visibility.
  - Validation 600 subset: 749 annotations, 0 collided patch cells.
  - Median original bbox area increases with visibility in the train subset:
    visibility 20 = 9,306 px², 40 = 12,444 px², 70 = 15,600 px²,
    100 = 16,074 px². Small/occluded persons are smaller, but center-cell
    collisions are not a major failure source.
- Implemented DINOv3 head v2 as a separate experiment under
  `outputs/dinov3_detector/head_v2_grid_focal/`.
- Head v2 architecture:
  - Frozen DINOv3 ViT-S/16 cached patch features.
  - 1x1 projection from 384 channels to 160 channels.
  - Two 3x3 convolution blocks over the patch grid to use neighboring patch
    context.
  - 1x1 objectness and box-regression heads.
  - Sigmoid focal loss for objectness with alpha 0.75 and gamma 2.0.
  - Smooth L1 box loss on positive center-patch cells.
- Trained DINOv3 head v2 for 20 epochs on the same cached 400-image train
  subset, without fine-tuning the DINOv3 backbone. Final loss: 0.0509.
- Evaluated DINOv3 head v1 recheck and head v2 with the same 600-image
  validation subset, IoU 0.50, one-to-one matching, and cached-head-only timing.
- DINOv3 head v2 selected threshold results:
  - Conf 0.25: recall 0.5741, precision 0.1785, FP/image 3.298; recall by
    visibility 20 = 0.3218, 40 = 0.5176, 70 = 0.6477, 100 = 0.7594.
  - Conf 0.30: recall 0.4579, precision 0.2546, FP/image 1.673; recall by
    visibility 20 = 0.2299, 40 = 0.4294, 70 = 0.5233, 100 = 0.6179.
  - Conf 0.39: recall 0.2430, precision 0.4365, FP/image 0.392; recall by
    visibility 20 = 0.1034, 40 = 0.2353, 70 = 0.2746, 100 = 0.3349.
  - Conf 0.40: recall 0.2323, precision 0.4690, FP/image 0.328; recall by
    visibility 20 = 0.0920, 40 = 0.2176, 70 = 0.2642, 100 = 0.3255.
  - Conf 0.45: recall 0.1348, precision 0.6121, FP/image 0.107; recall by
    visibility 20 = 0.0287, 40 = 0.1176, 70 = 0.1813, 100 = 0.1792.
- Matched false-positive comparison near YOLO11s's 0.425 FP/image:
  - YOLO11s reference at conf 0.05: recall 0.4673, precision 0.5785,
    FP/image 0.425, visibility-20 recall 0.1782.
  - DINOv3 head v1 recheck at conf 0.97: recall 0.1175, precision 0.2914,
    FP/image 0.357, visibility-20 recall 0.0460.
  - DINOv3 head v2 at conf 0.39: recall 0.2430, precision 0.4365,
    FP/image 0.392, visibility-20 recall 0.1034.
- Head v2 improves substantially over head v1 at matched false-positive rates,
  but it still does not beat YOLO11s overall or at 20% visibility when FP/image
  is held near YOLO11s's operating point.
- Saved DINOv3 head v2 artifacts:
  - Checkpoint: `outputs/dinov3_detector/head_v2_grid_focal/dinov3_head.pt`.
  - Report: `outputs/dinov3_detector/head_v2_grid_focal/README.md`.
  - Threshold summary:
    `outputs/dinov3_detector/head_v2_grid_focal_threshold_summary.csv`.
  - v1/v2 comparison:
    `outputs/dinov3_detector/head_v1_recheck_v2_comparison.csv`.
  - Plots:
    `outputs/dinov3_detector/plots/head_v1_recheck_v2_recall_vs_fp.png` and
    `outputs/dinov3_detector/plots/head_v1_recheck_v2_recall20_vs_fp.png`.
  - Visual comparisons at v2 conf 0.39:
    `outputs/dinov3_detector/eval_val600_head_v2_grid_focal_conf039/comparisons/`.
- Audited the DINOv3 evaluation path before implementing head v3:
  - Confirmed train/validation manifest overlap remains 0 image files.
  - Confirmed evaluations use the same 600 validation images, IoU 0.50,
    one-to-one greedy matching, and NMS before matching.
  - Confirmed future DINOv3 target building and decoding use separate
    `scale_x` and `scale_y` from cached resized dimensions.
  - No significant evaluation bug was found that invalidates the rechecked
    v1/v2 comparisons.
- Implemented DINOv3 head v3 as an FCOS-inspired detector under
  `outputs/dinov3_detector/head_v3_fcos/`.
- Head v3 architecture:
  - Frozen DINOv3 ViT-S/16 cached patch features.
  - 1x1 projection from 384 channels to 160 channels.
  - Two 3x3 convolution blocks over the patch grid.
  - Classification/objectness branch.
  - Box branch predicting left/top/right/bottom distances from each patch
    center.
  - Centerness branch.
- Head v3 assignment:
  - All patch centers inside a GT box and inside a center-sampling region are
    positive.
  - If a small GT box has no patch center inside, the nearest center patch is
    forced positive.
  - If multiple GT boxes compete for a patch, the smaller-area box wins.
  - Train target stats: 525 annotations, 4,140 positive cells, 10.35 positive
    cells/image, 0 center fallbacks, 16 multi-GT overwrites.
- Trained DINOv3 head v3 for 20 epochs on the same cached 400-image train
  subset, without fine-tuning the DINOv3 backbone. Final loss: 0.6459; final
  box loss: 0.0241.
- DINOv3 head v3 selected threshold results on 600 validation images / 749 GT:
  - Conf 0.45: recall 0.7864, precision 0.2349, FP/image 3.197; recall by
    visibility 20 = 0.5977, 40 = 0.7529, 70 = 0.8135, 100 = 0.9481.
  - Conf 0.50: recall 0.7637, precision 0.3555, FP/image 1.728; recall by
    visibility 20 = 0.5460, 40 = 0.7118, 70 = 0.8031, 100 = 0.9481.
  - Conf 0.60: recall 0.6702, precision 0.5955, FP/image 0.568; recall by
    visibility 20 = 0.4195, 40 = 0.5588, 70 = 0.7306, 100 = 0.9245.
  - Conf 0.62: recall 0.6395, precision 0.6447, FP/image 0.440; recall by
    visibility 20 = 0.3678, 40 = 0.5235, 70 = 0.6995, 100 = 0.9009.
  - Conf 0.64: recall 0.5995, precision 0.6887, FP/image 0.338; recall by
    visibility 20 = 0.3161, 40 = 0.4824, 70 = 0.6684, 100 = 0.8726.
- Matched false-positive comparison near YOLO11s's 0.425 FP/image:
  - YOLO11s reference at conf 0.05: recall 0.4673, precision 0.5785,
    FP/image 0.425, visibility-20 recall 0.1782.
  - DINOv3 head v1 recheck at conf 0.97: recall 0.1175, precision 0.2914,
    FP/image 0.357, visibility-20 recall 0.0460.
  - DINOv3 head v2 at conf 0.39: recall 0.2430, precision 0.4365,
    FP/image 0.392, visibility-20 recall 0.1034.
  - DINOv3 head v3 at conf 0.62: recall 0.6395, precision 0.6447,
    FP/image 0.440, visibility-20 recall 0.3678.
- Head v3 improves over v2 and also beats the current YOLO11s baseline at the
  closest matched-FP operating point on this 600-image validation subset.
- Head v3 failure patterns at conf 0.62:
  - False negatives by visibility: 20 = 110, 40 = 81, 70 = 58, 100 = 21.
  - Worst false-negative images often contain multiple people; the head may
    detect one instance but miss nearby additional people.
  - False positives are often near person-like vegetation/body-part patterns or
    around the true person but not localized tightly enough for IoU 0.50.
- Saved DINOv3 head v3 artifacts:
  - Checkpoint: `outputs/dinov3_detector/head_v3_fcos/dinov3_head.pt`.
  - Report: `outputs/dinov3_detector/head_v3_fcos/README.md`.
  - Threshold summary:
    `outputs/dinov3_detector/head_v3_fcos_threshold_summary.csv`.
  - v1/v2/v3 comparison:
    `outputs/dinov3_detector/head_v1_v2_v3_comparison.csv`.
  - Plots:
    `outputs/dinov3_detector/plots/head_v1_v2_v3_recall_vs_fp.png`,
    `outputs/dinov3_detector/plots/head_v1_v2_v3_recall20_vs_fp.png`, and
    `outputs/dinov3_detector/plots/head_v1_v2_v3_precision_recall.png`.
  - Error analysis:
    `outputs/dinov3_detector/error_analysis_head_v3_fcos_conf062/`.
  - Visual comparisons:
    `outputs/dinov3_detector/eval_val600_head_v3_fcos_conf062/comparisons/`.
- Prepared a proposed 5,000-image ForestPersons train subset manifest at
  `outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json`.
  This manifest preserves all previous 400 train images and has zero overlap
  with the 600-image validation subset.
- Proposed 5,000-image train subset selection:
  - Seed: 20261012.
  - Target image counts: 1,250 images each for visibility 20, 40, 70, and 100.
  - Actual annotation counts: visibility 20 = 1,423, 40 = 1,440,
    70 = 1,595, 100 = 1,957.
  - Pose image counts: sitting = 3,086, lying = 1,553, standing = 909.
  - Season image counts: summer = 4,203, fall = 435, winter = 362.
  - Camera image counts: See3CAM = 4,147, GoPro = 665,
    Sony a57 (24-70) = 188.
  - Place image counts: forest = 4,572, valley = 417, road = 11.
  - Weather image counts: clear = 4,383, rain = 617.
  - Sequence folders represented: 258.
- Estimated 5,000-image train subset storage before download:
  - Total selected image bytes from Hugging Face Content-Length metadata:
    2,417,548,434 bytes, approximately 2,305.55 MiB / 2.25 GiB.
  - Already local: 400 images, 196,234,816 bytes.
  - Additional image download required: 2,221,313,618 bytes,
    approximately 2,118.41 MiB / 2.07 GiB.
  - Estimated DINOv3 feature cache size at short-side 672:
    12,254,054,400 raw feature bytes, approximately 11.41 GiB.
- Added new preparation scripts:
  - `scripts/create_forestpersons_train_subset.py` for reproducible train
    subset manifests.
  - `scripts/update_manifest_hf_sizes.py` for parallel Hugging Face
    Content-Length metadata collection with normal SSL verification.
  - `scripts/download_forestpersons_subset_verified.py` for future subset
    downloads with normal SSL verification and RGB/dimension/byte checks.
- After user approval, downloaded and verified the approved 5,000-image train
  subset into `data/images/` without downloading the full dataset.
  - Downloaded new images: 4,600.
  - Already local images reused: 400.
  - Verified images: 5,000.
  - Additional bytes written: 2,221,313,618 bytes, approximately 2,118.41 MiB.
  - Total selected subset image bytes: 2,417,548,434 bytes, approximately
    2,305.55 MiB / 2.25 GiB.
- Reused the existing 400 cached DINOv3 feature files and extracted frozen
  DINOv3 ViT-S/16 features for the remaining 4,600 training images at
  short-side 672.
  - Feature cache root:
    `outputs/dinov3_detector/features/train_5000_seed20261012_short672/`.
  - Cached feature files: 5,000.
  - Missing feature files after extraction: 0.
  - Extraction device: MPS.
  - New extraction time: 2,102.48 seconds total, 0.457 seconds/image.
  - Feature cache disk usage: approximately 11 GiB.
- Trained a new DINOv3 FCOS-style detection head from scratch on the 5,000-image
  train subset, preserving the 400-image v3 checkpoint and all YOLO11 results.
  - Output directory:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/`.
  - Checkpoint: `dinov3_head.pt`.
  - Backbone: frozen `facebook/dinov3-vits16-pretrain-lvd1689m`.
  - Head architecture: same v3 FCOS-style head as the 400-image experiment.
  - Epochs: 20.
  - Learning rate: 0.0003.
  - Weight decay: 0.0001.
  - Seed: 20261012.
  - Final loss: 0.6321.
  - Final objectness loss: 0.00065.
  - Final box loss: 0.01827.
  - Final centerness loss: 0.59495.
  - Positive cells per epoch: 50,600.
  - Assignment collisions: 2.
  - Center fallbacks: 7.
  - Multi-GT overwrites: 121.
- Verified the 5,000-image train manifest and the 600-image validation manifest
  have zero overlapping image files.
- Evaluated the 5,000-image DINOv3 FCOS head on the same 600-image validation
  subset, using the same IoU 0.50, NMS 0.50, one-to-one matching protocol as
  the previous DINOv3 and YOLO11 comparisons.
- DINOv3 FCOS head trained on 5,000 images, selected threshold results on
  600 validation images / 749 GT persons:
  - Conf 0.50: recall 0.8331, precision 0.4883, FP/image 1.090; recall by
    visibility 20 = 0.6667, 40 = 0.7882, 70 = 0.8912, 100 = 0.9528.
  - Conf 0.55: recall 0.8198, precision 0.6158, FP/image 0.638; recall by
    visibility 20 = 0.6437, 40 = 0.7706, 70 = 0.8808, 100 = 0.9481.
  - Conf 0.60: recall 0.7784, precision 0.7127, FP/image 0.392; recall by
    visibility 20 = 0.5632, 40 = 0.7235, 70 = 0.8601, 100 = 0.9245.
  - Conf 0.62: recall 0.7570, precision 0.7580, FP/image 0.302; recall by
    visibility 20 = 0.5345, 40 = 0.6882, 70 = 0.8394, 100 = 0.9198.
  - Conf 0.64: recall 0.7316, precision 0.7908, FP/image 0.242; recall by
    visibility 20 = 0.4885, 40 = 0.6647, 70 = 0.8031, 100 = 0.9198.
  - Conf 0.66: recall 0.7156, precision 0.8221, FP/image 0.193; recall by
    visibility 20 = 0.4828, 40 = 0.6471, 70 = 0.7876, 100 = 0.8962.
- Matched false-positive comparison near YOLO11s's 0.425 FP/image:
  - YOLO11s pretrained at conf 0.05: recall 0.4673, precision 0.5785,
    FP/image 0.425, visibility-20 recall 0.1782.
  - DINOv3 FCOS v3 trained on 400 images at conf 0.62: recall 0.6395,
    precision 0.6447, FP/image 0.440, visibility-20 recall 0.3678.
  - DINOv3 FCOS v3 trained on 5,000 images at conf 0.60: recall 0.7784,
    precision 0.7127, FP/image 0.392, visibility-20 recall 0.5632.
- The 5,000-image DINOv3 head improves over the 400-image DINOv3 v3 head on
  this selected validation subset. The improvement is visible overall and at
  20% visibility, while using a slightly lower FP/image operating point than
  YOLO11s.
- DINOv3 5,000-image error analysis at conf 0.60:
  - False negatives by visibility: 20 = 76, 40 = 47, 70 = 27, 100 = 16.
  - False positives by nearest GT visibility: 20 = 74, 40 = 55, 70 = 58,
    100 = 48.
  - Remaining hard cases include very low-visibility people, winter scenes,
    images with multiple nearby people, and predictions near the true person
    that fail the IoU 0.50 localization threshold.
- Saved 5,000-image DINOv3 artifacts:
  - Experiment report:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/README.md`.
  - Checkpoint:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`.
  - Training history:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/train_history.json`.
  - Full threshold comparison:
    `outputs/dinov3_detector/head_v3_400_vs_5000_yolo11s_comparison.csv`.
  - Matched-FP comparison:
    `outputs/dinov3_detector/head_v3_400_vs_5000_yolo11s_matched_fp.csv`.
  - Plots:
    `outputs/dinov3_detector/plots/head_v3_400_vs_5000_precision_recall.png`,
    `outputs/dinov3_detector/plots/head_v3_400_vs_5000_recall_vs_fp.png`,
    and
    `outputs/dinov3_detector/plots/head_v3_400_vs_5000_recall20_vs_fp.png`.
  - Visual comparisons:
    `outputs/dinov3_detector/eval_val600_head_v3_fcos_train5000_conf060/comparisons/`.
  - Error analysis:
    `outputs/dinov3_detector/error_analysis_head_v3_fcos_train5000_conf060/`.
- Full-train scaling estimates based on the measured 5,000-image subset:
  - Full ForestPersons train split: 67,686 images.
  - Estimated full train image storage: approximately 32.73 GB / 30.48 GiB.
  - Estimated remaining train image download after the 5,000-image subset:
    approximately 30.31 GB / 28.23 GiB.
  - Estimated full train feature cache at short-side 672: approximately
    165.89 GB / 154.49 GiB.
  - Estimated remaining feature cache after the 5,000-image subset:
    approximately 153.63 GB / 143.08 GiB.
  - Estimated full train feature extraction time at the measured
    0.457 seconds/image MPS speed: approximately 8.59 hours from scratch, or
    7.96 additional hours for the remaining 62,686 train images.
  - Estimated full validation feature extraction time for 18,243 validation
    images at the same speed: approximately 2.32 hours, excluding image
    download time.
  - Head-only validation evaluation is fast once features are cached:
    0.00714 seconds/image for the 5,000-image head at conf 0.60 on CPU.
- Ran `PYTHONPATH=src .venv-dinov3/bin/python -m py_compile
  src/forestpersons/*.py scripts/*.py`: passed.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m pytest tests -q`: 5 tests
  passed.
- Began the controlled YOLO11s fine-tuning phase for comparison against the
  5,000-image frozen-DINOv3 FCOS detector.
- Re-read `AGENTS.md`, `PROGRESS.md`, and
  `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/README.md`.
- Verified the YOLO environment:
  - Ultralytics 8.4.175.
  - Torch 2.9.0.
  - Apple MPS is built and available.
  - `models/yolo11s.pt` loads successfully.
- Added `scripts/prepare_yolo_forestpersons.py` to convert approved
  ForestPersons subsets to Ultralytics YOLO detection format.
  - All visibility-labeled ForestPersons person categories are mapped to one
    class: `person`.
  - Visibility and pose metadata are preserved in sidecar CSV files for custom
    evaluation.
  - COCO xywh boxes are clipped to image bounds and converted to normalized
    YOLO center-x, center-y, width, height.
  - Images are symlinked rather than copied, avoiding another multi-GB image
    tree.
- Added `tests/test_yolo_conversion.py` for YOLO bbox normalization and
  clipping behavior.
- Prepared the YOLO training dataset at
  `outputs/yolo11_finetune/forestpersons_yolo_dataset/`.
  - Dataset YAML:
    `outputs/yolo11_finetune/forestpersons_yolo_dataset/forestpersons_yolo.yaml`.
  - Train images: 5,000.
  - Train label files: 5,000.
  - Train visibility-labeled person boxes: 6,431.
  - Validation images: 600.
  - Validation label files: 600.
  - Validation visibility-labeled person boxes: 749.
  - Train/validation overlap: 0 image files.
  - Boxes clipped or dropped by conversion: 0.
  - Validation annotations skipped without visibility level: 1 `unknown_0`
    annotation. This was excluded to match the DINOv3/custom evaluation
    protocol, which evaluates visibility-labeled persons.
  - YOLO dataset directory size: approximately 25 MB because image files are
    symlinks to `data/images/`.
- Ran a YOLO11s smoke-training test on a 64-image train / 64-image validation
  file-list subset.
  - Command used `models/yolo11s.pt`, image size 640, batch size 8, 1 epoch,
    device `mps`, seed 20261012, deterministic mode, and workers 0.
  - Hardware reported by Ultralytics: Apple M4 with MPS.
  - Reported train memory: about 4.45 GB.
  - Wall time: 39 seconds.
  - The run completed and wrote `best.pt` and `last.pt`.
  - Because the smoke command used a relative `project=` path, Ultralytics
    placed the output under
    `runs/detect/outputs/yolo11_finetune/smoke_runs/yolo11s_train5000_smoke64/`.
    The full run should use an absolute project path to keep outputs directly
    under `outputs/yolo11_finetune/train_runs/`.
- Added YOLO fine-tuning preparation notes and the proposed full training
  command to `outputs/yolo11_finetune/README.md`.
- Estimated full YOLO11s fine-tuning requirements before starting the long run:
  - Proposed configuration: 50 epochs, image size 640, batch size 8, MPS,
    seed 20261012.
  - Runtime estimate from the smoke run: several hours; roughly 8-20 hours is
    plausible because the smoke subset is too small for precise extrapolation.
  - Additional disk estimate: under roughly 1 GB for checkpoints, logs, plots,
    label caches, and validation outputs when periodic checkpoint saving is not
    enabled.
- Ran the full 5,000-image YOLO11s fine-tuning experiment after user
  confirmation.
  - Training data: exactly the 5,000-image ForestPersons train manifest used
    by the DINOv3 FCOS 5k experiment.
  - Validation data: the same 600-image ForestPersons validation manifest used
    by the DINOv3 and pretrained YOLO11s comparisons.
  - Configuration: `models/yolo11s.pt`, 1 `person` class, image size 640,
    batch size 8, device `mps`, workers 0, seed 20261012, deterministic mode.
  - The run completed validation through epoch 47. It was intentionally stopped
    during epoch 48 at the user's request (`ok 47 enough`), so epoch 48 is not
    treated as a completed result.
  - Completed epoch 47 elapsed time reported by Ultralytics:
    28,510.1 seconds, about 7.92 hours. The interrupted partial epoch 48 added
    a few more minutes.
  - MPS memory during training was stable at about 4.4-4.5 GB.
  - Best built-in validation checkpoints:
    - Best `mAP50`: epoch 44, 0.8965.
    - Best `mAP50-95`: epoch 46, 0.55849.
    - Best built-in recall: epoch 37, 0.86515.
  - Saved checkpoints:
    `outputs/yolo11_finetune/train_runs/yolo11s_forestpersons_train5000_seed20261012_img640_b8_e50/weights/best.pt`
    and `last.pt`.
- Evaluated the fine-tuned YOLO11s `best.pt` checkpoint with the existing
  custom ForestPersons protocol: 600 validation images, 749 visibility-labeled
  person GT boxes, IoU >= 0.50, one-to-one greedy matching, person-only
  predictions, and visibility-specific recall.
  - Evaluation used image size 640 and MPS inference.
  - Threshold sweep: 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70.
  - Closest measured point to the old 0.425 FP/image target was confidence
    0.025:
    - Recall: 91.46%.
    - Precision: 69.76%.
    - FP/image: 0.495.
    - False positives: 297.
    - False negatives: 64.
    - Recall by visibility: 20 = 78.16%, 40 = 90.00%,
      70 = 97.93%, 100 = 97.64%.
    - Inference time: about 0.0228 seconds/image on MPS, including per-image
      Ultralytics overhead.
  - A lower-FP operating point at confidence 0.05 achieved:
    - Recall: 90.39%.
    - Precision: 76.50%.
    - FP/image: 0.347.
    - Recall by visibility: 20 = 77.01%, 40 = 87.65%,
      70 = 97.41%, 100 = 97.17%.
  - At comparable FP/image, fine-tuned YOLO11s is now the strongest detector on
    the current selected 600-image validation subset.
    - Pretrained YOLO11s at confidence 0.05: recall 46.73%, precision 57.85%,
      FP/image 0.425, recall@20 17.82%.
    - DINOv3 FCOS 5k at confidence 0.60: recall 77.84%, precision 71.27%,
      FP/image 0.392, recall@20 56.32%.
    - Fine-tuned YOLO11s at confidence 0.025: recall 91.46%, precision 69.76%,
      FP/image 0.495, recall@20 78.16%.
    - Fine-tuned YOLO11s at confidence 0.05 also beats DINOv3 FCOS 5k while
      using fewer FP/image than DINOv3: recall 90.39%, precision 76.50%,
      FP/image 0.347, recall@20 77.01%.
  - Saved evaluation outputs:
    - `outputs/yolo11_finetune/eval_val600_best_epoch47_grid/`
    - `outputs/yolo11_finetune/summary_epoch47_best/yolo11s_finetuned_threshold_summary.csv`
    - `outputs/yolo11_finetune/summary_epoch47_best/model_comparison_all_thresholds.csv`
    - `outputs/yolo11_finetune/summary_epoch47_best/model_comparison_matched_fp.csv`
    - `outputs/yolo11_finetune/summary_epoch47_best/plots/`
- Added `scripts/summarize_yolo11_finetune.py` to aggregate YOLO fine-tuning
  sweep metrics and generate recall-vs-FP, recall@20-vs-FP, and
  precision-recall plots.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m pytest tests -q`: 8 tests
  passed.
- Ran `PYTHONPATH=src .venv-yolo/bin/python -m py_compile
  src/forestpersons/*.py scripts/*.py tests/*.py`: passed.

## Annotation Analysis

- Split counts:
  - Train: 67,686 images, 145,816 annotations.
  - Validation: 18,243 images, 37,395 annotations.
  - Test: 10,553 images, 20,867 annotations.
- Categories: 13 categories, all with `supercategory` set to `person`.
  Category names encode pose and visibility, e.g. `standing_100`,
  `sitting_70`, `lying_40`; category 13 is `unknown_0`.
- Visibility levels observed:
  - Train: 20 = 12,326; 40 = 14,708; 70 = 27,259; 100 = 91,523.
  - Validation: 20 = 3,303; 40 = 4,376; 70 = 7,945; 100 = 21,770.
  - Test: 20 = 1,946; 40 = 2,454; 70 = 4,499; 100 = 11,968.
- Poses observed:
  - Train: lying = 19,255; sitting = 43,069; standing = 83,492.
  - Validation: lying = 3,956; sitting = 8,983; standing = 24,456.
  - Test: lying = 2,733; sitting = 8,518; standing = 9,616.
- Bounding boxes:
  - All boxes have positive width and height.
  - All boxes are within their image dimensions.
  - All annotation `area` values match `bbox_width * bbox_height`.
- Images without annotations:
  - Train: 13.
  - Validation: 0.
  - Test: 7.
- Validation produced zero loader warnings for all three official split files.

## Blockers

- Only approved ForestPersons subsets have been downloaded: the 600-image
  validation subset and the 5,000-image train subset, which includes the
  original 400-image train subset. The full image dataset has not been
  downloaded.
- The base Anaconda environment still does not expose `transformers`,
  `datasets`, or `huggingface_hub`. Ultralytics is available only in the
  isolated `.venv-yolo` environment. DINOv3 dependencies are available only in
  `.venv-dinov3`, which reuses Anaconda Torch through `--system-site-packages`.
- Ground-truth visualization was first validated on the 32-image subset; later
  YOLO11 and DINOv3 visual comparisons were generated on the 600-image
  validation subset.
- YOLO11n metrics are smoke-test results only; the subset is small and balanced
  by target visibility, not representative of the full validation distribution.
- Per-visibility precision is approximate because predictions do not have
  visibility labels; unmatched predictions are assigned to the nearest
  ground-truth visibility in the same image for group FP accounting. Recall by
  visibility is the primary metric for this phase.
- The 600-image validation subset is more scientifically useful than the
  32-image smoke test, but it is still a subset and intentionally visibility-
  stratified rather than a full validation benchmark.
- Inference timings were measured on CPU and include per-image Ultralytics
  prediction overhead; they are useful for local comparison, not deployment
  timing.
- DINOv3 head v3 is currently the strongest DINOv3 detector and outperforms the
  pretrained YOLO11s baseline at the closest matched-FP operating point on the
  current 600-image validation subset. The 5,000-image version improves further
  but still needs confirmation on larger and less hand-balanced validation
  subsets.
- DINOv3 head inference timings reported in this phase are cached-head-only
  timings. End-to-end DINOv3 detector timing should include feature extraction,
  which benchmarked at about 0.455 seconds/image on MPS at short-side 672.
- The 5,000-image training script did not write precise total wall-clock
  training time. Future training scripts should record start/end timestamps and
  per-epoch wall time.
- The current comparison still uses a selected 600-image, visibility-stratified
  validation subset, not the official full ForestPersons validation benchmark.
- The first YOLO11s fine-tuning run was intentionally stopped during epoch 48.
  Completed epoch 47 and the saved `best.pt` checkpoint are valid for
  comparison, but the nominal 50-epoch run was not fully completed.
- YOLO11s fine-tuning and YOLO11s evaluation used image size 640. Earlier
  pretrained YOLO11s reference metrics used the existing protocol and should be
  interpreted as a reference baseline, not a fully optimized pretrained
  threshold/resolution search.
- Fine-tuned YOLO11s now outperforms frozen-DINOv3 FCOS on the current selected
  600-image validation subset, but this is still not the official full
  ForestPersons validation benchmark.

## Next Steps

1. Treat fine-tuned YOLO11s on the 5,000-image subset as the strongest current
   detector on the selected 600-image validation subset.
2. Use confidence 0.05 as a practical low-FP YOLO11s operating point and
   confidence 0.025 as the high-recall point closest to the previous
   0.425 FP/image comparison target.
3. Create a separate train-only development split for threshold/model selection
   so the 600-image validation subset remains a cleaner comparison set.
4. Evaluate fine-tuned YOLO11s and DINOv3 FCOS on a larger, less hand-balanced
   validation subset, then eventually the full 18,243-image ForestPersons
   validation split.
5. For DINOv3, prioritize architecture/hard-negative improvements only if the
   research goal requires frozen-backbone dense representations to compete with
   end-to-end fine-tuned YOLO; otherwise prioritize YOLO11s full-dataset
   fine-tuning as the practical detection baseline.

## 2026-10-11 — ForestPersons Evaluation Framework

Implemented a reusable evaluation layer before starting any new baseline
training.

Files added:

- `src/forestpersons/evaluation.py`
  - Local COCO-style one-class detection metrics:
    - AP averaged over IoU thresholds 0.50:0.95.
    - AP50.
    - AP75.
    - AR averaged over IoU thresholds with documented `maxDets = [1, 10, 100]`.
  - COCO result formatting using `[x, y, width, height]` boxes and a single
    `person` category ID.
  - Existing custom SAR metrics preserved:
    - Recall at explicit IoU threshold, default 0.50.
    - Precision.
    - False positives per image.
    - Recall by visibility level.
  - Confidence-threshold sweeps and matched FP/image budget selection.
  - Maximum recall is defined separately from COCO AR as the highest custom
    recall over the evaluated confidence thresholds at a stated IoU threshold.

- `scripts/evaluate_forestpersons_framework.py`
  - Unified local evaluator for existing YOLO and cached DINOv3 detector
    checkpoints.
  - Writes `summary.json`, `coco_results.json`, `confidence_sweep.csv`, and
    optional visual comparisons.
  - Keeps pilot-subset results explicitly labeled as non-official unless the
    manifest contains the complete official split.

- `tests/test_detection_evaluation.py`
  - Covers IoU behavior, duplicate detections, empty predictions, multiple
    people per image, COCO result formatting, COCO-style matching, and matched
    FP-budget selection.

Files modified:

- `src/forestpersons/__init__.py`
  - Exposes the new evaluation dataclasses and metric helpers.

Dependency finding:

- `pycocotools` is not installed in `.venv-yolo`.
- A small local COCO-style equivalent was implemented instead of installing new
  packages. This avoids changing the environment and keeps the smoke test
  reproducible on the current Mac setup.

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src $LOGXY_ARTIFACT_ROOT/.venv-yolo/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 15 passed.

Smoke tests:

- YOLO11s fine-tuned checkpoint:
  - Checkpoint:
    `outputs/yolo11_finetune/train_runs/yolo11s_forestpersons_train5000_seed20261012_img640_b8_e50/weights/best.pt`
  - Images: first 30 images from
    `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`
  - Ground-truth annotations: 33.
  - Device: CPU.
  - Image size: 640.
  - Prediction minimum confidence: 0.001.
  - NMS IoU: 0.50.
  - Custom matching IoU: 0.50.
  - COCO-style pilot metrics:
    - AP: 0.3890.
    - AP50: 0.6811.
    - AP75: 0.3970.
    - AR @ maxDets 100: 0.5424.
  - Best row under FP/image <= 0.425:
    - Confidence: 0.25.
    - Recall: 0.6364.
    - Precision: 0.6364.
    - FP/image: 0.4000.
  - Output directory:
    `outputs/evaluation_framework/smoke_yolo11s_finetuned_val30/`

- Frozen DINOv3 ViT-S/16 + FCOS trained on 5,000 images:
  - Checkpoint:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`
  - Feature cache:
    `outputs/dinov3_detector/features/val_600_short672`
  - Images: same first 30 validation-manifest images.
  - Ground-truth annotations: 33.
  - Device: CPU, cached-head-only inference.
  - Prediction minimum confidence: 0.001.
  - NMS IoU: 0.50.
  - Custom matching IoU: 0.50.
  - COCO-style pilot metrics:
    - AP: 0.1988.
    - AP50: 0.5352.
    - AP75: 0.0317.
    - AR @ maxDets 100: 0.3455.
  - Best row under FP/image <= 0.425 among the tested thresholds:
    - Confidence: 0.70.
    - Recall: 0.3939.
    - Precision: 0.7647.
    - FP/image: 0.1333.
  - Output directory:
    `outputs/evaluation_framework/smoke_dinov3_fcos5k_val30/`

Important limitation:

- The smoke-test subset contains only the first 30 images from the selected
  validation manifest and all 33 annotations in this small slice are 20%
  visibility. These numbers verify the evaluator plumbing, output format, and
  model integration only. They are not official ForestPersons benchmark scores
  and should not be used for scientific conclusions.

Recommended next action:

1. Run the new framework on the full existing 600-image validation subset for
   fine-tuned YOLO11s and DINOv3 FCOS, then compare those numbers with the
   previous custom evaluator outputs.
2. After confirming consistency, prepare the full validation-set image download
   and use this evaluator for official-style AP/AR on all 18,243 validation
   images.

## 2026-10-11 — 600-Image Pilot Evaluation With COCOeval Verification

Validated the new ForestPersons evaluation framework on the complete existing
600-image validation subset. No new images were downloaded, no training was
started, no checkpoints were modified, and the official test split was not
evaluated.

Dependency update:

- Installed `pycocotools==2.0.11` into `.venv-yolo`.
- Installed `pycocotools==2.0.11` into `.venv-dinov3`.
- The local COCO-style evaluator was compared against official
  `pycocotools.cocoeval.COCOeval` on the same predictions and ground truth.
  For both 600-image model outputs, local AP/AP50/AP75/AR@1/AR@10/AR@100
  matched pycocotools exactly.

Evaluation settings:

- Dataset: selected 600-image ForestPersons validation subset, not the full
  official validation split.
- Manifest:
  `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`
- Images evaluated: 600.
- Ground-truth person annotations evaluated: 749.
- COCO category mapping: all ForestPersons visibility/pose categories mapped
  to one `person` category with ID 1.
- COCO metrics:
  - AP over IoU thresholds 0.50:0.95.
  - AP50.
  - AP75.
  - AR@1, AR@10, AR@100.
- Custom metrics:
  - One-to-one greedy matching at IoU 0.50.
  - Recall, precision, false positives per image.
  - Recall by visibility level.
  - Confidence-threshold sweeps.
  - Matched FP/image budgets: 0.392, 0.425, and 0.500.
- Maximum recall definition:
  - Highest custom recall at IoU 0.50 over the evaluated confidence-threshold
    sweep, without a false-positive constraint.
  - This is separate from COCO AR.

Model settings:

- Fine-tuned YOLO11s:
  - Checkpoint:
    `outputs/yolo11_finetune/train_runs/yolo11s_forestpersons_train5000_seed20261012_img640_b8_e50/weights/best.pt`
  - Input size: `imgsz=640`.
  - Device for pilot evaluation: CPU.
  - Minimum prediction confidence collected before sweep: 0.001.
  - NMS IoU: 0.50.

- Frozen DINOv3 ViT-S/16 + FCOS, trained on 5,000 images:
  - Checkpoint:
    `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`
  - Feature cache:
    `outputs/dinov3_detector/features/val_600_short672`
  - Preprocessing: cached DINOv3 features from short-side 672 resizing.
  - Device for pilot evaluation: CPU, cached-head-only inference.
  - Minimum prediction confidence collected before sweep: 0.001.
  - NMS IoU: 0.50.

600-image pilot comparison:

| Model | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | Operating conf | Recall@0.50 | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned 5k | 0.5541 | 0.8941 | 0.5687 | 0.4869 | 0.6336 | 0.6339 | 0.05 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |
| DINOv3 FCOS 5k | 0.3113 | 0.7564 | 0.1738 | 0.3103 | 0.4250 | 0.4368 | 0.60 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |

Matched FP/image budget around 0.425:

| Model | Selected conf | Recall@0.50 | Precision | FP/image |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned 5k | 0.05 | 0.9039 | 0.7650 | 0.3467 |
| DINOv3 FCOS 5k | 0.60 | 0.7784 | 0.7127 | 0.3917 |

Maximum recall over evaluated thresholds:

| Model | Threshold | Max recall@0.50 | FP/image |
| --- | ---: | ---: | ---: |
| YOLO11s fine-tuned 5k | 0.001 | 0.9519 | 3.1817 |
| DINOv3 FCOS 5k | 0.001 | 0.9159 | 75.6683 |

Comparison with previous 600-image custom metrics:

- YOLO11s fine-tuned at confidence 0.05:
  - Recall, precision, FP/image, and all visibility-level recalls matched the
    previous evaluator exactly.
- YOLO11s fine-tuned at confidence 0.025:
  - Recall, precision, FP/image, and all visibility-level recalls matched the
    previous evaluator exactly.
- DINOv3 FCOS 5k at confidence 0.60:
  - Recall, precision, FP/image, and all visibility-level recalls matched the
    previous evaluator exactly.

Output files:

- `outputs/evaluation_framework/val600_yolo11s_finetuned/summary.json`
- `outputs/evaluation_framework/val600_yolo11s_finetuned/coco_results.json`
- `outputs/evaluation_framework/val600_yolo11s_finetuned/confidence_sweep.csv`
- `outputs/evaluation_framework/val600_dinov3_fcos5k/summary.json`
- `outputs/evaluation_framework/val600_dinov3_fcos5k/coco_results.json`
- `outputs/evaluation_framework/val600_dinov3_fcos5k/confidence_sweep.csv`
- `outputs/evaluation_framework/val600_comparison/val600_side_by_side.csv`
- `outputs/evaluation_framework/val600_comparison/val600_side_by_side.json`

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src $LOGXY_ARTIFACT_ROOT/.venv-yolo/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 16 passed.

Limitations:

- These remain pilot results on a selected, visibility-stratified 600-image
  validation subset. They are not official ForestPersons benchmark scores.
- The official ForestPersons test split was not evaluated.
- Fine-tuned YOLO11s has been trained end-to-end on 5,000 ForestPersons images;
  DINOv3 FCOS uses a frozen DINOv3 backbone and only trains the lightweight
  detection head. The comparison is useful but the training regimes differ.
- DINOv3 timing here is cached-head-only. End-to-end DINOv3 timing must include
  feature extraction.

Recommended next action:

1. Download and verify the complete official ForestPersons validation images
   only, then run this same framework on all 18,243 validation images.
2. Use the full validation AP/AR and recall-by-visibility results to decide
   whether to scale YOLO11s training to the full 67,686-image training split
   before reproducing heavier MMDetection/detrex baselines.

## 2026-10-11 — Deformable R-CNN and DETR Baseline Feasibility

Audited the ForestPersons paper for the published detector baselines requested
by Davide and prepared lightweight adapters for future MMDetection/detrex
integration. No new dataset images were downloaded, no full training was
started, no existing YOLO11s or DINOv3 checkpoints were modified, and the
official test split was not evaluated.

Verified paper details:

- Deformable R-CNN:
  - Described as deformable Faster R-CNN with a ResNet-50-based backbone.
  - Implemented with MMDetection.
  - Training hyperparameters: SGD, learning rate 2e-2, batch size 16, weight
    decay 1e-4, 12 epochs.
  - Published ForestPersons RGB benchmark:
    - Validation: AP 65.0, AP50 94.7, AP75 78.5, AR 70.0.
    - Test: AP 66.3, AP50 93.4, AP75 77.5, AR 71.3.

- DETR-family:
  - DETR row:
    - Original DETR-style detector.
    - Implemented with MMDetection.
    - Training hyperparameters: AdamW, learning rate 1e-4, batch size 16,
      weight decay 1e-4, 150 epochs.
    - Published ForestPersons RGB benchmark:
      - Validation: AP 55.3, AP50 93.0, AP75 59.9, AR 68.0.
      - Test: AP 53.9, AP50 88.7, AP75 59.4, AR 67.9.
  - DINO row:
    - DETR-family DINO detector implemented with detrex.
    - This is not the same as our frozen DINOv3 ViT-S/16 feature extractor.
    - Training hyperparameters: AdamW, learning rate 1e-4, batch size 16,
      weight decay 1e-4, 12 epochs.
    - Published ForestPersons RGB benchmark:
      - Validation: AP 59.9, AP50 91.7, AP75 69.1, AR 70.1.
      - Test: AP 65.3, AP50 94.0, AP75 76.2, AR 77.7.

Checkpoint availability:

- No local ForestPersons-trained Deformable R-CNN, DETR, or detrex-DINO
  checkpoints were found.
- No public ForestPersons-trained checkpoints were verified during this pass.
- Any run with public COCO-pretrained weights must be labeled as generic
  pretrained inference, not as published ForestPersons baseline reproduction.

Local environment findings:

- `.venv-yolo` currently has Torch/Torchvision and pycocotools.
- `.venv-dinov3` has Torch/Torchvision, transformers, huggingface_hub, and
  pycocotools.
- Neither environment currently has MMDetection, MMCV, MMEngine, Detectron2,
  detrex, or RF-DETR-style packages.
- No DETR, Deformable R-CNN, or Faster R-CNN weights were found in local
  Hugging Face or Torch caches.

Apple Silicon feasibility:

- Deformable R-CNN is not a good candidate for faithful local MPS training or
  inference because MMDetection DCN/deformable convolution depends on compiled
  custom ops that are CUDA-oriented and fragile on macOS/MPS.
- Plain DETR is more likely to run locally for inference than Deformable R-CNN,
  but faithful ForestPersons training is still unrealistic on an M4 MacBook Pro
  with 16 GB unified memory. The paper trained DETR models on NVIDIA A100 and
  A6000 GPUs.
- detrex-DINO is heavier and should be treated as a CUDA-baseline task rather
  than a local Mac task.

Adapters created:

- `scripts/export_forestpersons_coco_subset.py`
  - Exports a manifest subset as one-class COCO annotations.
  - Maps all standard ForestPersons visibility/pose categories to one `person`
    class.
  - Excludes unknown/no-visibility annotations by default to match the current
    evaluation policy.
  - `--include-unknown-visibility` can be used later if raw annotation coverage
    is needed.
- `scripts/evaluate_external_coco_results.py`
  - Evaluates external COCO-format prediction JSON from MMDetection, detrex,
    Detectron2, or similar frameworks.
  - Reuses the verified COCOeval/local AP-AR path and custom visibility-aware
    metrics.

Adapter smoke tests:

- Exported the selected 600-image validation subset:
  - Output:
    `outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json`
  - Images: 600.
  - Evaluated annotations: 749.
  - Skipped unknown/no-visibility annotations: 1.
- Evaluated existing YOLO11s COCO-format predictions through the external
  prediction adapter:
  - Output:
    `outputs/baseline_feasibility/external_coco_eval_smoke_yolo11s/summary.json`
  - AP/AP50/AP75/AR and custom metrics matched the verified framework.

Report:

- `outputs/baseline_feasibility/README.md`

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src $LOGXY_ARTIFACT_ROOT/.venv-yolo/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 16 passed.

Recommended next action:

1. Use a CUDA-backed environment for the published Deformable R-CNN and
   DETR-family baselines.
2. First run generic COCO-pretrained smoke inference on 10-30 validation images
   with MMDetection DETR R50 and, if DCN ops build cleanly, MMDetection
   Deformable Faster R-CNN.
3. Export predictions as COCO result JSON and evaluate them with
   `scripts/evaluate_external_coco_results.py`.
4. Only after the framework smoke tests pass, train on the existing 5,000-image
   subset. Do not claim paper reproduction until ForestPersons-trained
   configurations and weights match the paper.

## 2026-10-11 — Simple DINOv3 Linear Patch Decoder

Implemented and evaluated a genuinely simple DINOv3 detection head for Phase 3
of Davide's specification. This experiment is separate from the existing FCOS
prototype, which was preserved and not retrained or modified.

Architecture:

- Backbone: frozen `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Input: existing cached DINOv3 ViT-S/16 patch features, short-side 672.
- Head: one shared `nn.Linear(384, 5)` applied independently to each patch.
- Outputs per patch:
  - 1 objectness logit.
  - 4 box parameters: center offset x/y plus log width/log height.
- Trainable parameters: 1,925.
- GT assignment: one center patch per ground-truth person, using the existing
  center-patch dense target builder.
- Losses:
  - Focal objectness loss.
  - Smooth L1 box loss on positive center patches only.
- Inference:
  - Existing `decode_predictions` path.
  - Confidence filtering.
  - NMS IoU 0.50.
  - Separate `scale_x` and `scale_y` conversion back to original image
    coordinates.

Difference from FCOS:

- Linear patch head:
  - Independent per-patch prediction.
  - No convolutional context.
  - No multi-positive assignment.
  - No l/t/r/b distance regression.
  - No centerness branch.
  - 1,925 trainable parameters.
- Existing FCOS head:
  - Uses convolutional neighboring-patch context.
  - Uses multi-positive patch assignment inside/near each GT box.
  - Predicts box distances plus centerness.
  - Has substantially more trainable parameters and stronger localization
    capacity.

Files changed:

- `src/forestpersons/dinov3_detector.py`
  - Added `DINOv3LinearPatchHead`.
- `scripts/train_dinov3_detector_head.py`
  - Added `--head linear`.
  - Added per-epoch `latest.pt` and `train_history.json` checkpoint saving.
- `scripts/evaluate_dinov3_detector.py`
  - Added linear-head loading support.
- `scripts/evaluate_forestpersons_framework.py`
  - Added linear-head loading support.
- `tests/test_dinov3_linear_head.py`
  - Added focused tests for output shape, parameter count, dense target scaling,
    decode coordinate conversion, and low-confidence filtering.

Training:

- Train subset: existing 5,000 ForestPersons training images.
- Validation subset: existing 600 ForestPersons validation images.
- Feature cache reused:
  - `outputs/dinov3_detector/features/train_5000_seed20261012_short672`
  - `outputs/dinov3_detector/features/val_600_short672`
- Epochs: 20.
- Learning rate: 0.0005.
- Weight decay: 0.0001.
- Box loss weight: 5.0.
- Seed: 20261013.
- Device: CPU.
- Training time: 1,205 seconds, about 20.1 minutes.
- Positive cells per epoch: 6,429.
- Assignment collisions: 2.

Note:

- An earlier 20-epoch run was interrupted after epoch 17 because the original
  trainer only saved at the end. The trainer was then fixed to checkpoint after
  each epoch, and the completed rerun below is the reported experiment.

Artifacts:

- Checkpoint:
  `outputs/dinov3_detector/head_linear_patch_train5000_seed20261013_rerun/dinov3_head.pt`
- Latest per-epoch checkpoint:
  `outputs/dinov3_detector/head_linear_patch_train5000_seed20261013_rerun/latest.pt`
- Training history:
  `outputs/dinov3_detector/head_linear_patch_train5000_seed20261013_rerun/train_history.json`
- Experiment README:
  `outputs/dinov3_detector/head_linear_patch_train5000_seed20261013_rerun/README.md`
- Evaluation summary:
  `outputs/evaluation_framework/val600_dinov3_linear_patch5k/summary.json`
- COCO predictions:
  `outputs/evaluation_framework/val600_dinov3_linear_patch5k/coco_results.json`
- Confidence sweep:
  `outputs/evaluation_framework/val600_dinov3_linear_patch5k/confidence_sweep.csv`
- Comparison table:
  `outputs/evaluation_framework/val600_dinov3_linear_patch5k/comparison/comparison.csv`

Validation results on the selected 600-image subset:

| Model | Params | AP | AP50 | AP75 | AR@100 | Conf | Recall@0.50 | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned 5k | n/a | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.05 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |
| DINOv3 FCOS 5k | n/a | 0.3113 | 0.7564 | 0.1738 | 0.4368 | 0.60 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| DINOv3 linear patch 5k | 1,925 | 0.0614 | 0.2615 | 0.0050 | 0.2889 | 0.30 | 0.1041 | 0.3939 | 0.2000 | 0.0460 | 0.1294 | 0.0984 | 0.1368 |

Maximum recall over evaluated thresholds:

- DINOv3 linear patch head:
  - Threshold: 0.001.
  - Recall@IoU 0.50: 0.8451.
  - FP/image: 148.53.

Interpretation:

- The simple linear decoder is a useful lower-bound experiment but is not
  competitive.
- It can reach high recall only by accepting an unusable number of false
  positives.
- At practical FP/image budgets, recall collapses, especially at 20%
  visibility.
- The gap to FCOS strongly suggests that neighboring-patch context,
  multi-positive assignment, and centerness/objectness calibration are important
  for ForestPersons.

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 20 passed.

Recommended next action:

1. Do not invest further in independent per-patch linear decoding as a detector.
2. For the next DINOv3 phase, try a still-light but slightly more structured
   decoder, such as a 1x1 linear projection plus one depthwise 3x3 context
   layer, while keeping the parameter count small and comparing directly
   against FCOS.
3. If the priority is published baselines, switch back to CUDA-backed
   MMDetection/detrex smoke tests before adding more DINOv3 variants.

## 2026-10-11 — DINOv3 patch-kNN anomaly scoring pilot

Goal:

- Test Davide's patch-kNN idea using frozen DINOv3 ViT-S/16 patch features as an
  anomaly-localization signal.
- Measure whether person-overlapping patches are farther from normal forest
  background patches than non-person patches.
- Keep this separate from bounding-box detection unless a proposal method is
  explicitly defined.

Implementation:

- Added `src/forestpersons/patch_knn.py`.
- Added `scripts/run_dinov3_patch_knn_anomaly.py`.
- Added `tests/test_patch_knn.py`.
- Reused cached DINOv3 features:
  - `outputs/dinov3_detector/features/train_5000_seed20261012_short672`
  - `outputs/dinov3_detector/features/val_600_short672`
- Built a background reference bank from training-image patches that do not
  overlap person bounding boxes.
- Applied a 1-patch safety margin around boxes before selecting background
  patches to reduce person contamination.
- Used normalized patch features and cosine nearest-neighbor distance:
  `score = 1 - max_cosine_similarity(background_bank, patch_feature)`.
- Used CPU-friendly batched matrix multiplication, with no FAISS dependency.

Reference bank:

- Source: existing 5,000-image training subset.
- Sampled training images: 500.
- Bank size: 5,000 background patches.
- Feature dimension: 384.
- Candidate background patches considered: 766,639.
- Safety margin: 1 DINO patch, equal to 16 resized pixels.
- Estimated bank memory: 7.32 MiB as float32.
- Seed: 20261014.

Pilot runtime estimate:

- A 50-image validation pilot completed in 7.98 seconds.
- This indicated the full 600-image validation run would be comfortably below
  the 2-hour limit.

Full selected-val600 results:

- Validation images: 600.
- Validation patches: 1,917,636.
- Person-overlapping patches: 41,639.
- Background patches: 1,875,997.
- Runtime: 26.79 seconds total, 0.045 seconds/image.

| Group | Person patches | Background patches | AUROC | AP |
| --- | ---: | ---: | ---: | ---: |
| Overall | 41,639 | 1,875,997 | 0.8037 | 0.1338 |
| Visibility 20 | 7,357 | 1,875,997 | 0.7770 | 0.0216 |
| Visibility 40 | 9,151 | 1,875,997 | 0.7941 | 0.0339 |
| Visibility 70 | 11,544 | 1,875,997 | 0.8036 | 0.0441 |
| Visibility 100 | 13,587 | 1,875,997 | 0.8248 | 0.0583 |

Interpretation:

- Patch-kNN has real patch-level signal: person-overlapping patches usually
  score more anomalous than background.
- The signal improves with visibility, from AUROC 0.7770 at 20% visibility to
  AUROC 0.8248 at 100% visibility.
- AP is low because positive person patches are sparse and many unusual
  non-person regions also score highly.
- Qualitative heatmaps show that high anomaly scores can correspond to people,
  but also to unusual vegetation, shadows, blur, edges, and bright objects.
- This is an exploratory anomaly cue, not a detector. It should not be compared
  to YOLO11s or FCOS as a bounding-box detector without a defined proposal
  generation step.

Artifacts:

- Experiment README:
  `outputs/dinov3_patch_knn/val600_bank5000_seed20261014/README.md`
- Summary:
  `outputs/dinov3_patch_knn/val600_bank5000_seed20261014/summary.json`
- Per-image metrics:
  `outputs/dinov3_patch_knn/val600_bank5000_seed20261014/image_metrics.csv`
- Reference bank:
  `outputs/dinov3_patch_knn/val600_bank5000_seed20261014/reference_bank.pt`
- Heatmaps:
  `outputs/dinov3_patch_knn/val600_bank5000_seed20261014/heatmaps/`

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 25 passed.

Limitations:

- Patch labels are derived from overlap with resized GT boxes; this evaluates
  localization signal but not box detection quality.
- Visibility-specific metrics compare each visibility group's positives against
  the shared background-patch pool, so they should be read as discrimination
  diagnostics rather than independent detection benchmarks.
- Heatmaps are qualitative and not yet stratified into a curated visibility
  gallery.
- The current implementation uses exact batched comparison against a small
  bank. Larger banks may improve coverage but will increase runtime and memory.

Recommended next action:

1. Keep patch-kNN as a diagnostic/anomaly baseline.
2. If Davide wants an anomaly-detector comparison, define a proposal extraction
   method from heatmaps and evaluate those proposals with the existing COCO
   framework.
3. Otherwise, prioritize published detector baselines or full-dataset scaling of
   the strongest supervised models, because patch-kNN alone is not yet a
   person detector.

## 2026-10-11 — CUDA preparation for published Deformable R-CNN and DETR baselines

Goal:

- Prepare reproducible CUDA-side experiments for ForestPersons published
  detection baselines without installing CUDA/MMDetection locally and without
  claiming unrun results.
- Target Deformable Faster R-CNN R50-FPN with deformable convolutions and
  original DETR R50.

Verified context:

- Existing paper/audit findings say Deformable R-CNN is a Deformable Faster
  R-CNN style detector with a ResNet-50 backbone.
- DETR is the original DETR-family detector.
- The paper states most baselines except YOLOv11, DINO, and CZDet used
  MMDetection; paper DINO is detrex DINO, not DINOv3.
- No ForestPersons-trained Deformable R-CNN or DETR checkpoints were verified.
- Therefore, prepared checkpoint sources are COCO-pretrained only and must not
  be described as reproduced ForestPersons baselines.

Official implementation choices:

- Use MMDetection 2.x for both models:
  - `mmdetection==2.28.2`
  - `mmcv-full==1.7.1`
  - `torch==1.13.1`
  - `torchvision==0.14.1`
  - CUDA 11.7
- This version line was selected because it supports the classic MMDetection
  configs and DCN CUDA/MMCV operators. It is not confirmed as the exact paper
  environment.

Official COCO-pretrained sources:

- Deformable Faster R-CNN R50-FPN dconv c3-c5:
  - Config:
    `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`
  - Checkpoint:
    `https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth`
- DETR R50:
  - Config:
    `configs/detr/detr_r50_8x2_150e_coco.py`
  - Checkpoint:
    `https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth`

Dataset export update:

- Fixed `scripts/export_forestpersons_coco_subset.py` to prefer manifest
  `image_id`/`id` when available.
- Reason: the official train annotation JSON contains duplicate `file_name`
  values for some images, so selecting by filename alone can over-select.
- Regenerated one-class exports:
  - `outputs/baseline_feasibility/forestpersons_train5000_oneclass_coco.json`
    - 5,000 images.
    - 6,415 annotations.
    - Category id 1, `person`.
  - `outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json`
    - 600 images.
    - 749 evaluated annotations.
    - Category id 1, `person`.
    - 1 unknown-visibility annotation skipped.
- Export checks:
  - Unique image ids within train5000: 5,000/5,000.
  - Unique image ids within val600: 600/600.
  - Train/val filename overlap: 0.
  - Train/val numeric image id overlap: 45. This appears to be split-local id
    reuse across official JSON files, not image leakage, because filenames do
    not overlap.
  - All annotation image ids are present in their exported split.
  - All exported annotation category ids are 1.
  - No non-positive exported boxes found.

Files created:

- `experiments/published_baselines_cuda/README.md`
- `experiments/published_baselines_cuda/environment_cuda_mmdet2.yml`
- `experiments/published_baselines_cuda/configs/forestpersons_deformable_faster_rcnn_r50_fpn_dconv_c3-c5_1x_train5000.py`
- `experiments/published_baselines_cuda/configs/forestpersons_detr_r50_8x2_150e_train5000.py`
- `scripts/verify_mmdet_cuda_env.py`
- `scripts/mmdet_infer_forestpersons.py`
- `tests/test_coco_subset_export.py`
- `tests/test_published_baseline_cuda_prep.py`

Prepared commands:

- CUDA environment verification:
  `python scripts/verify_mmdet_cuda_env.py`
- Generic COCO-pretrained smoke inference for 10-30 validation images using
  official MMDetection configs and checkpoint paths.
- External evaluation via:
  `scripts/evaluate_external_coco_results.py`
- Fine-tuning commands for train5000 were documented but not executed.

Estimated requirements:

- Deformable Faster R-CNN smoke inference:
  - Approximately 6-10 GB VRAM for batch 1.
  - Requires MMCV DCN CUDA ops.
- Deformable Faster R-CNN train5000 fine-tuning:
  - Approximately 12-16 GB VRAM for small batch; 24 GB preferred.
  - Estimated 1-4 hours on a single RTX 3090/A5000-class GPU.
- DETR R50 smoke inference:
  - Approximately 8-12 GB VRAM for batch 1.
- DETR R50 train5000 fine-tuning:
  - Approximately 16-24 GB VRAM minimum for small batch; 24-48 GB preferred.
  - 150 epochs likely requires many hours to more than a day on a single GPU.

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 31 passed.

Remaining blockers:

- No CUDA machine has been used yet for these baselines.
- No checkpoints were downloaded in this step.
- MMDetection/MMCV custom ops were not installed or validated locally.
- Exact ForestPersons paper configs and trained checkpoints remain unavailable
  in the verified materials.
- Generic COCO-pretrained inference will be a smoke test only, not a published
  ForestPersons baseline reproduction.

Recommended next action:

1. Move or sync the project to a CUDA Linux machine.
2. Create the documented `logxy-mmdet2` environment and run
   `scripts/verify_mmdet_cuda_env.py`.
3. After approval to download the two COCO checkpoints, run 30-image smoke
   inference for DETR first, then Deformable Faster R-CNN.
4. Evaluate the smoke predictions with `scripts/evaluate_external_coco_results.py`.
5. Only after both smoke tests pass, consider train5000 fine-tuning.

## 2026-10-11 — RX and EfficientAD anomaly-baseline pass

Goal:

- Address Davide's Phase 2 anomaly-baseline requirement after the DINOv3
  patch-kNN experiment.
- Implement a simple classical RX anomaly baseline locally.
- Assess EfficientAD feasibility without downloading new data, installing CUDA,
  or claiming unrun results.

Method review:

- RX / Reed-Xiaoli detector:
  - Classical anomaly detector originally developed for multi-band optical /
    hyperspectral data.
  - Uses a background mean and covariance, then scores each observation with
    squared Mahalanobis distance:
    `(x - mu)^T Sigma^-1 (x - mu)`.
  - Adapted here to RGB UAV imagery by treating local RGB patch statistics as
    the feature vector.
- EfficientAD:
  - WACV 2024 method for visual anomaly detection.
  - Uses a lightweight feature extractor, student-teacher discrepancy for local
    anomalies, and an autoencoder branch for global/logical anomalies.
  - Common public path is an Anomalib implementation based on community
    EfficientAD reproductions.
  - No verified ForestPersons-trained EfficientAD weights are available.

RX implementation:

- Added `src/forestpersons/rx_anomaly.py`.
- Added `scripts/run_rx_anomaly_baseline.py`.
- Added `tests/test_rx_anomaly.py`.
- Feature representation:
  - Per 16 x 16 resized patch.
  - `[mean_r, mean_g, mean_b, std_r, std_g, std_b]` in RGB `[0, 1]`.
- Patch grid:
  - Short side 672.
  - Patch size 16.
  - Same patch-label definition as DINOv3 patch-kNN.
- Background fitting:
  - Train data only: existing train5000 subset.
  - Sampled training images: 500.
  - Background patches: 50,000.
  - Person regions excluded with a 1-patch safety margin.
  - Covariance regularization:
    `cov + I * 0.001 * mean(diag(cov))`.
- Validation:
  - Existing selected val600 subset.
  - Patch-level AUROC/AP.
  - Visibility-specific positive patches compared against the shared background
    patch pool, matching the patch-kNN diagnostic protocol.

Runtime estimate and execution:

- 50-image pilot completed in 25.26 seconds including background fitting.
- Full val600 run was therefore below the 2-hour threshold and was executed.
- Full runtime: 55.25 seconds including background fitting.

RX selected-val600 results:

| Group | Person patches | Background patches | AUROC | AP |
| --- | ---: | ---: | ---: | ---: |
| Overall | 41,639 | 1,875,997 | 0.5787 | 0.0314 |
| Visibility 20 | 7,357 | 1,875,997 | 0.5499 | 0.0047 |
| Visibility 40 | 9,151 | 1,875,997 | 0.5871 | 0.0073 |
| Visibility 70 | 11,544 | 1,875,997 | 0.5723 | 0.0089 |
| Visibility 100 | 13,587 | 1,875,997 | 0.5942 | 0.0118 |

Comparison with DINOv3 patch-kNN:

| Method | Feature dim | Background/reference patches | Overall AUROC | Overall AP |
| --- | ---: | ---: | ---: | ---: |
| RX RGB mean/std | 6 | 50,000 | 0.5787 | 0.0314 |
| DINOv3 patch-kNN | 384 | 5,000 | 0.8037 | 0.1338 |

Interpretation:

- RX is feasible and very lightweight on Apple Silicon CPU.
- RX is much weaker than DINOv3 patch-kNN because raw RGB patch statistics
  mostly measure color/illumination rarity rather than semantic person cues.
- High RX scores often correspond to shadows, roads, bright vegetation, seasonal
  color, or other low-level visual outliers.
- RX should be reported as a classical anomaly baseline, not a detector.

EfficientAD feasibility:

- Feasible in principle on Apple Silicon for small pilots, but not implemented
  in this step.
- Reasons:
  - EfficientAD requires training a student and autoencoder on normal data; it
    is not a zero-training detector like RX.
  - ForestPersons does not provide a clean normal-only forest training split, so
    we would need a carefully constructed background/normal protocol or masked
    training strategy.
  - Public implementations would introduce a larger dependency stack, likely
    Anomalib/PyTorch Lightning, and may assume industrial anomaly datasets.
  - No verified ForestPersons EfficientAD checkpoint exists.
  - Runtime/memory on M4 16 GB is uncertain until a dedicated small pilot is
    scoped.
- No EfficientAD metrics were produced, and no EfficientAD result is claimed.

Artifacts:

- RX summary:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/summary.json`
- RX per-image metrics:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/image_metrics.csv`
- RX model:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/rx_model.json`
- RX background features:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/background_features.npy`
- RX heatmaps:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/heatmaps/`
- RX experiment README:
  `outputs/rx_anomaly/val600_rgb6_bg50000_seed20261015/README.md`

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 34 passed.

Recommended next action:

1. Keep RX as the classical anomaly baseline.
2. Do not compare RX directly to YOLO/FCOS as a detector unless we define a
   heatmap-to-proposal method.
3. If EfficientAD is still required, prepare a small explicit pilot:
   200-500 normal/background-oriented train images, masked person regions,
   Anomalib EfficientAD in a separate environment, and evaluation with the same
   patch-level protocol before considering full train5000.

## 2026-10-11 — Google Colab workflow prepared for MMDetection baselines

Goal:

- Prepare a reliable Google Colab Free/T4 execution workflow for published
  baseline smoke tests:
  - DETR R50.
  - Deformable Faster R-CNN R50-FPN with DCN.
- Do not run GPU inference locally and do not start fine-tuning.

Compatibility decision:

- Kept the MMDetection 2.x architecture path because it directly supports the
  official model-zoo configs selected for the paper-style baselines:
  - `configs/detr/detr_r50_8x2_150e_coco.py`
  - `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`
- Avoided modifying Colab's default Python/PyTorch runtime in-place.
- Added an isolated micromamba workflow using:
  - Python 3.10.
  - PyTorch 1.13.1.
  - Torchvision 0.14.1.
  - CUDA runtime 11.7.
  - MMCV-full 1.7.1.
  - MMDetection v2.28.2.
- This is designed to be more robust against current Colab Python version
  changes than attempting to install old MMDetection 2.x directly into the base
  notebook environment.

Files created:

- `experiments/published_baselines_cuda/colab/LogXY_ForestPersons_MMdetection_Smoke.ipynb`
- `experiments/published_baselines_cuda/colab/setup_mmdet2_colab.sh`
- `experiments/published_baselines_cuda/colab/download_checkpoints_colab.sh`
- `experiments/published_baselines_cuda/colab/run_smoke30_colab.sh`

Files modified:

- `experiments/published_baselines_cuda/README.md`
  - Added Colab-specific setup/run/upload instructions and architecture support
    notes.
- `scripts/mmdet_infer_forestpersons.py`
  - Kept MMDetection 2.x as the target.
  - Added defensive support for MMDetection 3-style `DetDataSample`
    predictions to make accidental version mismatches easier to diagnose.
- `tests/test_published_baseline_cuda_prep.py`
  - Added coverage for the MMDetection 3-style prediction shape adapter.

Colab workflow summary:

1. Mount Google Drive.
2. Use `/content/drive/MyDrive/LogXY` as persistent project root.
3. Verify GPU with `nvidia-smi`.
4. Run `setup_mmdet2_colab.sh` to create the isolated `logxy-mmdet2`
   micromamba environment and clone MMDetection v2.28.2.
5. Run `scripts/verify_mmdet_cuda_env.py` inside that environment.
6. Download COCO-pretrained DETR and DCN Faster R-CNN checkpoints into Drive.
7. Run `run_smoke30_colab.sh` to produce COCO-format prediction JSON for the
   first 30 val600 images and evaluate with our existing external-results
   adapter.

Architecture support status:

- DETR R50:
  - Supported through official MMDetection 2.x DETR R50 COCO config/checkpoint.
  - Smoke test remains COCO-pretrained inference, not ForestPersons training.
- Deformable Faster R-CNN R50-FPN/DCN:
  - Supported through official MMDetection 2.x DCN Faster R-CNN config/checkpoint.
  - DCN is preserved; no plain Faster R-CNN substitution was made.
  - Requires `mmcv-full` CUDA ops to install/load successfully on Colab.

Not executed:

- No Colab runtime was started from this machine.
- No checkpoint was downloaded locally.
- No GPU inference, evaluation, or training was run for DETR/DCN.

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 35 passed.

Remaining blockers:

- Need to run the notebook on Colab Free/T4.
- Need live confirmation that `mmcv-full==1.7.1` installs successfully in the
  isolated Colab environment.
- Google Drive must contain the LogXY project, required val600 images, COCO
  export, scripts, and source package.

Recommended next action:

1. Upload/sync the full `LogXY` project folder to Google Drive at
   `/content/drive/MyDrive/LogXY`.
2. Open the prepared Colab notebook and run cells through environment
   verification.
3. If verification passes, download checkpoints and execute the 30-image smoke
   test.

## 2026-10-11 — Colab smoke-test handoff bundle prepared

Goal:

- Prepare the existing Colab workflow for the first real Google Colab Free/T4
  smoke test.
- Avoid uploading unnecessary local artifacts such as virtual environments,
  cached DINOv3 features, old outputs, or the full ForestPersons dataset.

Fixes and validation:

- Created a dedicated smoke manifest:
  `outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json`.
- Verified all first 30 images referenced by
  `forestpersons_val600_oneclass_coco.json` exist locally.
- Updated `run_smoke30_colab.sh` so smoke evaluation uses the same 30-image
  manifest as smoke inference, instead of evaluating 30-image predictions
  against all 600 validation images.
- Updated the Colab notebook path-check cell to verify:
  - `forestpersons_val600_oneclass_coco.json`
  - `forestpersons_val600_smoke30_manifest.json`
  - `data/annotations/val.json`
  - required scripts/source files
  - the first 30 image files
- Added explicit `$HOME/.local/bin` PATH handling for micromamba across Colab
  scripts and notebook verification.
- Added `scripts/prepare_colab_smoke_bundle.py`.

Prepared upload artifacts:

- Folder:
  `outputs/colab_upload/LogXY`
  - Size: 37 MB.
  - Files: 56.
  - JPEGs: 30.
- Zip:
  `outputs/colab_upload/LogXY_colab_smoke30.zip`
  - Size: about 12 MB.

The upload bundle is intended to become:

- Google Drive folder: `/content/drive/MyDrive/LogXY`

It contains only the files needed for the 30-image smoke test:

- required source package files under `src/forestpersons/`
- required scripts
- Colab notebook and shell scripts
- val annotation JSON
- val600 COCO export
- smoke30 manifest
- first 30 ForestPersons validation images

Not executed:

- No Colab GPU inference.
- No checkpoint download.
- No fine-tuning.
- No completed detector/anomaly outputs were modified.

Automated tests:

- Command:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
- Result: 36 passed.

Next manual step:

- Upload/extract the prepared bundle to Google Drive, open the notebook, run
  through the `nvidia-smi` GPU verification cell, and report that output before
  running the environment installation.

## 2026-10-12 — Fixed Colab micromamba strict-Bash setup failure

Context:

- Google Drive mount succeeded in Colab.
- The 30 smoke-test images were verified.
- Colab allocated an NVIDIA Tesla T4 with about 15 GB VRAM.
- The MMDetection setup cell failed before environment installation completed:
  `setup_mmdet2_colab.sh: line 112: MAMBA_ROOT_PREFIX: unbound variable`.

Fix:

- Updated `experiments/published_baselines_cuda/colab/setup_mmdet2_colab.sh`.
- Explicitly exports:
  - `MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"`
  - `MAMBA_NO_BANNER=1`
- Added a `find_micromamba` helper so reruns can find micromamba after a
  partial install in:
  - `$PATH`
  - `$HOME/.local/bin/micromamba`
  - `$HOME/bin/micromamba`
  - `/usr/local/bin/micromamba`
- Uses the resolved micromamba binary for all setup commands.
- Prints the micromamba path and root prefix for debugging.
- Leaves the MMDetection 2.x dependency choices unchanged:
  - Python 3.10
  - PyTorch 1.13.1
  - Torchvision 0.14.1
  - CUDA runtime 11.7
  - MMCV-full 1.7.1
  - MMDetection v2.28.2

Related updates:

- Updated `run_smoke30_colab.sh` to export the same `MAMBA_ROOT_PREFIX`.
- Updated the notebook verification cell to export `MAMBA_ROOT_PREFIX` before
  calling `micromamba run`.
- Regenerated the minimal Drive upload bundle:
  - Folder: `outputs/colab_upload/LogXY`
  - Zip: `outputs/colab_upload/LogXY_colab_smoke30.zip`

Validation:

- Shell syntax checks:
  - `bash -n experiments/published_baselines_cuda/colab/download_checkpoints_colab.sh`
  - `bash -n experiments/published_baselines_cuda/colab/run_smoke30_colab.sh`
  - `bash -n experiments/published_baselines_cuda/colab/setup_mmdet2_colab.sh`
  - Result: passed.
- Notebook JSON validation:
  - Result: passed.
- Targeted CUDA-prep tests:
  - `tests/test_published_baseline_cuda_prep.py`
  - Result: 6 passed.
- Full local test suite:
  - Command:
    `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest $LOGXY_ARTIFACT_ROOT/tests -q`
  - Result: 36 passed.

Next manual step:

- Sync the corrected `setup_mmdet2_colab.sh` and notebook to Google Drive, then
  rerun only the MMDetection setup cell.

## 2026-10-12 — Repository audit and experiment registry added

Context:

- The project had grown from Phase 1 ForestPersons preparation into multiple
  pilot experiments: YOLO11, DINOv3 detector heads, DINOv3 patch-kNN, RX anomaly
  scoring, and CUDA/Colab preparation for DETR and Deformable Faster R-CNN.
- The user requested a cleanup-oriented audit before starting new experiments.

Completed:

- Added a read-only repository audit:
  - `docs/audit/repository_audit_2026-10-12.md`
- Added an experiment registry:
  - `docs/experiment_registry/README.md`
  - `docs/experiment_registry/experiments.csv`
- Updated `README.md` and `AGENTS.md` so future sessions start from the audit
  and registry instead of treating Phase 1 as the only active project state.

Important preservation note:

- No datasets, checkpoints, cached DINOv3 features, metrics, experiment outputs,
  or scripts were moved, renamed, deleted, downloaded, trained, or evaluated in
  this cleanup step.

Current organization status:

- Dataset preparation: completed for annotation validation, train5000, and
  val600 pilot subsets.
- YOLO11 baselines: completed for pretrained and train5000 fine-tuned pilot
  evaluations.
- DINOv3 experiments: completed for FCOS prototypes, linear patch head, and
  patch-kNN anomaly scoring on pilot subsets.
- RX anomaly baseline: completed on val600 patch-level protocol.
- DETR and Deformable Faster R-CNN: CUDA/Colab workflows prepared, but no
  verified GPU inference or training result yet.
- EfficientAD: feasibility only.
- Fusion, LoRA, full-dataset training, and edge deployment: not started.

Recommended next cleanup step:

- Add lightweight README files to high-risk folders such as `data/`, `outputs/`,
  `outputs/dinov3_detector/`, `outputs/colab_upload/`, and `scripts/` before
  moving any artifacts.

## 2026-10-12 — DINOv3 LoRA + FCOS infrastructure implemented

Context:

- The project resumed Davide's Phase 3 ForestPersons DINOv3 work after
  repository cleanup.
- Objective: prepare a controlled comparison between the existing frozen DINOv3
  + FCOS-style detection head and a LoRA-adapted DINOv3 backbone using the same
  FCOS head.
- This step implemented and validated infrastructure only. It did not start the
  train400/train500 pilot or full train5000 training.

Implemented:

- Added custom LoRA utilities:
  - `src/forestpersons/dinov3_lora.py`
- Added online image-based LoRA + FCOS training/smoke script:
  - `scripts/train_dinov3_lora_fcos.py`
- Extended the verified evaluation framework with a `dinov3_lora` detector mode:
  - `scripts/evaluate_forestpersons_framework.py`
- Added focused tests:
  - `tests/test_dinov3_lora.py`
- Updated the experiment registry to record this as infrastructure/smoke work,
  not a scientific pilot result.

Verified DINOv3 attention module names:

- Actual model class:
  `transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTModel`
- Linear attention projections appear under:
  - `model.layer.N.attention.k_proj`
  - `model.layer.N.attention.v_proj`
  - `model.layer.N.attention.q_proj`
  - `model.layer.N.attention.o_proj`
- Initial LoRA targets:
  - `attention.q_proj`
  - `attention.v_proj`
- Applied targets: q/v projections in all 12 transformer layers, 24 LoRA-wrapped
  Linear modules total.

Parameter status:

- Original DINOv3 backbone parameters remain frozen.
- Trainable LoRA parameters: 73,728.
- Trainable FCOS head parameters: 524,646.
- Total trainable parameters for LoRA + FCOS: 598,374.
- Total parameters in loaded backbone + head: 22,194,918.

Smoke validation:

- Command: 10 training images, 1 pass, batch size 1, gradient accumulation 2,
  MPS device, short side 672.
- Output directory:
  `outputs/dinov3_lora/smoke10_infrastructure_seed20261012/`
- Result:
  - Images: 10.
  - Optimizer steps: 5.
  - Loss: 1.36844.
  - Classification loss: 0.02038.
  - Box loss: 0.32496.
  - Centerness loss: 0.69814.
  - Positive cells: 64.
  - LoRA gradients: 48 tensors with gradients, all nonzero.
  - FCOS head gradients: 18 tensors with gradients, all nonzero.
  - Runtime for the 10-image pass: 18.44 seconds.
  - MPS current allocated memory after epoch: about 104 MB.
  - MPS driver allocated memory: about 11.29 GB.
- Saved smoke checkpoint:
  `outputs/dinov3_lora/smoke10_infrastructure_seed20261012/dinov3_lora_fcos.pt`

Evaluation-framework compatibility smoke:

- Ran `dinov3_lora` evaluation mode on 2 validation images using the smoke
  checkpoint.
- Output directory:
  `outputs/dinov3_lora/smoke10_infrastructure_seed20261012/eval_val2_framework_smoke/`
- Result: completed and wrote `summary.json`, `coco_results.json`, and
  `confidence_sweep.csv`.
- The metrics are not meaningful because the checkpoint is only a 10-image
  infrastructure smoke checkpoint.

Tests:

- New LoRA-focused tests:
  - `tests/test_dinov3_lora.py`
  - Result before full suite: 7 passed.

Limitations:

- This is not a trained LoRA detector and should not be compared with YOLO11s or
  frozen-DINOv3 FCOS metrics.
- Existing frozen feature caches cannot be reused for LoRA optimization because
  LoRA changes the backbone outputs.
- Current LoRA path uses online image forwarding and is therefore much slower
  than cached-head-only training/evaluation.
- Full pilot training still needs an approved train400/train500 run.

## 2026-10-12 — ForestPersons Phase 1-2 consolidation report

Context:

- The project deadline is Friday, October 16, 2026.
- The current scope is ForestPersons only.
- Published detector baselines should be used as literature references rather
  than retrained this week.
- DINOv3 + LoRA infrastructure exists, but no real LoRA pilot training was
  started in this task.

Completed:

- Created:
  `docs/reports/forestpersons_phase1_phase2_2026-10-12.md`
- Consolidated ForestPersons dataset evidence:
  - Official train/val/test split sizes.
  - Category mapping from pose/visibility classes to one `person` class.
  - train5000 and val600 subset distributions.
  - Visibility-level annotation counts.
  - train/validation leakage check from existing YOLO conversion summary.
  - Evaluation protocol and limitations.
- Verified published ForestPersons benchmark values from the paper's Table 3 for:
  - YOLOv11.
  - DETR.
  - Deformable R-CNN.
  - Faster R-CNN.
  - DINO object detector, explicitly not DINOv3.
- Consolidated existing local pilot results for:
  - Fine-tuned YOLO11s on train5000/val600.
  - Frozen DINOv3 + FCOS.
  - Frozen DINOv3 + linear patch head.
  - DINOv3 patch-kNN anomaly scoring.
  - RX Mahalanobis anomaly scoring.
- Marked EfficientAD as not executed.

No training, Colab setup, dataset download, model download, or repository
reorganization was performed.

Status:

- Phase 1 can reasonably be considered complete for the ForestPersons-only pilot
  scope.
- Phase 2 can reasonably be considered complete for the ForestPersons-only
  scope, with published baselines documented as literature references and our
  own YOLO11s pilot retained for internal comparison.

Next approved research task:

- Run the first real ForestPersons DINOv3 LoRA + FCOS pilot training experiment.

## 2026-10-12 — First real DINOv3 LoRA + FCOS train400 pilot

Context:

- Phase 3 ForestPersons DINOv3 work began after Phase 1-2 consolidation.
- Goal: run a controlled LoRA pilot without starting full train5000 training.
- Existing datasets, frozen-feature caches, frozen DINOv3 checkpoints, YOLO
  checkpoints, and previous evaluation outputs were preserved.

Training configuration:

- Script: `scripts/train_dinov3_lora_fcos.py`.
- Train subset:
  `outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json`.
- Evaluation subset:
  `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`.
- Output directory:
  `outputs/dinov3_lora/train400_seed20261012_r4_qv_fcosinit_e2/`.
- Device: MPS.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261012.
- LoRA targets:
  - `attention.q_proj`
  - `attention.v_proj`
  - all 12 DINOv3 transformer layers.
- LoRA rank: 4.
- LoRA alpha: 16.
- LoRA dropout: 0.05.
- Original DINOv3 backbone weights frozen.
- Trainable parameters:
  - LoRA: 73,728.
  - FCOS head: 524,646.
  - Total trainable: 598,374.

Head initialization:

- The FCOS head was explicitly initialized from the existing frozen-DINOv3 FCOS
  train5000 checkpoint:
  `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`.
- The checkpoint loaded successfully with `head: fcos`.
- This pilot is therefore not a from-scratch head comparison. It tests whether a
  small LoRA adaptation can improve the already trained FCOS head.

Training results:

| Epoch | Images | Optimizer steps | Loss | Classification | Box | Centerness | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 400 | 100 | 0.63269 | 0.000738 | 0.018690 | 0.594573 | 456.19 |
| 2 | 400 | 200 | 0.61959 | 0.000679 | 0.013699 | 0.591510 | 476.46 |

Gradient status:

- Epoch 1:
  - LoRA tensors with nonzero gradients: 48/48.
  - FCOS head tensors with nonzero gradients: 18/18.
- Epoch 2:
  - LoRA tensors with nonzero gradients: 48/48.
  - FCOS head tensors with nonzero gradients: 18/18.

Runtime and memory:

- Total training runtime: 932.83 seconds, about 15.55 minutes.
- Peak observed MPS driver allocation from epoch summaries:
  11,297,947,648 bytes, about 10.52 GiB.
- No memory failure or MPS warning was reported.

Evaluation:

- Script: `scripts/evaluate_forestpersons_framework.py`.
- Detector mode: `dinov3_lora`.
- Output directory:
  `outputs/dinov3_lora/train400_seed20261012_r4_qv_fcosinit_e2/eval_val600/`.
- Evaluation uses online DINOv3 + LoRA feature extraction, not cached frozen
  features.
- End-to-end validation runtime:
  - 263.38 seconds for 600 images.
  - 0.43896 seconds/image.
  - 2.28 images/second.

Val600 COCO metrics:

| Model | AP | AP50 | AP75 | AR@100 |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + FCOS train5000 | 0.3113 | 0.7564 | 0.1738 | 0.4368 |
| DINOv3 LoRA + FCOS train400 e2 | 0.3334 | 0.7893 | 0.2211 | 0.4622 |

Val600 matched-FP/custom metrics:

| Model | Conf | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + FCOS train5000 | 0.60 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| DINOv3 LoRA + FCOS train400 e2 | 0.60 | 0.8011 | 0.7643 | 0.3083 | 0.5747 | 0.7706 | 0.8860 | 0.9340 |

Interpretation:

- The pilot is stable and produced a measurable improvement over the frozen
  DINOv3 + FCOS train5000 baseline on the selected val600 subset.
- The comparison is not a matched training comparison:
  - LoRA used only train400 for 2 epochs.
  - The FCOS head was initialized from the stronger frozen-DINOv3 train5000
    checkpoint.
  - The frozen baseline uses cached frozen DINOv3 features and a head trained on
    train5000 for 20 epochs.
- The result is still useful: small attention adaptation appears to improve the
  already trained FCOS head without exploding false positives.

Feasibility:

- M4/MPS full train5000 LoRA training is feasible in memory, but likely slow.
- Based on train400 runtime, one train5000 epoch would be roughly 95-100 minutes
  at similar throughput; multi-epoch train5000 would likely be several hours.
- Colab T4 or another CUDA machine may be preferable for a larger LoRA run, but
  the Mac can support smaller pilots and possibly overnight-scale runs.

Next recommended Phase 3 step:

- Run a slightly longer controlled LoRA pilot, such as train400 for 5-8 epochs
  or train1000 for 2-3 epochs, before committing to full train5000 LoRA.

## 2026-10-13 — DINOv3 LoRA + FCOS train5000 pilot

Completed the controlled ForestPersons DINOv3 + LoRA + FCOS train5000
experiment for the Friday meeting preparation.

Scope:

- Dataset: existing ForestPersons train5000 manifest only, not the full
  downloaded dataset.
- Validation: existing val600 manifest.
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Original DINOv3 weights frozen.
- Trainable parameters:
  - LoRA adapters on q/v attention projections in all 12 transformer layers:
    73,728 parameters.
  - Existing FCOS-style head: 524,646 parameters.
  - Total trainable: 598,374 parameters.
- LoRA configuration: rank 4, alpha 16, dropout 0.05.
- FCOS initialization: loaded from
  `outputs/dinov3_detector/head_v3_fcos_train5000_seed20261012/dinov3_head.pt`.
- LoRA initialization: fresh standard initialization, not train400 pilot
  weights.
- Training: 2 epochs on MPS, batch size 1, gradient accumulation 4, seed
  20261012.

Outputs:

- Experiment directory:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/`.
- Checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.
- Training history:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/train_history.json`.
- Training summary:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/summary.json`.
- Training log:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/train.log`.
- Evaluation summary:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/summary.json`.

Training results:

| Epoch | Images | Optimizer steps | Loss | Classification | Box | Centerness | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,000 | 1,250 | 0.62687 | 0.000637 | 0.016233 | 0.593770 | 5,751.02 |
| 2 | 5,000 | 2,500 | 0.62129 | 0.000613 | 0.014137 | 0.592403 | 6,171.96 |

Training stability:

- LoRA gradients were nonzero for 48/48 LoRA tensors in both epochs.
- FCOS head gradients were nonzero for 18/18 head tensors in both epochs.
- Total training runtime: 11,923.15 seconds, about 3.31 hours.
- Peak observed MPS driver allocation: 11,404,902,400 bytes, about 10.62 GiB.
- No non-finite loss or MPS memory failure was observed.

Val600 evaluation:

- Script: `scripts/evaluate_forestpersons_framework.py`.
- Detector mode: `dinov3_lora`.
- Device: MPS.
- End-to-end evaluation runtime: 269.54 seconds for 600 images,
  0.449 seconds/image.

COCO metrics on selected val600 subset:

| Model | AP | AP50 | AP75 | AR@100 |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + FCOS train5000 | 0.3113 | 0.7564 | 0.1738 | 0.4368 |
| DINOv3 LoRA + FCOS train400 e2 | 0.3334 | 0.7893 | 0.2211 | 0.4622 |
| DINOv3 LoRA + FCOS train5000 e2 | 0.3587 | 0.8139 | 0.2379 | 0.4704 |
| YOLO11s fine-tuned train5000 | 0.5541 | 0.8941 | 0.5687 | 0.6339 |

Custom metrics at confidence 0.60:

| Model | Recall | Precision | FP/image | R@20 | R@40 | R@70 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + FCOS train5000 | 0.7784 | 0.7127 | 0.3917 | 0.5632 | 0.7235 | 0.8601 | 0.9245 |
| DINOv3 LoRA + FCOS train400 e2 | 0.8011 | 0.7643 | 0.3083 | 0.5747 | 0.7706 | 0.8860 | 0.9340 |
| DINOv3 LoRA + FCOS train5000 e2 | 0.8158 | 0.7833 | 0.2817 | 0.6149 | 0.7588 | 0.8912 | 0.9575 |
| YOLO11s fine-tuned train5000 | 0.9039 | 0.7650 | 0.3467 | 0.7701 | 0.8765 | 0.9741 | 0.9717 |

Matched false-positive budget:

- At target FP/image 0.425, the LoRA train5000 run selected confidence 0.55:
  - Recall: 0.8465.
  - Precision: 0.7172.
  - FP/image: 0.4167.
  - R@20: 0.6667.
  - R@40: 0.8059.
  - R@70: 0.9171.
  - R@100: 0.9623.

Interpretation:

- LoRA train5000 improved over the frozen DINOv3 + FCOS train5000 baseline on
  AP, AP50, AP75, AR@100, overall recall, FP/image at confidence 0.60, and
  low-visibility recall.
- Recall@20 increased from 0.5632 for frozen DINOv3 + FCOS to 0.6149 at
  confidence 0.60, and to 0.6667 at the matched 0.425 FP/image budget.
- YOLO11s fine-tuned train5000 remains stronger overall, especially AP and
  Recall@20.
- This is not a fully matched protocol comparison:
  - DINOv3 LoRA adapts a frozen self-supervised backbone with a loaded FCOS head.
  - YOLO11s was fine-tuned end-to-end.
  - All reported values are on the selected val600 pilot subset, not the
    official full ForestPersons validation benchmark.

Recommended next step:

- Use the train5000 LoRA result as the primary DINOv3 Phase 3 finding for the
  Friday discussion, then consider either additional LoRA epochs or a YOLO11s
  proposal + DINOv3 re-scoring hybrid focused on visibility 20/40 recall.

## 2026-10-13 — EfficientAD feasibility re-check

Goal:

- Revisit Davide's Phase 2 anomaly-baseline requirement for EfficientAD after
  RX, DINOv3 patch-kNN, and detector experiments were completed.
- Prioritize scientific validity over producing a weak numerical result.

What was inspected:

- ForestPersons official COCO annotations:
  `data/annotations/train.json`, `val.json`, and `test.json`.
- Existing subset manifests:
  `outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json`
  and `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`.
- Existing RX and DINOv3 patch-kNN anomaly protocols.
- Local `.venv-dinov3` package availability.

Verified dataset facts:

| Split/subset | Images | Annotations | Annotation-free images |
| --- | ---: | ---: | ---: |
| Official train | 67,686 | 145,816 | 13 |
| Official val | 18,243 | 37,395 | 0 |
| Official test | 10,553 | 20,867 | 7 |
| train5000 | 5,000 | 6,415 | 0 |
| val600 | 600 | 750 | 0 |

EfficientAD decision:

- No EfficientAD pilot was trained.
- Reason:
  - Standard EfficientAD needs a normal-only training set.
  - train5000 contains zero annotation-free images.
  - The official train split contains only 13 annotation-free images out of
    67,686 images.
  - COCO-style absence of annotations is not a verified guarantee that a frame
    is person-free; those frames may be unlabeled edge cases.
- Therefore a standard normal-image EfficientAD protocol is not scientifically
  valid with the current ForestPersons subsets.

Environment notes:

- `anomalib`, `lightning`, and `pytorch_lightning` are not installed in
  `.venv-dinov3`.
- `torch` 2.9.0 and `torchvision` 0.24.0 are available.
- Preferred future implementation path, if needed, is Anomalib EfficientAD in a
  separate isolated environment.

Artifacts:

- Feasibility script:
  `scripts/analyze_efficientad_feasibility.py`.
- Summary JSON:
  `outputs/efficientad_feasibility/forestpersons_20261013/summary.json`.
- Markdown report:
  `outputs/efficientad_feasibility/forestpersons_20261013/efficientad_feasibility_report.md`.
- Registry updated:
  `docs/experiment_registry/experiments.csv`.

Recommended valid alternative:

- Build a masked-background/crop dataset from train images, excluding annotated
  person boxes with a safety margin.
- Treat that as a masked-background EfficientAD adaptation, not as a standard
  normal-image EfficientAD baseline.
- Compare it only with RX and DINOv3 patch-kNN using patch/heatmap AUROC and AP
  unless a fixed heatmap-to-box proposal method is defined before evaluation.

## 2026-10-13 — DINOv3 ViT-L web-vs-satellite feasibility smoke

Goal:

- Test whether a controlled web-pretrained versus satellite-pretrained DINOv3
  comparison is feasible while controlling architecture size.
- Use ViT-L/16 for both models:
  - Web: `facebook/dinov3-vitl16-pretrain-lvd1689m`.
  - Satellite: `facebook/dinov3-vitl16-pretrain-sat493m`.
- Do not train a detector and do not modify existing ViT-S checkpoints.

Implementation:

- Added `scripts/benchmark_dinov3_vitl_feasibility.py`.
- Added tests in `tests/test_dinov3_vitl_feasibility.py`.
- Saved report:
  `docs/reports/dinov3_vitl_web_sat_feasibility_2026-10-13.md`.
- Saved machine-readable summary:
  `outputs/dinov3_vitl_feasibility/web_vs_sat_val10_short336_20261013/summary.json`.

Access and checkpoint facts:

- Both Hugging Face repositories are public but gated with manual access.
- Local authenticated access was available; no licensing or authentication
  restriction was bypassed.
- Config/preprocessor files and model weights were accessible.
- Each ViT-L checkpoint has a `model.safetensors` file of 1,212,559,808 bytes,
  about 1.13 GiB.
- Both checkpoints together require about 2.26 GiB of cached weights.
- Both configs are architecture matched:
  - hidden size 1024.
  - 24 transformer layers.
  - 16 attention heads.
  - patch size 16.

Smoke protocol:

- Device: MPS.
- Images: first 10 images from
  `outputs/subsets/forestpersons_val_600_seed20261010_manifest.json`.
- Initial forward: one image, short side 224.
- Benchmark forward: 10 images, short side 336.
- Backward smoke: one image, short side 224, rank-4 q/v LoRA plus a ViT-L FCOS
  head. No optimizer step was run.

Results:

| Model | Load time | Forward short336 total | Seconds/image | Peak observed MPS driver memory | Backward short224 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 94.21 s | 3.65 s | 0.365 s | 2.41 GiB | 1.64 s |
| Satellite ViT-L | 391.74 s | 3.37 s | 0.337 s | 2.41 GiB | 0.32 s |

Feature shapes:

- Short side 224 on a 1920 x 1080 image resized to 400 x 224 produced
  14 x 25 x 1024 features.
- Short side 336 on a 1920 x 1080 image resized to 592 x 336 produced
  21 x 37 x 1024 features.
- Short side 336 on a 2288 x 1080 image resized to 704 x 336 produced
  21 x 44 x 1024 features.

Detector compatibility:

- Existing ViT-S FCOS/LoRA checkpoints are not compatible with ViT-L because the
  feature dimension changes from 384 to 1024.
- ViT-L needs `DINOv3FCOSHead(input_dim=1024)`.
- Rank-4 q/v LoRA across 24 ViT-L layers has 393,216 trainable LoRA parameters.
- ViT-L FCOS head has 627,046 trainable parameters.
- Total trainable parameters for ViT-L LoRA + FCOS would be about 1,020,262.
- Existing ViT-S feature caches cannot be reused.

Interpretation:

- Reduced-resolution ViT-L forward inference is feasible on the Mac M4 MPS.
- A one-image ViT-L LoRA + FCOS backward smoke is feasible at short side 224.
- This does not yet prove that the canonical short-side-672 LoRA training setup
  is feasible for train400, because token count and attention cost grow sharply
  with resolution.
- No AP, recall, or detector metrics were produced.

Recommended next action:

1. Run a 5-10 image ViT-L LoRA + FCOS training smoke at short side 336.
2. If stable, run a 20-50 image smoke at short side 672.
3. Only then consider a matched train400 experiment comparing web ViT-L and
   satellite ViT-L with identical training budget and val600 evaluation.

## 2026-10-13 — DINOv3 ViT-L LoRA + FCOS training smoke

Goal:

- Test whether matched LoRA + FCOS detector training is feasible for the
  official DINOv3 ViT-L web and satellite checkpoints.
- Preserve all existing ViT-S checkpoints/results.
- Do not start train400 or train5000.

Code changes:

- Parameterized `src/forestpersons/dinov3_lora.py` with optional `model_id`.
- LoRA checkpoints now save `backbone_model_id` and `head_input_dim`, allowing
  ViT-L checkpoints to load with the correct 1024-dimensional FCOS head.
- Added `--model-id` and `--head-init-seed` to
  `scripts/train_dinov3_lora_fcos.py`.
- Added `--short-side` to `scripts/evaluate_forestpersons_framework.py` for
  online LoRA evaluation.
- Added report:
  `docs/reports/dinov3_vitl_lora_fcos_training_smoke_2026-10-13.md`.

Matched short-side-336 smoke:

- Models:
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`.
  - `facebook/dinov3-vitl16-pretrain-sat493m`.
- Train images: first 10 from
  `outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json`.
- Epochs: 1.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Head init seed: 20261013.
- Head initialization: random and matched; no ViT-S head weights reused.
- LoRA: rank 4, alpha 16, dropout 0.05 on q/v attention projections.
- Optimizer/loss settings: same as the existing LoRA FCOS script.

Results:

| Model | Images | Optimizer steps | Loss | Cls | Box | Centerness | Seconds | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 10 | 3 | 1.1591 | 0.02235 | 0.19944 | 0.73783 | 9.76 | 7.21 GiB |
| Satellite ViT-L | 10 | 3 | 1.1571 | 0.02247 | 0.20198 | 0.73071 | 8.47 | 7.21 GiB |

Gradient and parameter checks:

- Both models had nonzero gradients for 96/96 trainable LoRA tensors and 18/18
  FCOS head tensors.
- Trainable LoRA parameters: 393,216.
- Trainable ViT-L FCOS head parameters: 627,046.
- Total trainable parameters: 1,020,262.
- Total parameters including frozen backbone: 304,149,862.

Evaluation sanity check:

- Each short-336 checkpoint was loaded by the canonical evaluation framework.
- Evaluated on the first 5 images from val600 at short side 336.
- This is a pipeline sanity check only; AP/recall are not meaningful because the
  heads were trained on only 10 images for one epoch.

| Model | Val images | GT | Raw predictions | End-to-end sec/image |
| --- | ---: | ---: | ---: | ---: |
| Web ViT-L smoke | 5 | 6 | 1,500 | 0.385 |
| Satellite ViT-L smoke | 5 | 6 | 1,500 | 0.364 |

Short-side-672 attempt:

- After both short-336 runs succeeded, attempted web ViT-L on 20 train images at
  short side 672.
- Result: failed before the first epoch completed with MPS out-of-memory:
  `MPS allocated: 19.36 GiB`, `other allocations: 626.77 MiB`,
  `max allowed: 20.13 GiB`; attempted allocation was about 607.55 MiB.
- Satellite short-side-672 was not attempted after the web run failed.
- No MPS high-watermark override was used.

Artifacts:

- Web short-336 checkpoint:
  `outputs/dinov3_vitl_lora_smoke/web_vitl_train10_short336_seed20261013/dinov3_lora_fcos.pt`.
- Satellite short-336 checkpoint:
  `outputs/dinov3_vitl_lora_smoke/sat_vitl_train10_short336_seed20261013/dinov3_lora_fcos.pt`.
- Web sanity evaluation:
  `outputs/dinov3_vitl_lora_smoke/web_vitl_train10_short336_seed20261013/eval_val5_sanity/summary.json`.
- Satellite sanity evaluation:
  `outputs/dinov3_vitl_lora_smoke/sat_vitl_train10_short336_seed20261013/eval_val5_sanity/summary.json`.
- Short-672 failure log:
  `outputs/dinov3_vitl_lora_smoke_web_train20_short672.log`.

Conclusion:

- ViT-L web-vs-satellite LoRA + FCOS training is feasible on this Mac only at
  reduced resolution in the tested setup.
- Canonical short-side-672 ViT-L LoRA training is blocked by MPS memory pressure
  on the M4 16 GB machine.
- A fair next experiment is either:
  1. a reduced-resolution train400 comparison at short side 336, clearly labeled
     as not directly matched to ViT-S short-side-672 results; or
  2. move the short-side-672 ViT-L comparison to a CUDA machine with more GPU
     memory.

## DINOv3 ViT-L Web vs Satellite Train400 Short-Side-336 Pilot

Completed a controlled reduced-resolution train400 comparison between the two
official ViT-L/16 DINOv3 checkpoints:

- Web: `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Satellite: `facebook/dinov3-vitl16-pretrain-sat493m`.

Matched protocol:

- Existing ForestPersons train400 subset and val600 subset.
- Short side 336.
- 2 epochs.
- Batch size 1.
- Gradient accumulation 4.
- Seed 20261013.
- Random matched FCOS head initialization with `--head-init-seed 20261013`.
- LoRA rank 4, alpha 16, dropout 0.05 on q/v attention projections in all 24
  ViT-L transformer layers.
- Frozen original DINOv3 backbone weights.
- Trainable parameters: 1,020,262 total; 393,216 LoRA and 627,046 FCOS head.

Training completed on MPS for both models:

| Model | Epoch 1 loss | Epoch 2 loss | Optimizer steps | Total time | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.9464 | 0.7823 | 200 | 640.4 s | 7.26 GiB |
| Satellite ViT-L | 0.9879 | 0.8291 | 200 | 635.2 s | 7.26 GiB |

Val600 results using the canonical evaluation framework:

| Model | AP | AP50 | AP75 | AR@100 | Closest FP/image | Recall | Precision | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Web ViT-L | 0.1409 | 0.4562 | 0.0332 | 0.2987 | 0.3867 | 0.4379 | 0.5857 | 0.2586 | 0.3412 | 0.4870 | 0.6179 |
| Satellite ViT-L | 0.0656 | 0.2745 | 0.0085 | 0.2435 | 0.4033 | 0.3057 | 0.4862 | 0.1322 | 0.2176 | 0.3472 | 0.4811 |

Interpretation:

- In this reduced-resolution train400 setting, web ViT-L outperformed satellite
  ViT-L across AP, AP50, AP75, AR@100, matched-FP recall, and low-visibility
  recall.
- This is not enough to conclude that web pretraining is intrinsically better:
  the pilot uses short-side 336, only 400 training images, 2 epochs, and random
  FCOS heads. ForestPersons RGB UAV imagery may also be closer to the web
  checkpoint's pretraining domain than to the satellite checkpoint's domain.
- Short-side-672 ViT-L training remains blocked on the 16 GB Mac because the
  previous bounded smoke test hit MPS out-of-memory.

Artifacts:

- Web checkpoint:
  `outputs/dinov3_vitl_lora_train400/web_vitl_short336_seed20261013_e2/dinov3_lora_fcos.pt`.
- Satellite checkpoint:
  `outputs/dinov3_vitl_lora_train400/sat_vitl_short336_seed20261013_e2/dinov3_lora_fcos.pt`.
- Comparison CSV:
  `outputs/dinov3_vitl_lora_train400/comparison_short336_seed20261013/summary_metrics.csv`.
- Report:
  `docs/reports/dinov3_vitl_web_sat_train400_short336_2026-10-13.md`.

## DINOv3 ViT-S LoRA Lightweight FPN FCOS Implementation

Implemented and smoke-tested a lightweight multi-scale FCOS-style head for the
existing DINOv3 ViT-S + LoRA pipeline. No full training was started.

Sanity checks:

- Reviewed the existing single-scale FCOS box scaling, assignment, loss, and
  decoding path.
- No verified coordinate-scaling bug was found.
- Existing code already uses separate `scale_x` and `scale_y` when converting
  between original and resized image coordinates.

New implementation:

- Added `DINOv3FPNFCOSHead` to `src/forestpersons/dinov3_detector.py`.
- Added `build_fpn_fcos_targets` for scale-aware FCOS target assignment.
- Added `decode_fpn_fcos_predictions` for joint multi-level decoding and NMS.
- Added `load_compatible_fcos_weights_into_fpn` to reuse compatible
  single-scale FCOS weights.
- Added bounded smoke script:
  `scripts/smoke_dinov3_lora_fpn_fcos.py`.
- Added tests:
  `tests/test_dinov3_fpn_fcos.py`.

Architecture:

- Fine level: DINO patch grid, stride 16.
- Medium level: pooled fine grid with lightweight adapter, stride 32.
- Coarse level: pooled medium grid with lightweight adapter, stride 64.
- Shared FCOS tower and prediction heads are applied across all levels.

Weight reuse from the existing best ViT-S LoRA + FCOS checkpoint:

- Reused: `proj.*`, `shared.*`, `classification.*`, `box.*`, and
  `centerness.*`.
- Newly initialized: `pyramid_adapters.medium.*` and
  `pyramid_adapters.coarse.*`.

MPS smoke test:

- Checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.
- Images: 2 real train400 images.
- Short side: 672.
- Runtime: 3.32 seconds.
- Peak observed MPS driver memory: approximately 5.95 GB.
- Losses were finite.
- LoRA gradients were nonzero for 48/48 trainable tensors.
- FPN head gradients were nonzero for 26/26 trainable tensors.
- Smoke output:
  `outputs/dinov3_lora_fpn_fcos/smoke_train2_seed20261013/summary.json`.

Validation:

- Ran full test suite:
  `PYTHONPATH=$LOGXY_ARTIFACT_ROOT/src:$LOGXY_ARTIFACT_ROOT $LOGXY_ARTIFACT_ROOT/.venv-dinov3/bin/python -m pytest tests -q`
- Result: 53 tests passed.

Report:

- `docs/reports/dinov3_lora_fpn_fcos_implementation_2026-10-13.md`.

Next recommended step:

- After approval, run a bounded FPN training pilot and compare against the
  existing single-scale ViT-S LoRA + FCOS baseline using the same val600
  evaluation framework and matched FP/image budgets.

## DINOv3 ViT-S LoRA FPN FCOS Train400 Pilot

Completed the first bounded FPN FCOS training pilot. No train5000 or longer
training was started.

Configuration:

- Dataset: existing ForestPersons train400 subset.
- Evaluation: existing val600 subset.
- Resolution: short side 672.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Device: Apple MPS.
- Initialization checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

Initialization details:

- Loaded LoRA adapters from the trained single-scale LoRA + FCOS checkpoint.
- Reused compatible FCOS head weights: `proj.*`, `shared.*`,
  `classification.*`, `box.*`, and `centerness.*`.
- Newly initialized: `pyramid_adapters.medium.*` and
  `pyramid_adapters.coarse.*`.

Training:

| Epoch | Loss | Cls | Box | Centerness | Positive cells | Optimizer steps | Seconds | MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2371 | 0.0109 | 0.1317 | 0.9628 | 4,245 | 100 | 569.0 | 10.56 GiB |
| 2 | 1.0790 | 0.0048 | 0.0712 | 0.9319 | 4,245 | 200 | 435.2 | 10.56 GiB |

Total runtime: 1004.4 seconds, approximately 16.7 minutes. No MPS out-of-memory
error occurred.

Val600 comparison at the best evaluated threshold with FP/image <= 0.425:

| Model | AP | AP50 | AP75 | AR@100 | Conf | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-scale LoRA + FCOS baseline | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| FPN LoRA + FCOS train400 pilot | 0.3062 | 0.7836 | 0.1622 | 0.4579 | 0.60 | 0.8024 | 0.7365 | 0.3583 | 0.6149 | 0.7471 | 0.8964 | 0.9151 |

Conclusion:

- The FPN pilot trained successfully and was memory-feasible.
- It did not improve AP75, matched-FP recall, or 20% visibility recall.
- The existing single-scale ViT-S LoRA + FCOS detector remains the stronger
  DINOv3 detector.
- Do not scale this exact FPN setup to train5000 before inspecting level usage,
  false positives, and the assignment/loss balance.

Artifacts:

- Checkpoint:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_fcos.pt`.
- Training summary:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/summary.json`.
- Evaluation summary:
  `outputs/dinov3_lora_fpn_fcos/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/summary.json`.
- Report:
  `docs/reports/dinov3_lora_fpn_fcos_train400_pilot_2026-10-13.md`.

## DINOv3 ViT-S LoRA FCOS Localization and Assignment Audit

Completed a read-only audit of the current single-scale DINOv3 ViT-S + LoRA +
FCOS detector. No implementation change, training, inference, or checkpoint
modification was performed.

Findings:

- Current box regression uses `SmoothL1Loss(log1p(softplus(pred_distance)),
  log1p(target_distance), beta=0.25)`.
- The current loss is not IoU, GIoU, DIoU, or CIoU.
- Positive assignment uses the stride-16 DINO patch grid, center sampling with
  radius 24 resized pixels, smaller-area conflict resolution, and center
  fallback if no positive cell exists.
- Train5000 assignment audit:
  - 6,431 GT annotations.
  - 50,600 positive cells.
  - 7 center fallbacks.
  - 121 multi-GT overwrites.
  - All GT annotations received at least one positive.
- 20%-visibility people receive fewer positives on average:
  - Visibility 20: 7.10 positives/GT.
  - Visibility 40: 7.77 positives/GT.
  - Visibility 70: 8.25 positives/GT.
  - Visibility 100: 8.29 positives/GT.
- No confirmed correctness bug was found in coordinate scaling, assignment, or
  decoding.

Recommendation:

- Do not change architecture next.
- Run one controlled single-scale experiment adding an IoU-family localization
  term, preferably `SmoothL1 distance loss + lambda_iou * (1 - GIoU)`, while
  keeping data, initialization, optimizer, resolution, LoRA configuration, and
  training budget matched to the existing baseline.
- Primary success target: improve AP75 above 0.2379 without reducing recall at
  FP/image <= 0.425 below 0.8465 or 20%-visibility recall below 0.6667.

Report:

- `docs/reports/dinov3_lora_fcos_localization_assignment_audit_2026-10-13.md`.

## DINOv3 ViT-S LoRA FCOS GIoU Localization Loss Pilot

Implemented and tested an optional GIoU localization term for the existing
single-scale DINOv3 ViT-S + LoRA + FCOS training path.

Implementation:

- Added configurable `--giou-loss-weight` to
  `scripts/train_dinov3_lora_fcos.py`.
- The default `lambda_iou = 0` preserves the previous Smooth L1-only behavior.
- New box objective:
  `smooth_l1_log_distance_loss + lambda_iou * (1 - GIoU)`.
- Predicted and target FCOS L/T/R/B distances are decoded into centered boxes
  in the same distance coordinate system before GIoU is computed.
- Training history now logs Smooth L1 and GIoU box components separately.
- Added focused unit tests in `tests/test_fcos_giou_loss.py`.

Smoke test:

- 2-image MPS forward/backward smoke with `lambda_iou = 1` completed.
- Loss was finite.
- LoRA gradients: 48/48 nonzero.
- Head gradients: 18/18 nonzero.
- Peak MPS driver memory: approximately 5.52 GiB.

Controlled train400 comparison:

- Dataset: existing ForestPersons train400 subset.
- Evaluation: existing val600 subset.
- Resolution: short side 672.
- Epochs: 2.
- Seed: 20261013.
- Device: Apple MPS.
- Both runs initialized from:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.

Training:

| Run | Lambda | Epoch 2 loss | Epoch 2 Smooth L1 | Epoch 2 GIoU | Runtime | Peak MPS driver memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train400 continuation | 0 | 0.6101 | 0.0100 | 0.0000 | 854.3 s | 10.62 GiB |
| Train400 continuation | 1 | 0.9028 | 0.0109 | 0.1452 | 936.4 s | 10.53 GiB |

Val600 results at the best evaluated threshold with FP/image <= 0.425:

| Model | AP | AP50 | AP75 | AR@100 | Conf | Recall | Precision | FP/image | R20 | R40 | R70 | R100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-scale LoRA + FCOS train5000 reference | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.55 | 0.8465 | 0.7172 | 0.4167 | 0.6667 | 0.8059 | 0.9171 | 0.9623 |
| Train400 continuation, lambda=0 | 0.3495 | 0.8117 | 0.2326 | 0.4657 | 0.60 | 0.8171 | 0.7574 | 0.3267 | 0.6092 | 0.7765 | 0.8912 | 0.9528 |
| Train400 continuation, lambda=1 | 0.3516 | 0.8125 | 0.2321 | 0.4685 | 0.60 | 0.8198 | 0.7599 | 0.3233 | 0.6322 | 0.7706 | 0.8860 | 0.9528 |

Conclusion:

- GIoU `lambda_iou = 1` slightly improved AP, AR@100, matched-FP recall,
  precision, and 20%-visibility recall relative to the matched lambda=0
  continuation.
- It did not improve AP75; AP75 changed from 0.2326 to 0.2321.
- Because AP75/localization was the primary target, this is not a clear
  improvement and should not replace the current train5000 LoRA baseline.
- Do not scale this exact GIoU configuration to train5000 unless the next goal
  explicitly prioritizes low-visibility recall over localization precision.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_fcos_giou_train400_comparison_2026-10-13.md`.
- Comparison CSV:
  `outputs/dinov3_lora_giou/train400_lambda_comparison_seed20261013.csv`.
- Lambda 0 output:
  `outputs/dinov3_lora_giou/train400_lambda0_seed20261013_from_lora5k_e2`.
- Lambda 1 output:
  `outputs/dinov3_lora_giou/train400_lambda1_seed20261013_from_lora5k_e2`.

## DINOv3 ViT-S LoRA FCOS Candidate Failure Analysis

Completed a read-only diagnostic analysis of the best existing DINOv3 ViT-S +
LoRA + single-scale FCOS train5000 checkpoint on val600.

No training, downloads, checkpoint changes, or new model inference were
performed. The analysis reused saved post-NMS COCO prediction files.

Important limitation:

- Saved predictions are post-NMS and contain only final combined scores.
- Pre-NMS candidates, raw classification logits, and raw centerness logits are
  unavailable.
- Therefore, NMS suppression and separate class-vs-centerness suppression cannot
  be proven from these artifacts.

Operating points:

- DINOv3 LoRA + FCOS: confidence 0.55, IoU 0.50, FP/image 0.4167.
- YOLO11s fine-tuned: confidence 0.05, IoU 0.50, FP/image 0.3467.

Overall candidate quality:

| Model | Post-NMS predictions | Detected | Best IoU >= 0.50 | Best IoU >= 0.75 | Mean best IoU | Median best IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3 LoRA + FCOS | 46,586 | 634/749 | 690/749 | 309/749 | 0.705 | 0.727 |
| YOLO11s fine-tuned | 2,622 | 677/749 | 713/749 | 501/749 | 0.781 | 0.824 |

DINOv3 miss categories at confidence 0.55:

| Category | Count |
| --- | ---: |
| Detected | 634 |
| Candidate overlaps but IoU < 0.50 | 59 |
| Candidate IoU >= 0.50 but confidence < 0.55 | 56 |
| No meaningful surviving candidate | 0 |

Visibility findings:

| Visibility | GT | Detected | Missed | Localization miss | Confidence miss | Mean best confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 174 | 116 | 58 | 32 | 26 | 0.583 |
| 40 | 170 | 137 | 33 | 19 | 14 | 0.656 |
| 70 | 193 | 177 | 16 | 4 | 12 | 0.718 |
| 100 | 212 | 204 | 8 | 4 | 4 | 0.770 |

Conclusion:

- DINOv3 usually has a surviving post-NMS candidate near each person.
- The main failures are split between localization below IoU 0.50 and low
  confidence despite IoU >= 0.50.
- 20%-visibility people account for the largest remaining miss group.
- YOLO remains better localized, with many more best candidates above IoU 0.75.

Recommended next experiment:

- Run a controlled scoring/centerness diagnostic before changing architecture:
  save pre-NMS DINOv3 FCOS candidates with separate class score, centerness
  score, decoded box, and final score; compare fixed scoring formulas using a
  training-derived calibration split; then evaluate the locked rule on val600 at
  FP/image <= 0.425.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_fcos_val600_candidate_failure_analysis_2026-10-13.md`.
- Script:
  `scripts/analyze_lora_candidate_failures.py`.
- Output directory:
  `outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013`.

## DINOv3 ViT-S LoRA FCOS Classification/Centerness Scoring Diagnostic

Completed a controlled scoring diagnostic for the best DINOv3 ViT-S + LoRA +
single-scale FCOS checkpoint. No training, downloads, or checkpoint changes were
performed.

Current FCOS inference score:

```text
score = sqrt(class_score * centerness_score)
```

The decoder filters by score, keeps top 300 candidates per image, decodes boxes,
and applies NMS at IoU 0.50.

Pre-NMS candidate export:

| Split | Images | Candidates | Runtime | Seconds/image | File size |
| --- | ---: | ---: | ---: | ---: | ---: |
| train400 calibration | 400 | 1,281,168 | 213.2 s | 0.533 | 64 MiB |
| val600 | 600 | 1,917,636 | 310.6 s | 0.518 | 96 MiB |

Calibration:

- Used existing train400 only, with zero overlap with val600.
- Tested fixed formulas:
  - `sqrt(class_score * centerness_score)`.
  - `class_score`.
  - `class_score * centerness_score^0.5`.
  - `class_score * centerness_score`.
- Selection criterion: highest recall under FP/image <= 0.425.

Calibration selected:

| Rule | Threshold | Recall | Precision | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current rule | 0.55 | 0.9219 | 0.7732 | 0.3550 | 0.8036 |
| Selected class-only rule | 0.50 | 0.9352 | 0.7519 | 0.4050 | 0.8571 |

Val600 locked evaluation:

| Rule | Threshold | AP | AP50 | AP75 | AR@100 | Recall | Precision | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current rule | 0.55 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8465 | 0.7172 | 0.4167 | 0.6667 |
| Calibrated class-only rule | 0.50 | 0.3564 | 0.8213 | 0.2299 | 0.4689 | 0.8531 | 0.6733 | 0.5167 | 0.6724 |

Result:

- The calibrated class-only rule recovered 5 additional true positives and 1
  additional 20%-visibility true positive.
- It also added 60 false positives and exceeded the target FP/image budget on
  val600.
- AP, AP75, and AR@100 did not improve.
- Under the val600 FP budget, alternative formulas did not produce a meaningful
  20%-visibility recall improvement over the current rule.

Conclusion:

- Do not replace the current FCOS score with class-only scoring.
- Centerness suppression is not the dominant bottleneck at the accepted
  FP/image operating point.
- The next controlled experiment should focus on localization/assignment,
  especially increasing useful supervision for 20%-visibility and small persons.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_fcos_scoring_calibration_2026-10-13.md`.
- Script:
  `scripts/analyze_fcos_scoring_calibration.py`.
- Output directory:
  `outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013`.

## DINOv3 ViT-S LoRA YOLO-Style DFL/IoU Head Implementation

Implemented a separate YOLO-inspired one-stage detection head for DINOv3
ViT-S/16 patch features without modifying the existing FCOS implementation or
checkpoints.

Key implementation details:

- New module: `src/forestpersons/dinov3_yolo.py`.
- New smoke script: `scripts/smoke_dinov3_lora_yolo_head.py`.
- Uses the existing DINOv3 ViT-S LoRA backbone checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.
- New YOLO-style head is freshly initialized; no FCOS head weights are reused.
- Single stride-16 patch feature map only.
- Separate classification and regression branches.
- Distribution Focal Loss with default `reg_max=32`.
- IoU regression loss on decoded L/T/R/B boxes.
- Task-aligned positive assignment using class score and IoU.
- Original DINOv3 weights remain frozen; LoRA adapters and the new detection
  head are trainable for future experiments.

Smoke test:

- Command:
  `PYTHONPATH=src .venv-dinov3/bin/python scripts/smoke_dinov3_lora_yolo_head.py --output-dir outputs/dinov3_lora_yolo_head/smoke_train2_seed20261013 --max-images 2 --short-side 672 --device mps`
- Images: 2 real ForestPersons train400 images.
- Runtime: 4.24 s.
- Peak MPS driver memory: 5,922,734,080 bytes, about 5.52 GiB.
- Trainable parameters: 619,141 total; 73,728 LoRA and 545,413 YOLO head.
- Gradients: nonzero gradients observed for all 48 trainable LoRA tensors and
  all 18 trainable head tensors.
- Losses were finite. Initial classification loss is large because the YOLO
  BCE term is summed over stride-16 cells and normalized by positive target
  score; monitor this during train400.

Tests:

- Added focused tests in `tests/test_dinov3_yolo_head.py` for output shapes,
  DFL decoding/loss gradients, IoU loss, task-aligned assignment, tiny-box
  fallback, no-positive stability, and separate x/y coordinate scaling.
- Ran targeted tests:
  `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests/test_dinov3_yolo_head.py tests/test_dinov3_lora.py -q`
  with 15 tests passed.
- Ran full local test suite:
  `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests -q`
  with 66 tests passed.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_yolo_head_implementation_2026-10-13.md`.
- Smoke output:
  `outputs/dinov3_lora_yolo_head/smoke_train2_seed20261013/summary.json`.
- Smoke head checkpoint:
  `outputs/dinov3_lora_yolo_head/smoke_train2_seed20261013/smoke_yolo_head.pt`.

Conclusion:

- The architecture is ready for a controlled train400 pilot on Mac MPS.
- Main limitation remains the single stride-16 feature map, which may limit
  small-person localization and AP75 despite better YOLO-style assignment/losses.

## DINOv3 ViT-S LoRA YOLO-Style Head Train400 Pilot

Completed the first controlled train400 pilot for the YOLO-style DFL/IoU head.
No train5000 training, dataset downloads, FCOS checkpoint changes, Faster R-CNN,
or Deformable DETR work was performed.

Training configuration:

- Dataset: existing ForestPersons train400 subset.
- Evaluation: existing val600 subset.
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Initialization:
  - DINOv3 LoRA weights loaded from the best existing ViT-S LoRA + FCOS
    train5000 checkpoint.
  - YOLO-style detection head freshly initialized.
  - FCOS head weights were intentionally not reused.
- Resolution: short-side 672.
- Epochs: 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Seed: 20261013.
- Device: MPS.

Training diagnostics:

| Epoch | Loss | Classification | IoU | DFL | Positives/image | Optimizer steps | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.8272 | 36.6765 | 0.6439 | 2.2143 | 11.70 | 100 | 499.5 |
| 2 | 9.0459 | 3.3684 | 0.4191 | 1.6896 | 11.71 | 200 | 447.4 |

Stability:

- Non-finite losses: none.
- Center fallbacks: 0.
- Clipped target distances: 0.
- Multi-GT overwrites: 32 in epoch 1, 28 in epoch 2.
- Nonzero gradients in epoch 2: 48/48 LoRA tensors and 18/18 head tensors.
- Parameter update check:
  - First LoRA trainable tensor max absolute change: 0.0069.
  - First head trainable tensor max absolute change: 0.0395.
- Total runtime: 947.1 s, about 15.8 min.
- Peak MPS driver memory: 11,645,124,608 bytes, about 10.85 GiB.

Val600 metrics:

| Metric | Value |
| --- | ---: |
| AP@[0.50:0.95] | 0.1339 |
| AP50 | 0.5094 |
| AP75 | 0.0171 |
| AR@100 | 0.3198 |

Fixed confidence operating point, confidence 0.60:

| Metric | Value |
| --- | ---: |
| Recall | 0.3992 |
| Precision | 0.6308 |
| FP/image | 0.2917 |
| R20 | 0.3161 |
| R40 | 0.3706 |
| R70 | 0.4974 |
| R100 | 0.4009 |

Exploratory matched-FP result:

- The val600 confidence sweep selected threshold 0.60 for FP/image <= 0.425.
- Recall: 0.3992.
- Precision: 0.6308.
- FP/image: 0.2917.
- R20: 0.3161.
- This threshold selection is exploratory because it uses val600.

Comparison on val600:

| Model | AP | AP50 | AP75 | AR@100 | Recall at FP<=0.425 | Precision | FP/image | R20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s fine-tuned train5000 | 0.5541 | 0.8941 | 0.5687 | 0.6339 | 0.9039 | 0.7650 | 0.3467 | 0.7701 |
| DINOv3 LoRA + FCOS train5000 | 0.3587 | 0.8139 | 0.2379 | 0.4704 | 0.8465 | 0.7172 | 0.4167 | 0.6667 |
| DINOv3 LoRA + FCOS train400 | 0.3334 | 0.7893 | 0.2211 | 0.4622 | 0.8011 | 0.7643 | 0.3083 | 0.5747 |
| DINOv3 LoRA + YOLO-style train400 | 0.1339 | 0.5094 | 0.0171 | 0.3198 | 0.3992 | 0.6308 | 0.2917 | 0.3161 |

Conclusion:

- The YOLO-style head is technically functional and safe on the Mac for
  train400-scale pilots.
- It is not ready to scale to train5000 in this exact form.
- The main issue is not trainability but performance: AP75 and recall at the
  accepted FP/image budget are far below FCOS.
- Recommended next step is a targeted diagnostic/correction of YOLO confidence
  ranking and localization before any larger training run.
- Verification: `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests -q`
  passed 66 tests.

Artifacts:

- Script:
  `scripts/train_dinov3_lora_yolo_head.py`.
- Checkpoint:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt`.
- Training summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/summary.json`.
- Evaluation summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/summary.json`.
- Report:
  `docs/reports/dinov3_lora_yolo_head_train400_pilot_2026-10-13.md`.

## DINOv3 ViT-S LoRA YOLO-Style Head AP75 Diagnosis

Completed a read-only diagnosis of the poor AP75 from the YOLO-style train400
pilot. No retraining, checkpoint modification, architecture change, or dataset
download was performed.

Confirmed findings:

- No DFL bin, stride, coordinate scaling, or clipping bug was found.
- Training assignment produced meaningful positives and no clipped target
  distances.
- Saved post-NMS val600 predictions show:
  - 749 GT persons.
  - Mean best-candidate IoU: 0.627.
  - Median best-candidate IoU: 0.641.
  - GT with best candidate IoU >= 0.50: 89.05%.
  - GT with best candidate IoU >= 0.75: 9.88%.
  - Only 35.23% of GT with IoU >= 0.50 also had score >= 0.60.
- Visibility breakdown for best candidate IoU >= 0.75:
  - Visibility 20: 4.60%.
  - Visibility 40: 10.59%.
  - Visibility 70: 8.81%.
  - Visibility 100: 14.62%.
- Compared with DINOv3 LoRA + FCOS train5000:
  - FCOS best candidate IoU >= 0.75: 41.26%.
  - YOLO-style best candidate IoU >= 0.75: 9.88%.

Bounded pre-NMS check:

- Ran on the first 30 val600 images only; no training.
- GT persons: 33.
- Pre-NMS candidates with IoU >= 0.75 existed for 36.36% of GT.
- Post-NMS candidates with IoU >= 0.75 remained for only 3.03% of GT.
- 11 GT instances had a pre-NMS IoU >= 0.75 candidate that was lost after
  score-ranked NMS.

Conclusion:

- The immediate failure is localization-aware ranking, not a simple coordinate
  transform bug.
- The model can sometimes produce well-localized boxes before NMS, but class
  score alone does not rank them above worse overlapping boxes.
- Do not scale this exact YOLO-style head to train5000.
- Recommended next correction: add localization-aware scoring, for example a
  lightweight predicted-IoU/quality branch or a quality-focal/varifocal-style
  class target, then rank/NMS with `class_score * quality_score` in a new
  controlled train400 pilot.
- Verification: `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests -q`
  passed 66 tests.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_yolo_head_ap75_diagnosis_2026-10-13.md`.
- Analysis script:
  `scripts/analyze_dinov3_yolo_head_failures.py`.
- Summary:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/summary.json`.
- Per-GT CSV:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/gt_best_candidate_analysis.csv`.
- Pre-NMS subset:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/pre_nms_subset30.json`.

## DINOv3 ViT-S LoRA YOLO-Style Quality Branch

Verified the NMS ranking failure and implemented an optional localization
quality branch for the YOLO-style DFL head. No train400/train5000 training was
run and no existing checkpoints were modified.

NMS verification on the same first 30 val600 images:

| Quantity | Count |
| --- | ---: |
| Images | 30 |
| GT persons | 33 |
| GT with pre-NMS best IoU >= 0.75 | 12 |
| Lost at confidence threshold | 0 |
| Lost at top-k | 0 |
| Lost at NMS | 12 |
| Kept after NMS | 0 |

This confirms the failure mechanism: class-score top-k preserved the good
candidates, but class-score-ranked NMS suppressed them with worse-localized
overlapping boxes that had higher class scores.

Implementation details:

- Updated `src/forestpersons/dinov3_yolo.py`.
- Added optional `include_quality` branch to `DINOv3YOLOHead`.
- Existing class-only behavior remains the default.
- Existing class-only YOLO-style checkpoints remain compatible.
- Quality target for positive cells is detached IoU between decoded prediction
  and assigned GT target.
- Background/unmatched quality target is zero.
- Quality loss is BCE-with-logits averaged over the grid.
- Optional inference mode uses:
  `score = class_score * predicted_quality`.
- Ground-truth IoU is not used during inference.

Updated scripts:

- `scripts/smoke_dinov3_lora_yolo_head.py`.
- `scripts/train_dinov3_lora_yolo_head.py`.
- `scripts/evaluate_forestpersons_framework.py`.

Quality smoke test:

- Command used `--include-quality --quality-loss-weight 1.0 --score-mode class_quality`.
- Images: 2 real train400 images.
- Runtime: 2.75 s.
- Peak MPS driver memory: 5,937,463,296 bytes, about 5.53 GiB.
- Trainable parameters:
  - LoRA: 73,728.
  - YOLO head + quality branch: 776,774.
  - Total: 850,502.
- Nonzero gradients:
  - LoRA: 48/48 tensors.
  - Head: 26/26 tensors.
- Quality loss after corrected normalization:
  - Image 1: 0.5512.
  - Image 2: 0.1825.
- No non-finite losses.

Tests:

- Added/updated tests for quality output shape, quality gradients, score
  ordering, NMS survivor changes, and class-only state-dict compatibility.
- Targeted test command:
  `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests/test_dinov3_yolo_head.py tests/test_detection_evaluation.py -q`
  passed 21 tests.
- Full test suite:
  `PYTHONPATH=src .venv-dinov3/bin/python -m pytest tests -q`
  passed 71 tests.

Conclusion:

- The quality-aware YOLO-style head is ready for a controlled train400 pilot.
- Recommended next pilot: same train400/val600 protocol as the previous
  YOLO-style run, but enable `--include-quality` and evaluate with both
  class-only and `class_quality` score modes.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_yolo_quality_branch_implementation_2026-10-13.md`.
- NMS verification:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis/nms_ranking_verification_subset30.json`.
- Smoke summary:
  `outputs/dinov3_lora_yolo_head/smoke_quality_train2_seed20261013_v2/summary.json`.

## DINOv3 ViT-S LoRA YOLO-Style Quality Branch Train400 Control

Completed the controlled train400 continuation experiment for the YOLO-style
quality branch. No train5000 training was run.

Setup:

- Both runs started from the original YOLO-style train400 checkpoint:
  `outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt`.
- Run A continued the class-only head for 2 additional epochs.
- Run B reused the same LoRA backbone and shared YOLO head weights, then added
  a freshly initialized quality branch for 2 additional epochs.
- Dataset: existing train400 subset.
- Evaluation: existing val600 subset.
- Resolution: short side 672.
- Batch size 1, gradient accumulation 4, seed 20261013, MPS.

Quality loss balance check:

- On a deterministic 50-image train400 diagnostic, raw full-grid BCE would have
  been 99.77% dominated by background negatives.
- Updated optional quality loss to use:
  `positive_mean_bce + quality_negative_weight * negative_mean_bce`.
- Existing class-only behavior remains backward-compatible when quality is not
  enabled.

Training:

| Run | Runtime | Optimizer steps | Peak MPS driver memory | Notes |
| --- | ---: | ---: | ---: | --- |
| A class-only continuation | 1185.4 s | 200 | 11,645,206,528 bytes | Finite losses; LoRA/head gradients nonzero |
| B quality branch | 912.2 s | 200 | 11,657,707,520 bytes | Finite losses; LoRA/head/quality gradients nonzero |

Val600 results:

| Model | Scoring | AP | AP50 | AP75 | AR@100 | Recall @ FP<=0.425 | FP/img | R20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original YOLO train400 e2 | class | 0.1339 | 0.5094 | 0.0171 | 0.3198 | 0.3992 | 0.2917 | 0.3161 |
| Run A continue e2 | class | 0.1643 | 0.6128 | 0.0234 | 0.3446 | 0.5901 | 0.2967 | 0.3851 |
| Run B quality e2 | class | 0.1835 | 0.6597 | 0.0260 | 0.3366 | 0.6088 | 0.2633 | 0.4368 |
| Run B quality e2 | class * quality | 0.1870 | 0.6644 | 0.0287 | 0.3332 | 0.6542 | 0.3467 | 0.4598 |

NMS diagnostic on the same first 30 val600 images:

- Original class-only checkpoint: 12/12 high-IoU pre-NMS candidates were lost at
  NMS.
- Quality checkpoint with class scoring: 1/15 high-IoU candidates survived.
- Quality checkpoint with class*quality scoring: 1/15 high-IoU candidates
  survived.
- Predicted quality had positive but imperfect correlation with actual IoU:
  Pearson 0.2979 on quality-aware top-k candidates and 0.5303 on quality-aware
  NMS survivors.

Conclusion:

- The quality branch gives a modest real improvement over the original
  YOLO-style head and over class-only continuation.
- It does not substantially fix AP75 or the NMS/localization-ranking failure.
- Recommendation: do not scale this YOLO-style head to train5000 yet; return to
  the stronger FCOS line or test a more structural localization/ranking fix.

Artifacts:

- Report:
  `docs/reports/dinov3_lora_yolo_quality_branch_train400_control_2026-10-14.md`.
- Comparison CSV:
  `outputs/dinov3_lora_yolo_head/quality_branch_control_train400_continuation/val600_comparison.csv`.
- Quality balance diagnostic:
  `outputs/dinov3_lora_yolo_head/quality_branch_control_train400_continuation/quality_balance_train50.json`.

## DEIMv2 Feasibility Assessment

Completed a read-only feasibility assessment for DEIMv2 as the next
DETR-family/DINOv3 detector candidate. No DEIMv2 model was trained, no large
checkpoint was downloaded, and existing FCOS/YOLO code/results were not
modified.

Verified sources:

- Paper/arXiv: `https://arxiv.org/abs/2509.20787`.
- Official repository from arXiv metadata:
  `https://github.com/Intellindust-AI-Lab/DEIMv2`.
- Official Hugging Face checkpoints are public and not gated for inspected
  models:
  - `Intellindust/DEIMv2_DINOv3_S_COCO`, about 37.6 MiB.
  - `Intellindust/DEIMv2_DINOv3_L_COCO`, about 124.5 MiB.

Main findings:

- DEIMv2 is not just a simple head on our existing cached DINOv3 features. It
  uses `DINOv3STAs`, a Spatial Tuning Adapter, three feature levels, a
  HybridEncoder, a DEIM transformer decoder, 300 object queries, denoising
  queries, Hungarian matching, MAL/L1/GIoU/FGL/DDF losses, and COCO-style
  postprocessing.
- Official DEIMv2 S/M configs use distilled tiny ViT backbones, not our
  `facebook/dinov3-vits16-pretrain-lvd1689m`.
- The official DEIMv2 L config is the closest to our DINOv3 ViT-S/16 backbone:
  it uses `dinov3_vits16`, `interaction_indexes: [5, 8, 11]`, STA, and a
  224-dimensional detector feature space.
- Our existing Hugging Face DINOv3 LoRA adapters are not plug-and-play with
  DEIMv2 because DEIMv2's local DINOv3 implementation uses fused `qkv`
  attention layers, while our LoRA targets separate `q_proj`/`v_proj` modules.
- Official DEIMv2 training is CUDA-oriented and long-schedule. MPS training on
  the 16 GB Mac is unverified and risky, especially because the official code
  uses SyncBatchNorm, heavy augmentations, and CUDA-aware utilities.
- A faithful DEIMv2 experiment should reuse official components/checkpoints,
  preferably on CUDA, instead of rebuilding the architecture in our codebase.

Recommendation:

- Do not implement DEIMv2 from scratch inside the current DINOv3 FCOS code.
- If proceeding, prepare a separate CUDA/Colab-style official DEIMv2 smoke
  workflow using ForestPersons COCO exports.
- Start with inference on 10-30 val600 images using official
  `DEIMv2_DINOv3_L_COCO`, then attempt a tiny train400 CUDA smoke only if
  inference and one-class adaptation work.
- If CUDA access is limited, continue improving the proven DINOv3 LoRA + FCOS
  baseline instead.

Artifact:

- Report:
  `docs/reports/deimv2_forestpersons_feasibility_2026-10-14.md`.

## DINOv3 LoRA + FCOS Lightweight RoI Refiner

Implemented a separate lightweight second-stage RoI refinement head for the
best DINOv3 ViT-S + LoRA + single-scale FCOS checkpoint. No train400/train5000
training or val600 evaluation was run, and the original detector code path,
checkpoint, and metrics remain unchanged.

Implementation:

- Module: `src/forestpersons/dinov3_roi_refiner.py`.
- Smoke runner: `scripts/smoke_dinov3_roi_refiner.py`.
- Tests: `tests/test_dinov3_roi_refiner.py`.
- Source detector checkpoint:
  `outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt`.
- The DINOv3 backbone, LoRA adapters, and FCOS head are frozen; only the new
  refiner is trainable.
- FCOS candidates are decoded before NMS in resized-image coordinates.
- RoIAlign uses the stride-16 DINOv3 patch map, 3x3 pooling,
  `spatial_scale=1/16`, `sampling_ratio=2`, and aligned coordinates.
- Proposal matching uses IoU >= 0.50 positives, IoU < 0.30 negatives, and
  ignores the interval between them.
- The refiner predicts a person/quality score and class-agnostic Faster R-CNN
  style box deltas, trained with BCE and positive-only Smooth L1.
- Trainable parameters: 99,269.

MPS verification:

- PyTorch 2.9.0 / torchvision 0.24.0 RoIAlign forward and backward succeeded
  on MPS with finite gradients.
- A four-image real train400 smoke at short side 672 completed in 14.89 s
  (3.72 s/image).
- Peak MPS driver memory: 1.64 GiB.
- Each image supplied 1,000 pre-NMS proposals; sampled positives ranged from 8
  to 14 and sampled negatives from 114 to 120.
- All 10 refiner gradient tensors were finite and nonzero on every step; the
  frozen detector had zero trainable parameters.
- Losses were finite; total loss was 1.3143 on the first image and 1.0680 on
  the fourth. This is a smoke observation, not a performance result.

Artifacts:

- `outputs/dinov3_roi_refiner/smoke_train4_seed20261014/summary.json`.
- `outputs/dinov3_roi_refiner/smoke_train4_seed20261014/roi_refiner_smoke.pt`.
- `docs/reports/dinov3_lora_fcos_roi_refiner_implementation_2026-10-14.md`.

Conclusion:

- The lightweight refiner is ready for a controlled train400 pilot.
- Its proposal recall is bounded by frozen FCOS pre-NMS proposals, and stride-16
  patch features remain a limitation for very small people.

## DINOv3 LoRA + FCOS RoI Refiner Train400 Pilot

Completed the controlled two-epoch train400 pilot with the DINOv3 ViT-S, LoRA
adapters, and FCOS proposal head frozen. Only the 99,269-parameter RoI refiner
was optimized. Evaluation used the unchanged selected val600 subset.

Training:

- Seed 20261014, short side 672, batch size 1, AdamW learning rate 3e-4.
- 1,000 pre-NMS proposals/image for matching; 128 sampled RoIs/image.
- IoU >= 0.50 positives and IoU < 0.30 negatives.
- Epoch mean loss: 0.7145 then 0.6120.
- Classification loss: 0.1416 then 0.0778.
- Box loss: 0.5728 then 0.5342.
- 800 optimizer steps, 759.0 s total runtime, 2.27 GiB peak MPS driver memory.
- Every image had positive proposals; mean 24.74 positives and 103.26 negatives
  per image.
- Frozen-detector trainable parameters remained zero.

Val600 comparison:

- Original FCOS reference was reproduced exactly: AP 0.3587, AP50 0.8139,
  AP75 0.2379, AR@100 0.4704, recall 0.8465 and R20 0.6667 at 0.4167 FP/image.
- Box refinement with original FCOS scores: AP 0.3666, AP50 0.8248, AP75
  0.2719, AR@100 0.4734, recall 0.8545 and R20 0.6724 at 0.4183 FP/image.
- Box refinement plus RoI rescoring: AP 0.3634, AP50 0.8153, AP75 0.2675,
  AR@100 0.4737, recall 0.8158 and R20 0.5690 at 0.4183 FP/image.
- Maximum IoU-0.50 recall at threshold 0.0 was unchanged at 0.9212 for box-only
  refinement. Rescoring reached 0.9306 only at 77.67 FP/image.
- At the matched 0.425 FP budget, box-only refinement recovered 14 FCOS misses
  and lost 8 FCOS detections, net +6. For visibility 20 it recovered 4 and lost
  3, net +1.
- At the same fixed FCOS threshold 0.55, box refinement recovered 8 and lost 8,
  confirming no net recall increase without the exploratory threshold change.
- RoI rescoring recovered 18 but lost 41 at the matched budget; 20 losses were
  visibility 20. The quality score suppresses heavily occluded true positives.

Conclusion:

- The box-only branch provides a measurable localization improvement (AP75
  +0.0340) but only a small recall improvement (overall +0.0080 and R20 +0.0057
  at the matched budget).
- The train400 result does not justify train5000 scaling for the primary recall
  objective. Retain box refinement as a localization option; improving proposal
  generation or feature resolution is more promising for low-visibility recall.
- All val600-selected thresholds are exploratory because val600 was used for
  operating-point selection.

Artifacts:

- `outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k/`.
- `docs/reports/dinov3_lora_fcos_roi_refiner_train400_pilot_2026-10-14.md`.

## DINOv3 LoRA + Stride-8 FCOS Implementation

Implemented and validated an optional learned stride-8 single-level FCOS path
without changing the existing stride-16 defaults or checkpoints. No train400,
train5000, or val600 experiment was run.

Implementation:

- Added `DINOv3Stride8FCOSHead` and `Stride8FeatureAdapter` in
  `src/forestpersons/dinov3_detector.py`.
- The adapter is a depthwise 2x2 stride-2 transposed convolution followed by a
  1x1 channel mixer, initialized as exact feature replication and identity
  channel mixing.
- The new adapter has 26,400 parameters. The complete trainable configuration
  contains 73,728 LoRA, 26,400 adapter, and 524,646 reused FCOS parameters,
  totaling 624,774.
- The trained q/v LoRA weights and all compatible `proj`, `shared`,
  classification, box, and centerness FCOS tensors are reused from the best
  train5000 checkpoint. Only the three adapter tensors are new.
- FCOS target construction and decoding now accept an explicit stride, while
  retaining stride 16 as the default. Grid centers, L/T/R/B units, decoding,
  independent x/y scale inversion, clipping, and NMS were verified.
- The canonical LoRA trainer and evaluator accept `head_kind=stride8_fcos`;
  existing `fcos` and `fpn_fcos` behavior is preserved.

Assignment analysis on the deterministic train400 manifest at short side 672:

- 525 GT people across 400 images.
- Mean final positive locations increased from 7.89 at stride 16 to 8.95 at
  stride 8; stride 8 gave more positives for 148 GT.
- Visibility-20 mean positives increased from 7.19 to 8.89.
- For the 10 smallest available boxes (maximum resized side 16-32 pixels), the
  mean increased from 3.20 to 8.40.
- Stride 8 removed one overlap-related zero-positive assignment present at
  stride 16. Train400 contains no boxes with maximum resized side below 16
  pixels, so truly tiny-box coverage could not be measured empirically.

MPS smoke:

- Four real train400 images, short side 672, batch size 1, four optimizer steps.
- 5.53 s total (1.38 s/image); peak observed MPS driver memory 5.52 GiB.
- Losses were finite (0.7010-1.1498).
- Every LoRA, adapter, and reused FCOS trainable tensor received finite nonzero
  gradients on every step.
- A `stride8_fcos` checkpoint saved and reloaded through the standard loader.
- Maintained test suite: 87 passed. Repository-root collection additionally
  found two preserved standalone scripts requiring Ultralytics, which is not
  installed in `.venv-dinov3`; these were not modified.

Artifacts:

- `outputs/dinov3_lora_stride8_fcos/assignment_train400_short672/`.
- `outputs/dinov3_lora_stride8_fcos/smoke_train4_seed20261014/`.
- `docs/reports/dinov3_lora_stride8_fcos_implementation_2026-10-14.md`.

Conclusion: the implementation is ready for a controlled train400 pilot at
short side 672. Assignment density improved, but detection-metric improvement
has not yet been tested and must not be assumed.

## DINOv3 LoRA FCOS Stride-16 vs Stride-8 Train400 Comparison

Completed the controlled two-epoch comparison from the same best LoRA+FCOS
train5000 checkpoint. Both runs used train400, val600, short side 672, seed
20261014, batch size 1, gradient accumulation 4, AdamW, identical learning
rates, and unchanged focal/Smooth-L1/centerness losses. A dedicated DataLoader
RNG produced identical recorded image-order digests across both runs.

Training:

- Stride 16: losses 0.6181 then 0.6099, 4,140 positives/epoch, 200 optimizer
  steps, 910.3 s, 10.62 GiB peak observed MPS driver memory.
- Stride 8: losses 0.7311 then 0.6518, 4,697 positives/epoch, 200 optimizer
  steps, 822.1 s, 10.62 GiB peak observed MPS driver memory.
- Both runs were numerically stable with nonzero LoRA/head gradients.

Val600 COCO metrics:

- Stride 16 control: AP 0.3503, AP50 0.8116, AP75 0.2421, AR@100 0.4714.
- Stride 8: AP 0.2188, AP50 0.6693, AP75 0.0576, AR@100 0.3943.

At fixed confidence 0.60, stride 16 achieved recall 0.8211, precision 0.7555,
0.3317 FP/image, and R20 0.6207. Stride 8 achieved recall 0.7250, precision
0.5934, 0.6200 FP/image, and R20 0.5287.

On an exploratory dense threshold sweep selected on val600, both models reached
exactly 0.4167 FP/image: stride 16 at threshold 0.570 achieved recall 0.8438 and
R20 0.6724; stride 8 at threshold 0.630 achieved recall 0.6662 and R20 0.4598.
These selected thresholds are not independently validated.

Maximum IoU-0.50 recall over saved predictions down to confidence 0.001 was
0.9399 at 76.275 FP/image for stride 16 and 0.9159 at 52.177 FP/image for
stride 8. This is separate from COCO AR and has no practical FP constraint.

At the matched 0.4167 FP/image point, stride 8 recovered 22 GT missed by stride
16 but lost 155 detections, including 6 recovered and 43 lost at visibility 20.
Medium-size recall was 0.7143 versus 0.6905 at matched FP; the one COCO-small GT
was missed by both. The apparent stride-8 medium-size gain at fixed confidence
required nearly twice the FP/image and disappeared at matched FP.

Conclusion: the current stride-8 adapter does not meaningfully improve recall,
R20, AR, or localization and should not be scaled to train5000. Preserve it as
a completed negative experiment. Full details:
`docs/reports/dinov3_lora_stride8_fcos_train400_comparison_2026-10-14.md`.

## DINOv3 LoRA FCOS Hyperparameter Audit

Completed a read-only audit of the best ViT-S/16 LoRA + stride-16 FCOS
checkpoint, its frozen-head initialization, optimizer, losses, assignment,
preprocessing, and prior controlled experiments. No training or inference was
run.

Verified configuration:

- The FCOS head received 20 cached-feature epochs (100,000 updates) before two
  online LoRA+head epochs (2,500 updates); it was not trained for only two
  epochs overall.
- Online optimization used AdamW with LoRA LR 1e-4, head LR 3e-4, default
  betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 1e-4, no scheduler, no
  warmup, batch 1, gradient accumulation 4, and global gradient clipping 5.
- The best checkpoint contains model weights and history but no optimizer or
  scheduler state, so a future continuation cannot be a true optimizer resume.
- Losses are focal classification (alpha 0.75, gamma 2), log-distance Smooth L1
  (beta 0.25, weight 2), and positive-only centerness BCE (weight 1). GIoU is
  disabled.
- The head and backbone use GELU; the head has no dropout, the loaded backbone
  has no `nn.Dropout` modules and drop-path 0, and LoRA input dropout is 0.05.
- Preprocessing uses short side 672, bicubic rounding to the stride-16 grid,
  separate x/y scales, processor ImageNet normalization, no padding, and no
  data augmentation.

Important findings:

- Classification focal loss is averaged over all grid cells rather than
  normalized by positives. Train5000 has 15,955,800 grid locations and 50,600
  positives (315.3 locations per positive). At online epoch 2, classification
  contributes 0.10% of the recorded scalar objective, versus 95.35% for
  centerness. This differs from the positive-count normalization described in
  the FCOS and focal-loss papers.
- Production assignment replay found 6,431 trainer-resolved GT and 50,600
  final positive cells. Visibility-20 GT averages 7.08 positive locations,
  versus 8.26 for visibility 100. All visibility-20 GT has at least one final
  positive; one visibility-70 overlap case ends with zero.
- A confirmed lookup issue exists: the official train JSON has duplicate
  filenames, while the trainer indexes manifest rows by filename alone. This
  selects a different image ID/annotation record for 18 train5000 rows and 3
  train400 rows. Val600 has no mismatch. Existing val600 metrics remain valid,
  but future training must use image-ID-safe lookup.
- Loss decline suggests possible optimization headroom, but undertraining or
  overfitting cannot be confirmed because per-epoch validation was not saved.
  A same-LR train400 continuation did not produce a broad gain, so unchanged
  longer training is not the preferred next test.

Single recommended next experiment: after fixing manifest resolution by image
ID for both arms, run a matched two-epoch train400 A/B comparing the existing
grid-mean focal loss against focal-loss sum normalized by positive locations.
Keep alpha/gamma, model, initialization, data order, optimizer, LRs, all other
losses, and evaluation fixed. Scale to train5000 only if positive normalization
improves recall or R20 at FP/image <= 0.425 without an AR/AP collapse.

Validation: 87 maintained tests passed in 4.14 seconds with `PYTHONPATH=src`.
An initial test invocation without that source path failed at import collection;
this was an invocation issue, not a test failure in the implementation.

Full audit:
`docs/reports/dinov3_fcos_hyperparameter_audit_2026-10-14.md`.

## DINOv3 LoRA FCOS Focal-Normalization Train400 Control

Completed the controlled two-epoch comparison of grid-mean focal loss versus
focal-loss sum normalized by positive locations. Before training, manifest
resolution was corrected to use COCO `image_id` and verify `file_name`.
Corrected train400 contains 524 annotations (previous filename resolution used
525 and mismatched three rows); corrected train5000 contains 6,415 annotations
(previously 6,431 with 18 mismatched rows). Regression tests cover duplicated
filenames with distinct image IDs and filename mismatch rejection.

Both arms started from the same best train5000 LoRA+FCOS checkpoint and used
identical corrected train400 samples, order digests, seed 20261014, short side
672, AdamW parameter groups, two epochs, 200 optimizer steps, and 4,128 positive
locations per epoch. Mean training took 936.4 s and positive-count training
took 1,071.3 s. Peak observed MPS driver memory was 10.62 and 10.53 GiB.
All losses and gradients remained finite; the positive-count arm clipped 7/200
steps at the unchanged global norm 5.0, while the mean arm clipped none.

Val600 pycocotools results:

- Corrected mean continuation: AP 0.3465, AP50 0.8075, AP75 0.2247,
  AR@100 0.4685. At FP/image <= 0.425, threshold 0.570 gives recall 0.8411,
  precision 0.7135, FP/image 0.4217, and R20 0.6782.
- Positive-count continuation: AP 0.3520, AP50 0.8157, AP75 0.2327,
  AR@100 0.4697. At FP/image <= 0.425, threshold 0.560 gives recall 0.8465,
  precision 0.7172, FP/image 0.4167, and R20 0.6839.
- Original uncontinued train5000 reference: AP 0.3587, AP75 0.2379,
  AR@100 0.4704, matched-budget recall 0.8465, and R20 0.6667.

Positive-count normalization therefore improves every COCO metric and gains
four net matched GT over the corrected mean arm, including one net v20 GT,
without a higher matched FP cost. It detects 13 GT unique to that arm and loses
9 mean-arm detections. The effect is modest and does not surpass the original
reference in overall AP, AP75, or AR. A matched train5000 scale-up is cautiously
justified, but scaling only one arm would confound the loss change with the
annotation-lookup correction and additional optimization. Val600-selected
thresholds remain exploratory.

Artifacts:

- `outputs/dinov3_lora_focal_norm/train400_mean_seed20261014_from_lora5k_e2/`
- `outputs/dinov3_lora_focal_norm/train400_positive_seed20261014_from_lora5k_e2/`
- `outputs/dinov3_lora_focal_norm/train400_mean_vs_positive_seed20261014_comparison/`
- `docs/reports/dinov3_lora_fcos_focal_normalization_train400_2026-10-14.md`

Validation: 93 maintained tests passed.

## YOLO11s vs DINOv3 LoRA FCOS Paired Val600 Error Analysis

Completed a saved-prediction-only paired analysis of the fine-tuned YOLO11s
train5000 detector and the original DINOv3 ViT-S/16 + LoRA + stride-16 FCOS
train5000 detector. No inference, training, data download, or checkpoint change
was performed. Both models were evaluated on the identical val600 manifest and
749 visibility-labeled GTs with IoU-0.50 one-to-one matching. Reference
operating points were YOLO confidence 0.05 and DINO confidence 0.55.

Verified outcomes:

- Shared TP: 606; YOLO-only TP: 71; DINO-only TP: 28; shared misses: 44.
- YOLO recall 0.9039, R20 0.7701, and 0.3467 FP/image.
- DINO recall 0.8465, R20 0.6667, and 0.4167 FP/image.
- The oracle union reaches recall 0.9413 and R20 0.8448, but this uses GT to
  select detections and is not achievable fusion performance.
- Of 28 DINO-only detections, 22 are at visibility 20 or 40. Complementarity is
  real but modest; prior union/NMS and weighted-box fusion still underperformed
  YOLO at matched FP budgets.

DINOv3's 115 misses split into 59 localization misses and 56 confidence misses.
Among the 71 YOLO-only GTs, 35 are DINO localization failures and 36 have a
saved IoU>=0.50 DINO box below confidence 0.55. YOLO best candidates reach
IoU>=0.75 for 501/749 GTs versus 309/749 for DINO, confirming materially weaker
DINO localization. The largest subgroup gap is small GTs (<96-pixel maximum
side): YOLO recall 0.9000 versus DINO 0.7167.

At matched recall, DINOv3 never provides a lower tested false-alarm rate. Around
0.85 recall, YOLO requires 0.1267 FP/image versus DINO 0.4850; around 0.90,
YOLO requires 0.3467 versus DINO 5.1383. At reference points, exhaustive
unmatched prediction counts are 208 for YOLO and 250 for DINO.

Geometric unmatched-prediction categories and a stratified manual review were
saved separately. Manual crops confirm nonhuman detections for both models
(vegetation/branches for DINO; vegetation and concrete structures for YOLO),
but also reveal possible missing GT people. Therefore unmatched boxes are not
automatically labeled nonhuman. No verified person-absent local set exists;
val600 has no annotation-free images, and annotation-free train images are not
assumed person-free.

Single recommended next experiment: run a matched train5000 comparison of mean
versus positive-count focal normalization with the corrected image-ID lookup.
This targets the measured confidence/ranking deficit and follows the small but
consistent train400 gain. It is expected to be incremental and will not solve
the separate localization deficit.

Artifacts:

- `docs/reports/forestpersons_yolo11s_dinov3_lora_paired_error_analysis_2026-10-15.md`
- `scripts/analyze_yolo_dinov3_paired_errors.py`
- `outputs/error_analysis/yolo11s_vs_dinov3_lora_paired_val600_20261015/`

Validation: the analysis regenerated deterministically from saved predictions;
749 per-GT rows, 458 exhaustive unmatched-prediction rows, 32 manually reviewed
crops, and all expected plots/contact sheets were verified. The maintained test
suite passed: 93 tests in 7.89 seconds.

## DINOv3-Only and Localization Failure Analysis

Completed a saved-artifact analysis of all 28 GT people detected by DINOv3
LoRA FCOS but missed by YOLO11s, plus all 59 DINO post-NMS localization
failures. No inference, training, download, or checkpoint change was performed.
The analysis reused the val600 COCO predictions and the preserved 1,917,636-row
DINO pre-NMS candidate export with separate class and centerness probabilities.

Verified findings:

- 22/28 DINO-only GTs have visibility 20 or 40. Twenty have a valid saved YOLO
  box below confidence 0.05, five have a YOLO localization failure, and three
  have no useful saved YOLO candidate.
- DINO-only GTs are predominantly medium-size (19/28) and sitting (21/28).
  Their median area is 11,877 pixels versus 8,216 for YOLO-only GTs. No unique
  camera or weather subgroup was found.
- Of 59 DINO localization failures, 35 are scale-only by fixed geometry rules,
  23 combine center and scale error, and one is center-only. Selected boxes are
  evenly split between too large and too small (28 each), so there is no single
  directional size bias.
- The median best IoU rises from 0.438 post-NMS to 0.558 before NMS. Forty of
  the 59 failures have a raw IoU>=0.50 candidate. Of those 40, 24 are removed
  by confirmed NMS competition and 16 fall below top-300 ranking.
- The best-localized raw boxes have much lower median combined/class/centerness
  scores (0.207/0.102/0.326) than the selected post-NMS boxes
  (0.468/0.368/0.572), confirming score/localization misalignment.
- Across all 749 GTs, YOLO best saved candidates reach IoU>=0.75 for 66.9%
  versus 41.3% for DINO. At visibility 20 the rates are 47.1% versus 25.3%.
  DINO localization is weaker at every size, not only for small people.

Single recommended next experiment: add a lightweight FCOS IoU-quality scalar
used only for top-k/NMS ordering, while retaining the original FCOS confidence
for final filtering. Test first on train400 and judge it by AP75, AR@100, and
recall/R20 at FP/image <=0.425. This directly targets the 40 valid raw boxes
without repeating failed GIoU, FPN, stride-8, or global rescoring changes.

Artifacts:

- `docs/reports/dinov3_only_localization_failure_analysis_2026-10-15.md`
- `scripts/analyze_dinov3_only_localization_failures.py`
- `outputs/error_analysis/dinov3_only_localization_val600_20261015/`

Validation: 93 maintained tests passed in 4.01 seconds with
`PYTHONPATH=src .venv-dinov3/bin/python -m pytest -q tests`. An unscoped root
`pytest` invocation additionally collected `test_dinov3.py` and `test_yolo.py`
and stopped because `.venv-dinov3` intentionally lacks Ultralytics; this did
not affect the maintained `tests/` suite.

## DINOv3 FCOS Ranking Oracle and Candidate Recovery

Completed an analysis-only ranking oracle study from the preserved val600
pre-NMS FCOS candidate dump. No inference, training, dataset download, or
checkpoint change was performed.

Canonical post-processing was reproduced exactly: the 0.001 combined-score
floor, top 300 before NMS, and NMS IoU 0.50 produced the same 46,586 boxes as
the saved canonical output. Maximum box error was 0.0 pixels and maximum score
error was 4.82e-8.

Verified findings:

- Classification-only top-k raises maximum recall from 0.9212 to 0.9306 and
  AR@100 from 0.4704 to 0.4730, but gives no recall or R20 gain at
  FP/image <=0.425. Classification-only NMS ordering and centerness-based
  variants reduce matched-budget recall.
- Strict IoU-oracle top-k retains all 40 known valid raw candidates, but current
  NMS keeps only 5. Strict IoU-oracle NMS within the current top-300 recovers a
  final IoU>=0.50 candidate for 32/40. Combined oracle top-k and NMS recovers
  all 40 at the candidate floor.
- With original final confidence, the combined oracle detects only 4/40 at the
  target FP budget. Ranking alone is therefore insufficient.
- Combined oracle ordering raises maximum recall to 0.9746, maximum R20 to
  0.9310, and AR@100 to 0.6415, but AP falls because the recovered boxes retain
  low original confidence.
- A strict, GT-informed one-candidate-per-GT ceiling reaches AP 0.6436, AP75
  0.7228, AR@100 0.6441, recall 0.9746, and R20 0.9310. This is explicitly a
  non-deployable geometry ceiling, not a detector result.

Decision: proceed with a controlled train400 FCOS IoU-quality-head pilot, but
reject a rank-only design. Predicted quality must participate in top-k, NMS,
and final confidence calibration, with mixing selected on training-only data.
Judge success by AP/AR and recall/R20 at FP/image <=0.425; retaining more boxes
alone is insufficient.

Artifacts:

- `docs/reports/dinov3_fcos_ranking_oracle_2026-10-15.md`
- `scripts/analyze_fcos_ranking_oracle.py`
- `outputs/error_analysis/dinov3_fcos_ranking_oracle_val600_20261015/`

Validation: ranking-oracle artifact assertions passed and all 93 maintained
tests passed in 5.05 seconds.

## DINOv3 LoRA FCOS IoU-Quality Head Train400 Control

Completed a matched two-epoch train400 continuation testing an optional
161-parameter positive-only IoU-quality branch. Both arms started from the same
original train5000 LoRA+FCOS checkpoint, used corrected image-ID lookup,
positive-count focal normalization, seed 20261015, identical sample order,
200 optimizer steps, and unchanged stride-16/top-300/NMS settings. Control
training took 1,188.7 s with 10.53 GiB peak observed MPS driver memory; quality
training took 1,142.8 s with 10.62 GiB. Losses and gradients remained finite.

Quality-score powers and thresholds were selected on a deterministic 100-image
holdout from train5000 excluding train400. Calibration selected the original
score (`quality_power=0`); the best nonzero ablation was power 0.25. On val600,
the matched control obtained AP 0.3510, AP75 0.2260, AR@100 0.4692, recall
0.8478 and R20 0.6897 at 0.4117 FP/image. The quality model with its original
score obtained AP 0.3526, AP75 0.2312, AR@100 0.4677, recall 0.8478 and R20
0.6839 at the same FP rate. Quality-aware scoring obtained AP 0.3485, AP75
0.2234, AR@100 0.4682, recall 0.8451 and R20 0.6782 at 0.4217 FP/image.

Predicted quality correlates moderately with actual candidate IoU (Pearson
0.496 and Spearman 0.499 on 98,923 overlapping raw candidates), but does not
improve the known failure set. For the prior 40 cases, both legacy and
quality-aware scoring retain 33 through top-300, 15 after NMS, and six at their
matched-budget thresholds. Quality-aware scoring recovers one overall GT and
loses three versus control. Calibration-locked thresholds also exceed the
val600 FP budget, indicating score-distribution shift.

Decision: do not scale this multiplicative IoU-quality formulation to
train5000. It learns a real signal but does not improve top-k/NMS recovery,
confidence calibration, AP/AR, recall, or R20 at the operational FP budget.

Artifacts:

- `docs/reports/dinov3_lora_fcos_iou_quality_train400_2026-10-15.md`
- `outputs/dinov3_lora_fcos_quality/`
- `scripts/prepare_fcos_quality_calibration.py`
- `scripts/analyze_fcos_quality_head.py`
- `scripts/summarize_fcos_quality_experiment.py`

## Official DEIMv2-S Zero-Shot ForestPersons Pilot

Evaluated the official `Intellindust/DEIMv2_DINOv3_S_COCO` checkpoint locally
on Apple M4 MPS without training or checkpoint conversion. The exact official
`PyTorchModelHubMixin` wrapper and repository commit `1d2ca421` were used in a
separate `.venv-deimv2` environment. Direct Transformers `AutoConfig` loading
does not support this exact custom model repository, even though recent
Transformers releases contain the DEIMv2 architecture.

The official S model has 9,779,838 parameters and uses a 192-dimensional
ViT-Tiny distilled from DINOv3-S, Spatial Tuning Adapter features at strides
8/16/32, and a four-layer 300-query decoder. It is not the same DINOv3 ViT-S
backbone used by our custom detector. The 39.4 MB official weights and config
were pinned and hashed. Research evaluation is permitted; commercial use is
not permitted without a separate license.

MPS inference succeeded on smoke4, val32, and all val600 images. Val600 used
749 GT persons and produced 14,381 saved person-class query predictions. Local
COCO metrics matched pycocotools exactly:

- AP 0.3408, AP50 0.5693, AP75 0.3544, AR@100 0.6016.
- At the exploratory val600-selected confidence 0.175: recall 0.5033,
  precision 0.6294, FP/image 0.3700, R20 0.1782, R40 0.4059, R70 0.5907,
  and R100 0.7689.
- Maximum saved-query recall at score 0 was 0.9146 but required 22.83
  FP/image; this is threshold-specific recall, not COCO AR.
- Steady MPS latency was 0.0790 s/image, val600 inference took 47.63 s, and
  observed peak MPS driver allocation was 1.16 GiB. CPU fallback was not used.

Conclusion: zero-shot DEIMv2-S is not competitive with the ForestPersons-
trained YOLO11s or DINOv3 LoRA FCOS models at the operational FP budget,
especially at 20% visibility. It nevertheless offers a stronger localization
foundation than the custom FCOS head: AR@100 is 0.6016 and AP75/AP50 is 0.623,
versus 0.4704 and 0.292 for LoRA FCOS. A controlled train400 DEIMv2-S
fine-tuning feasibility pilot is justified; a DEIMv2-L val32 smoke is also
scientifically useful because L uses DINOv3 ViT-S.

Artifacts:

- `docs/reports/deimv2_s_zero_shot_forestpersons_2026-10-15.md`
- `outputs/deimv2_zero_shot/`
- `scripts/evaluate_deimv2_forestpersons.py`
- `src/forestpersons/deimv2_adapter.py`

## DEIMv2-S Fine-Tuning Feasibility on Apple M4

Completed a one-class DEIMv2-S training feasibility audit and two-image
gradient smoke. No multi-epoch training, new checkpoint download, or baseline
modification was performed.

Verified findings:

- The official trainer is CUDA-oriented, but the active detector and losses do
  not require a compiled CUDA extension. The active deformable attention uses
  PyTorch `grid_sample`.
- PyTorch 2.9 MPS cannot run native `grid_sampler_2d_backward`. Training works
  with the documented `PYTORCH_ENABLE_MPS_FALLBACK=1` CPU fallback.
- The official distribution integral's valid one-dimensional `F.linear`
  triggers an MPSGraph shape assertion during training. An algebraically
  identical MPS weighted-sum implementation was added and verified against
  official CPU values and gradients; upstream source was not modified.
- ForestPersons category ID 1 is mapped to internal label 0. The pretrained
  COCO person row initializes all one-class score heads and the denoising
  class embedding. All other compatible weights are retained.
- Existing data remain valid by image ID: train400 has 400 readable images and
  524 raw annotations; the one-class train5000 export has 5,000 images and
  6,415 annotations; val600 has 600 images and 749 valid annotations.
- Frozen-transformer smoke: 4,202,225 trainable parameters, total loss
  30.1222, all finite gradients, 1.59 s step, 1.34 GiB peak observed MPS
  driver memory.
- Full-model smoke: 9,688,433 trainable parameters, total loss 24.6675, all
  finite gradients, 0.88 s step, 2.45 GiB peak observed MPS driver memory.
  One optimizer step changed 479 trainable parameter tensors.
- Full-model train400 fine-tuning is memory-feasible at batch 1 in float32.
  A two-epoch pilot is estimated at roughly 15-35 minutes including normal
  overhead, but this estimate requires a 20-image timing check.
- DEIMv2-L zero-shot is likely locally feasible: it uses the actual DINOv3
  ViT-S/16 backbone, has 32.2M parameters and 96.7 GFLOPs, and does not need
  backward. No L weights were downloaded or executed.
- The DEIMv2 License is non-commercial. LogXY needs explicit commercial
  authorization before product integration or deployment; third-party license
  terms remain applicable.

Decision: a controlled train400 pilot is technically justified, using full
one-class fine-tuning, batch 1, accumulation 4, float32, official losses/LRs,
and explicit MPS CPU fallback. This audit proves gradient execution only, not
convergence or detection improvement.

Artifacts:

- `docs/reports/deimv2_s_mps_finetuning_feasibility_2026-10-15.md`
- `scripts/smoke_deimv2_finetune_mps.py`
- `outputs/deimv2_local/finetune_smoke/`
- `src/forestpersons/deimv2_adapter.py`

Validation: 105 maintained tests passed in 3.83 seconds. The smoke script and
adapter passed Python compilation, and the saved MPS smoke JSON validated.

## DEIMv2-S Train400 Full-Model Pilot Readiness

Prepared, but did not start, the controlled two-epoch DEIMv2-S train400 pilot.
The full command is gated by explicit non-commercial-research and supervisor
confirmation flags because the DEIMv2 license prohibits commercial use.

Implemented:

- `scripts/train_deimv2_forestpersons.py`: guarded full-model MPS trainer with
  batch 1, accumulation 4, float32, exact official one-class losses, four
  official optimizer parameter groups, finite checks, atomic resumable
  checkpoints, JSONL logs, loss history, and memory/timing monitoring.
- `src/forestpersons/deimv2_training.py`: image-ID-safe manifest resolution,
  clipped xywh-to-normalized-cxcywh conversion, official criterion/optimizer
  groups, pilot-scaled upstream flat-cosine scheduler, RNG state, and atomic
  checkpoint utilities.
- Fine-tuned one-class checkpoint loading in the existing DEIMv2 adapter and
  evaluation runner. Zero-shot behavior remains unchanged.
- Train-derived calibration locking through
  `scripts/select_deimv2_calibration_threshold.py`; it rejects val summaries.
- Machine-readable configuration at
  `experiments/deimv2_local/configs/forestpersons_train400_fullmodel_seed20261015.json`.

Verified data and initialization:

- train400: 400 images and 524 corrected annotations, resolved by image ID.
- calibration100: 100 train images with zero filename overlap with train400;
  calibration is excluded from gradient updates.
- All ForestPersons categories map to internal person label 0. Eleven
  class-dependent tensors are initialized from the COCO person row.
- Actual optimizer groups use LR 2.5e-5 for DINOv3-distilled backbone decay
  and no-decay parameters, LR 5e-4 for detector decay/no-decay parameters, and
  zero weight decay for normalization/BN/bias parameters.

Completed only a bounded 20-image/five-update technical timing validation:

- Mean loss 34.3155; all losses and gradients remained finite.
- One expected `grid_sampler_2d_backward` CPU-fallback warning; no OOM.
- Warm steady-state accumulated-step median 1.906 s.
- Peak observed MPS driver memory 2.583 GiB.
- Projected 200-step model loop 6.35 minutes; practical two-epoch estimate
  8-12 minutes plus calibration and val600 evaluation.
- Fine-tuned checkpoint reload and two-image evaluation smoke passed. Its
  two-image metrics are not research results.
- Resumable checkpoint size is approximately 112 MiB; planned outputs need
  roughly 450 MiB.

Full train400 status: **not started**. Exact post-approval training and
evaluation commands are recorded in
`docs/reports/deimv2_s_train400_readiness_2026-10-15.md`.

Artifacts:

- `docs/reports/deimv2_s_train400_readiness_2026-10-15.md`
- `outputs/deimv2_finetune/readiness_benchmark20_seed20261015/`
- `outputs/deimv2_finetune_readiness_benchmark20_seed20261015.log`

Validation: 111 maintained tests passed in 4.00 seconds. Syntax/configuration
checks, checkpoint reload, evaluation smoke, and the full-run supervisor gate
passed. No full train400 output directory was created.

## YOLO11s + DINOv3 FCOS Selective Fusion

Completed a prediction-level selective-fusion study using the existing
train5000 YOLO11s and DINOv3 ViT-S LoRA FCOS checkpoints. No detector training,
checkpoint modification, or dataset download was performed.

- Verified val600 saved prediction floors: YOLO 0.0010 with 1,737 boxes below
  0.05; DINOv3 actual minimum 0.0957. Existing val600 inference was reused.
- Reproduced the canonical fixed points exactly: YOLO conf 0.05 recall 0.9039,
  precision 0.7650, 0.3467 FP/image; DINO conf 0.55 recall 0.8465, precision
  0.7172, 0.4167 FP/image.
- Created calibration200 from the official validation split outside val600:
  200 local images, 50 targeted per visibility, 252 GT, zero val600 overlap.
- Calibration selected selective addition gates (YOLO >= 0.075, DINO >= 0.55,
  duplicate IoU 0.30) and supported YOLO rescue gates (YOLO keep >= 0.10,
  rescue >= 0.01, DINO support >= 0.50, support IoU >= 0.30).
- Calibration-locked fusion transferred with higher recall but exceeded the
  target FP budget on val600: method A recall 0.9226/R20 0.8161 at 0.4633
  FP/image; method B recall 0.9212/R20 0.7989 at 0.4383 FP/image.
- Exploratory val600 matched-budget method B reached recall 0.9212 and R20
  0.7989 at exactly 0.425 FP/image versus YOLO recall 0.9119 and R20 0.7759 at
  0.4100 FP/image. It gained ten GT, lost three and added nine FP relative to
  matched-budget YOLO. These thresholds are val600-selected and not unbiased.
- Selective A/B reduced AP and AR@100 relative to YOLO. Calibrated support
  ranking raised AR@100 to 0.6409 but reduced operational recall.
- Decision: promising exploratory complementarity, but no confirmed operational
  fusion win. Freeze parameters and confirm once on an independent held-out
  validation subset before any complex fusion training.

Artifacts:

- `docs/reports/forestpersons_yolo_dinov3_selective_fusion_2026-10-15.md`
- `src/forestpersons/selective_fusion.py`
- `scripts/create_fusion_calibration_subset.py`
- `scripts/evaluate_selective_fusion.py`
- `tests/test_selective_fusion.py`
- `outputs/fusion/yolo_dino_selective_20261015/selective_eval_verified/`

Validation: 116 maintained tests passed in 3.92 seconds. Python compilation,
JSON/CSV integrity checks, canonical metric reproduction and plot rendering all
passed. Bare repository-wide pytest discovery additionally collects vendored
DEIMv2 tests and legacy root smoke scripts; it stops on missing optional
`tensorboard`/`ultralytics` dependencies before reaching the maintained suite.

## Independent YOLO11s + DINOv3 Selective-Fusion Holdout

Completed a frozen, independent evaluation on 1,200 additional official
ForestPersons validation images. No training, checkpoint changes, calibration
refit, threshold adjustment, or fusion-parameter tuning was performed.

- Holdout: 1,200 readable images and 1,461 GT; 300 target images per visibility;
  zero filename/image-ID/SHA-256 overlap with val600, fusion calibration200,
  train400, train5000, calibration100 and earlier smoke manifests.
- Limitation: all 50 selected sequence folders also occur in protected
  manifests, so the holdout is exact-image-disjoint but not scene-disjoint.
- Frozen YOLO at conf 0.04: recall 0.9062, R20 0.7859, precision 0.7662,
  0.3367 FP/image, AR@100 0.6373 (1,324 TP / 404 FP / 137 FN).
- Frozen Selective A: recall 0.9172, R20 0.8035, precision 0.7334,
  0.4058 FP/image, AR@100 0.6238. It rescued 24, lost 8, netted +16 GT and
  added 83 FP versus YOLO.
- Frozen Selective B: recall 0.9185, R20 0.8035, precision 0.7599,
  0.3533 FP/image, AR@100 0.6260. It rescued 23, lost 5, netted +18 GT and
  added 20 FP versus YOLO.
- Both fusion methods satisfy the strict 0.425 FP/image limit without holdout
  tuning. Paired 10,000-replicate bootstrap overall recall differences exclude
  zero: A [+0.0035,+0.0184], B [+0.0049,+0.0202]. R20 differences are positive
  but their intervals include zero.
- AP/AR remain lower than YOLO, so fusion is a recall-oriented operating mode.
  Keep Selective B; deprioritize A and do not start complex fusion training.

Artifacts:

- `docs/reports/forestpersons_yolo_dinov3_selective_fusion_independent1200_2026-10-15.md`
- `outputs/fusion/yolo_dino_selective_independent1200_20261015/`
- `outputs/subsets/forestpersons_val_fusion_independent_1200_seed20261015_manifest.json`
- `experiments/fusion/configs/yolo_dino_selective_frozen_20261015.json`
- `scripts/create_fusion_independent_holdout.py`
- `scripts/evaluate_selective_fusion_holdout.py`

Validation: 118 maintained tests passed in 4.97 seconds. The two new paired
bootstrap tests passed; both holdout scripts compiled; manifest, frozen-config,
summary, comparison CSV and experiment-registry structure checks passed.

## Weitefeld Cross-Dataset Pilot200

Completed a local, zero-shot transfer pilot using the frozen ForestPersons
YOLO11s, DINOv3 ViT-S LoRA + FCOS, and Selective Fusion B systems. No model was
trained or retuned, and no bulk strip archive was downloaded.

- Audited all 34,424 official metadata rows. After exact deduplication there
  are 33,367 labels, including 1,741 class-3 candidate labels (1,740 valid) in
  1,416 images. The paper's 405 physical findings span all classes; `data.txt`
  does not expose a finding ID, so exact unique class-3 identities are unknown.
- Verified category IDs and the official lower-left box conversion against the
  web viewer and released benchmark notebook: `(x, y-height, x+width, y)`.
- Built deterministic seed-20261015 pilot200: 127 class-3 track-view images,
  73 unannotated-context images (explicitly unknown, not negative), 155 valid
  class-3 boxes, all 15 strips, and 48 provisional strip-track proxies.
- Selectively downloaded and verified 200/200 official 8416x6032 JPEGs through
  direct URLs supplied by the official strip indexes. Transfer: 2.615 GiB in
  567.6 seconds. No 144.1 GB strip archive was downloaded.
- Ran full-image and equal-coverage 2048/256 tiled inference. YOLO full/tiled
  latency was 0.407/1.909 s per image; DINO was 0.769/4.134 s per image.
- At frozen ForestPersons operating points, all systems localized 0/155
  class-3 boxes at IoU 0.50; AP/AR are zero. This was not a coordinate bug:
  low-score tiled candidates maxed at IoU 0.254 (YOLO) and 0.451 (DINO).
- Median class-3 size is 41x37 source pixels. Full-image resizing reduces this
  to roughly 3-5 model-input pixels. Tiling improves overlap but predicted
  boxes remain much larger than official tiny boxes.
- Unmatched outputs per image are nondefinitive because labels are incomplete,
  and class 3 includes suspected people later described by police as non-person
  objects or no abnormality. The result is a class-3 localization diagnostic,
  not clean person precision/recall.
- Decision: do not automatically download the remaining 800 images. First
  obtain/adjudicate explicit person views and predeclare a tiny-target metric
  such as center-in-box alongside IoU, preferably with physical finding IDs.

Artifacts:

- `docs/reports/weitefeld_cross_dataset_pilot200_2026-10-15.md`
- `outputs/weitefeld/pilot200_manifest.json`
- `outputs/weitefeld/pilot200_predictions/`
- `outputs/weitefeld/pilot200_evaluation/`
- `outputs/weitefeld/pilot200_analysis/`
- `data/weitefeld/images/pilot200/download_verification.json`
- `src/forestpersons/weitefeld.py`
- `scripts/prepare_weitefeld_pilot.py`
- `scripts/download_weitefeld_pilot.py`
- `scripts/predict_weitefeld_pilot.py`
- `scripts/evaluate_weitefeld_pilot.py`
- `scripts/analyze_weitefeld_pilot.py`

Validation: 126 maintained tests passed in 5.43 seconds. All six Weitefeld
scripts/modules compiled; deterministic manifest regeneration was byte-for-byte
identical; 200/200 images reopened successfully at 8416x6032 with zero missing
or corrupt files; registry CSV structure remained valid.

## Weitefeld Annotation Verification and Tiny-Object Localization

Completed an analysis-only audit of all 155 class-3 boxes in the existing
pilot200. No images were downloaded, no inference was rerun, and no model,
checkpoint, fusion rule, or threshold was changed.

- Generated a reviewer-ready CSV, local HTML gallery, 155 individual
  full/context/target panels, and 39 contact-sheet pages. No annotation was
  automatically marked confirmed person.
- Police-comment evidence: 0 confirmed persons; 134 provisional non-person
  boxes; 18 uncertain person/cyclist descriptions explicitly marked not
  checked; and 3 uncertain rows with no police comment.
- Found annotation multiplicity: 25 images contain multiple class-3 boxes;
  11/31 within-image annotation pairs overlap at IoU >=0.50. Missing physical
  finding IDs prevent resolving whether these are distinct targets.
- Full-image preprocessing reduces the median 41x37 annotation to 3.12x2.81
  YOLO pixels or 4.57x4.12 DINO pixels. Existing tiling raises these to
  12.81x11.56 and 13.45x12.14 pixels.
- Frozen operating points remain zero at IoU 0.50. Tiled diagnostic recall is
  3/155 YOLO at IoU 0.10 and 1/155 DINO at IoU 0.10/0.25; fusion is zero.
- At the non-deployable saved-candidate floor 0.001, tiled YOLO has 10/155 GT
  with best IoU >=0.10 and 2/155 >=0.25; DINO has 24/155 and 7/155. Neither has
  an IoU >=0.50 candidate.
- DINO shows better regional localization: median tiled selected-candidate
  center error 104 pixels versus YOLO 424 pixels. However, DINO's median
  overlapping candidate is 44.4x the annotation area, and 125/155 cases are
  categorized as overlap with severe scale mismatch. YOLO's dominant failure
  is no nearby candidate (113/155).
- DINO's advantage is absent for the 58 COCO-small source boxes (0/58 at IoU
  >=0.10) and concentrated in medium/large boxes. It is therefore not evidence
  that DINO solves Weitefeld's smallest person candidates.
- Decision: Weitefeld is not suitable for confirmed-person recall in its
  current form. Manually review the 21 uncertain boxes before any bounded
  finer-tiling diagnostic. Use an exhaustively labeled person dataset such as
  HERIDAL or SARD for the next quantitative cross-dataset benchmark, subject to
  a separate license/split audit. Do not download the remaining 800 images.

Artifacts:

- `docs/reports/weitefeld_annotation_verification_tiny_localization_2026-10-16.md`
- `scripts/review_weitefeld_annotations.py`
- `tests/test_weitefeld_review.py`
- `outputs/weitefeld/annotation_review_verified_20261016/`

Validation: 129 maintained tests passed in 5.53 seconds. The analysis script
compiled; all 155 CSV rows, 155 individual panels, 39 contact sheets, summary
JSON, plots, overlap-pair table, and HTML index were generated; registry CSV
structure and review-status vocabulary were checked.

## SARD Above-Canopy RGB Cross-Dataset Pilot

Completed a local, frozen-model benchmark on the complete 200-frame SARD v1.0
mountain scenario. HERIDAL was audited first but not used because its original
host refused connections and current official files, split counts, and license
could not be verified. No model was trained or retuned.

- Sparse-checked out the official SARD v1.0 mountain directory at commit
  `8961f735580b9127e24f1a8bad3ad8bbbe1a3d03`: 200 3840x2160 images, 200 label
  files, 305 valid person boxes, 101 positive frames, and 99 empty-label frames.
- Verified every image/label pair and visualized sample boxes. The release
  files differ by one frame from the README's stated 100 positive/100 negative
  mountain split.
- The source is sequence-correlated: dHash identifies seven clip segments and
  median adjacent-frame distance is 7/256 bits. Results are not 200 independent
  scenes.
- At the frozen operating points, YOLO11s achieved recall 0.0787, AP 0.0052,
  AR@100 0.0482, and apparent 8.575 FP/image. DINOv3 LoRA FCOS achieved recall
  0.1311, AP 0.0160, AR@100 0.1000, and apparent 0.800 FP/image. Frozen Fusion B
  achieved recall 0.0721, AP 0.0048, AR@100 0.0249, and apparent 5.750 FP/image.
- Complementarity: both 17, YOLO-only 7, DINO-only 23, both-miss 258. Fusion B
  rescued 1 YOLO miss but lost 3 YOLO true detections.
- All 207 COCO-medium source boxes were missed at fixed thresholds. Median
  people shrink to 9.17x18.0 pixels at YOLO input and 17.11x33.6 at DINO input.
- A separate post-hoc 30-frame tiling diagnostic on the 150-person hardest
  clip recovered only 3 YOLO and 0 DINO detections, while raising apparent
  FP/image to 44.1 and 2.8 respectively. Do not mix this with the primary result.
- SARD Zenodo is open access but specifies only `other-open`; no concrete
  license text exists in the repository. Treat the data as research-only until
  redistribution and company-use rights are clarified.

Artifacts:

- `docs/reports/sard_above_canopy_rgb_cross_dataset_pilot_2026-10-16.md`
- `src/forestpersons/sard.py`
- `scripts/prepare_sard_pilot.py`
- `scripts/predict_sard_pilot.py`
- `scripts/evaluate_sard_pilot.py`
- `scripts/analyze_sard_pilot.py`
- `tests/test_sard.py`
- `outputs/sard/`

Decision: do not tune fusion or detectors on the seven correlated SARD clips.
The next defensible above-canopy benchmark is the official HERIDAL held-out
split, after official access and licensing are resolved.

Validation: 135 maintained tests passed in 5.60 seconds. All five new SARD
modules/scripts compiled; 200/200 images and labels were verified; prediction
outputs cover all 200 image IDs; pycocotools evaluation completed for all three
methods; primary and tiled galleries, comparison CSVs, per-GT diagnostics,
segment tables, and recall-FP curves were saved.

## 2026-10-16 — DINOv3 Multi-View Feature Fusion Feasibility and Pilot

Cancelled the unfinished custom DINOv3 DETR-head experiment before training.
The project directory is not a Git repository, so cleanup was verified by
filesystem inspection rather than Git status/diff. Removed only the three new
unfinished DETR files: `src/forestpersons/dinov3_detr.py`,
`scripts/train_dinov3_lora_detr.py`, and
`scripts/evaluate_dinov3_lora_detr.py`. No DETR checkpoint/result/process
existed; historical MMDetection preparation and all validated experiments were
preserved.

Dataset audit:

- Weitefeld flight strips have no local camera calibration and class 3 is not
  confirmed-person GT (0 confirmed, 134 explicit negative, 21 uncertain), so
  they cannot support person-recall multi-view evaluation.
- SARD mountain frames are sequence-correlated and annotated, but lack local
  camera poses/calibration.
- ForestPersons contains adjacent frames in 262 train and 57 validation
  sequence folders with confirmed per-frame person boxes, but no official
  camera poses, homographies, or track IDs.
- The official person-search AOS release has 18 flights, KML flight metadata,
  and rectification code. Its smallest relevant packages are approximately
  4.5 GB low-resolution RGB and 8.0 GB prepared YOLO data, so no download was
  made without approval.

Implemented an exploratory inference-only pilot using the existing DINOv3
ViT-S rank-4 q/v LoRA + stride-16 FCOS train5000 checkpoint. Eight unique
ForestPersons validation sequences supplied eight three-view groups (24 views,
3 visibility-20 and 5 visibility-40 reference people). ORB/RANSAC image
homographies were retained only with strong image registration and projected
source/reference GT IoU >=0.50. GT therefore qualifies correspondence and makes
the sample selection-biased; it is not used in homography estimation or model
inference.

The shared encoder extracts each view at short side 672. Source patch features
are warped to reference patch centers, averaged only where valid, L2-normalized,
and passed through the unchanged FCOS head. At fixed confidence 0.55 and NMS
IoU 0.50:

- Single reference: 7 TP, 3 FP, 1 FN; recall 0.875; precision 0.700;
  FP/image 0.375; AP 0.3703; AP50 0.8500; AP75 0.2145; AR@100 0.4375.
- Aligned three-view fusion: 8 TP, 0 FP, 0 FN; recall/precision 1.000;
  FP/image 0; AP 0.4215; AP50 1.0000; AP75 0.2145; AR@100 0.5000.
- Group 08 (visibility 40) was rescued, with best candidate IoU increasing
  from 0.5525 to 0.7464; no fixed-threshold detections were lost.
- Runtime was 0.471 s/group single-view and 1.524 s/group end-to-end
  three-view; peak MPS driver allocation was about 1.63 GiB.

These eight selected cases prove technical feasibility, not general benefit.
AP75 did not improve and best-candidate IoU worsened in three groups. The next
step is an approved official AOS data/metadata audit and a larger
geometry-grounded inference-only benchmark before any fusion training.

Artifacts:

- `src/forestpersons/multiview.py`
- `scripts/prepare_forestpersons_multiview_pilot.py`
- `scripts/evaluate_dinov3_multiview_pilot.py`
- `tests/test_multiview.py`
- `docs/reports/dinov3_multiview_feature_fusion_pilot_2026-10-16.md`
- `outputs/multiview/forestpersons_val_pilot8_seed20261016/`

Validation: 140 maintained tests passed in 3.96 seconds, including five new
multi-view geometry/fusion tests. New scripts compiled successfully and the
48-row experiment registry parsed with the expected nine columns. An
unrestricted repository-root pytest command additionally collected vendored
DEIMv2 tests and legacy root smoke scripts, which require optional
`tensorboard`/`ultralytics` packages not installed in `.venv-dinov3`; the
maintained `tests/` suite is unaffected.

## 2026-10-17 — ForestPersons Multi-View Feature Fusion Benchmark49

Scaled the inference-only DINOv3 multi-view study without training, checkpoint
changes, downloads, or threshold tuning. A new selector used only validation
sequence/frame metadata and image-only ORB/RANSAC registration; annotations,
visibility and detector outputs were excluded from group selection.

Selection and integrity:

- Requested 50 groups; 49 three-view groups passed unchanged thresholds,
  yielding 147 unique images from 49 unique sequences.
- One earlier and one later source frame were required for each reference.
- Logged 640 rejected reference attempts and 4,690 source registration
  attempts with explicit rejection reasons.
- Zero train/validation filename or sequence overlap and zero exact-image
  overlap with the prior pilot8. Six sequence names overlap the pilot8 scenes.
- The frozen references contain 85 unique official annotations: visibility
  counts 20/40/70/100 = 12/15/22/36. Twenty-three reference frames contain
  multiple people. One-to-one reference-view matching prevents duplicate GT
  credit.

Frozen evaluation used the existing DINOv3 ViT-S q/v LoRA + stride-16 FCOS
train5000 checkpoint, short side 672, confidence 0.55, and NMS/match IoU 0.50.
Equal valid-region feature averaging was applied before the unchanged FCOS
head.

| Metric | Single view | Multi-view mean |
|---|---:|---:|
| TP / FP / FN | 74 / 12 / 11 | 50 / 21 / 35 |
| Recall | 0.8706 | 0.5882 |
| R20 / R40 / R70 / R100 | 0.6667 / 0.8000 / 0.9091 / 0.9444 | 0.6667 / 0.6000 / 0.5909 / 0.5556 |
| FP/image | 0.2449 | 0.4286 |
| AP / AP50 / AP75 | 0.4142 / 0.8778 / 0.3099 | 0.2291 / 0.6124 / 0.0628 |
| AR@100 | 0.5094 | 0.3600 |

Fusion rescued 2 annotations and lost 26, net -24. The sequence-cluster paired
bootstrap recall difference is -0.2824 with 95% interval
[-0.4142, -0.1591]. R20 is unchanged but based on only 12 people.

Post-selection correspondence analysis found that global image registration
does not reliably align people: 23/85 GT align at IoU>=0.50 in both source
views, 51/85 in at least one, and 34/85 in neither. In the neither-source group,
single recall is 0.882 versus fused recall 0.235, accounting for a net loss of
22 detections. Moving people, pose changes, depth parallax and dynamic
occlusion remain after static forest background alignment. Naive equal feature
averaging should not be trained or scaled.

Runtime was 0.461 s/group for single-reference inference and 1.513 s/group for
the complete three-view pipeline; peak MPS driver allocation was about 2.26
GiB.

Recommended next step: with approval, audit official AOS
`RAW_lowresRGB.zip` (4,516,263,778 bytes) for KML poses/rectification and
`YOLO.zip` (7,955,819,284 bytes) for labels and split mapping. Verify raw-frame,
camera-pose, rectified-image and annotation identity before implementing a
target-depth-aware or correspondence-aware fusion method. No AOS data were
downloaded.

Artifacts:

- `docs/reports/dinov3_multiview_feature_fusion_benchmark49_2026-10-17.md`
- `scripts/prepare_forestpersons_multiview_benchmark.py`
- `scripts/evaluate_dinov3_multiview_benchmark.py`
- `scripts/analyze_forestpersons_multiview_correspondence.py`
- `scripts/summarize_forestpersons_multiview_benchmark.py`
- `tests/test_multiview_benchmark.py`
- `outputs/multiview/forestpersons_val_benchmark50_seed20261016/preparation/`
- `outputs/multiview/forestpersons_val_benchmark49_seed20261016/`

Validation: 143 maintained tests passed in 4.65 seconds. All new scripts
compiled in their intended environments; the 49-row experiment registry
parsed correctly; artifact checks confirmed 49 groups, 147 unique views, 85
per-GT rows, 12 visual examples, 2 rescues and 26 losses.

## 2026-10-17 — Alignment-Aware Selective DINOv3 Multi-View Fusion

Completed a no-training A/B/C experiment on the existing ForestPersons
benchmark49 development set. The new non-trainable gate preserves reference
features, requires pointwise and local 3x3 DINOv3 cosine agreement plus valid
homography support, caps combined source weight at 0.50, and falls back exactly
to the reference where no source is reliable. No annotations or detections are
used by the gate.

Frozen A/B COCO dumps reproduced saved recall, FP/image, AP and AR@100 exactly
before selective inference. At fixed confidence 0.55:

| Metric | Single | Equal mean | Selective |
|---|---:|---:|---:|
| Recall | 0.8706 | 0.5882 | 0.8824 |
| R20 / R40 / R70 / R100 | .6667/.8000/.9091/.9444 | .6667/.6000/.5909/.5556 | .6667/.8000/.9091/.9722 |
| Precision / FP-image | .8605 / .2449 | .7042 / .4286 | .8721 / .2245 |
| AP / AP50 / AP75 | .4142/.8778/.3099 | .2291/.6124/.0628 | .4183/.8971/.2758 |
| AR@100 | .5094 | .3600 | .5235 |

Selective fusion restored all 26 single-view detections lost by equal averaging,
lost none, and added one visibility-100 GT through a borderline IoU shift
(0.489 to 0.506). It did not improve R20/R40/R70. Sequence-bootstrap recall
difference C-A was +0.0118 with 95% interval [0.0000, 0.0380]. AP75 decreased.
Mean active gate area was 77.5%, but 44/85 GT centers had zero source weight;
the only added GT also had zero center weight. This is not evidence that an
extra view revealed an occluded person.

Selective runtime was 1.384 s/group and peak MPS driver allocation 2.26 GiB,
versus 0.461 s/group single-view. Conclusion: selective fusion prevents damage
but provides no demonstrated meaningful multi-view gain on ForestPersons.
Stop further ForestPersons homography-gate tuning and prioritize geometry-aware
AOS data with camera pose/calibration and official rectification metadata.

Artifacts:

- `docs/reports/dinov3_alignment_aware_selective_multiview_benchmark49_2026-10-17.md`
- `scripts/evaluate_dinov3_selective_multiview_fusion.py`
- `scripts/summarize_dinov3_selective_multiview_fusion.py`
- `outputs/multiview/forestpersons_val_benchmark49_selective_seed20261017_rerun1/`

## 2026-10-17 — ForestPersons Train5000 Multi-View Training Audit

Audited the existing train5000 subset for learnable three-view fusion without
training or downloads. Group qualification used split-qualified filenames,
sequence/frame metadata and unchanged benchmark49 ORB/RANSAC checks. The 228
qualified groups were frozen and hashed before annotations were opened.

Results:

- 5,000 readable images from 258 sequence folders;
- 2,302 metadata-eligible references and 12,022 possible offset triplets;
- 228 registration-qualified references across 87 sequences;
- 145 frame-disjoint groups containing 208 reference people;
- visibility 20/40/70/100 = 49/47/55/57;
- 144/208 people have projected annotation IoU >=0.50 in at least one source,
  76/208 in both, and 64/208 in neither;
- no train5000 sequence overlap with benchmark49 or pilot8.

A frozen eligibility list was partitioned by sequence with post-selection
visibility stratification. The proposed pilot split has 112 training groups
from 67 sequences and 160 people (38/36/42/44), plus 33 validation groups from
20 sequences and 48 people (11/11/13/13). Sequence and split-qualified image
overlap are both zero. ForestPersons lacks site IDs and track identities, so
site independence and same-person correspondence cannot be confirmed.

Recommended pilot architecture: shared frozen DINOv3 ViT-S + frozen q/v LoRA,
fixed feature warping, a roughly 37k-parameter shared local source gate with a
0.50 total source-weight cap and exact invalid-source fallback, followed by the
frozen trained FCOS head. Cache the LoRA features, train only fusion for 10
epochs, and use the existing FCOS losses. This is feasible on the M4 16 GB but
is exploratory; benchmark49 is not an untouched final test. If the gate only
reproduces reference performance, stop ForestPersons fusion and move to AOS
geometry.

Artifacts:

- `docs/reports/forestpersons_train5000_multiview_training_feasibility_2026-10-17.md`
- `scripts/audit_forestpersons_train5000_multiview.py`
- `scripts/analyze_forestpersons_train5000_multiview_correspondence.py`
- `scripts/design_forestpersons_multiview_sequence_split.py`
- `outputs/multiview/forestpersons_train5000_multiview_audit_seed20261017/`

## 2026-10-17 — Learnable DINOv3 Multi-View Fusion Pilot

Completed the single approved trainable-fusion pilot using the frozen audited
split: 112 training groups from 67 sequences and 33 validation groups from 20
sequence-disjoint sequences. The shared local gate has 37,377 trainable
parameters; DINOv3, rank-4 q/v LoRA, and FCOS remained frozen. The MPS smoke
verified finite gradients, an optimizer update confined to the gate, unchanged
FCOS weights, and exact invalid-mask reference fallback.

Training used AdamW LR 3e-4 for 10 epochs (280 optimizer steps). Epoch 1 was
selected by the predeclared AR@100/recall/FP criterion. On the 33-group/48-GT
validation split at confidence 0.55:

- single: recall 0.8542, R20 0.8182, FP/image 0.1818, AP 0.6134, AR@100 0.7021;
- equal mean: recall 0.6667, R20 0.4545, FP/image 0.3636, AP 0.3246, AR@100 0.5125;
- fixed selective: recall 0.8125, R20 0.7273, FP/image 0.1818, AP 0.5918, AR@100 0.7000;
- learned selective: recall 0.8542, R20 0.8182, FP/image 0.1818, AP 0.6132, AR@100 0.7021.

Learned fusion rescued zero GT and lost zero versus single view. Mean source
weight was only 0.0168 at the selected epoch. It prevented feature corruption
but provided no detection gain. Gate training took 33.11 s; peak MPS driver
memory was 1.10 GiB during training and 2.26 GiB during feature extraction.
The validation sequences are held out from gate training but were previously
seen by the train5000 detector, so this is exploratory rather than independent.

Decision: stop ForestPersons multi-view fusion tuning. The next research phase
is official AOS geometry-aware multi-view work with calibration/pose metadata
and a genuinely scene-disjoint evaluation.

Artifacts:

- `docs/reports/dinov3_learnable_multiview_fusion_train112_2026-10-17.md`
- `scripts/train_dinov3_learnable_multiview_fusion.py`
- `scripts/summarize_dinov3_learnable_multiview_fusion.py`
- `outputs/multiview/forestpersons_learnable_fusion_train112_seed20261017/`

## 2026-10-18 — Official AOS Dataset and Geometry Audit

Completed a no-download audit of Zenodo records 4024677 and 6382373, the
search-and-rescue paper, and the official JKU-ICG/AOS `stable_release` branch.
ZIP inventories were inspected through central-directory byte ranges; only
small metadata/sample entries were read to verify schemas and dimensions.

Verified that `RAW_lowresRGB.zip` is 4.206 GiB and expands to 9.586 GiB. It
contains 18 flights, 6,271 rectified 512x512 RGB frames, corresponding thermal
variants, per-frame RGB/thermal pose JSON, COLMAP reconstructions, KML tracks,
camera calibration and rectification material. The RGB/thermal integral-image
supplement is 4.689 GiB and expands to 4.716 GiB; it contains 9,612 labeled RGB
and 9,612 corresponding thermal 512x512 integral products. These are augmented
products, not independent scenes.

The paper's original labels were 3D polygonal contours and were projected to
single views, but the public archives visibly expose derived 2D YOLO boxes.
Prepared single-view labels in `YOLO.zip` are for thermal frames; directly
labeled raw RGB frames and a standalone original 3D-contour file were not found
in the archive inventories. A fair paired RGB comparison therefore requires a
verification gate for virtual-camera reproduction and label projection.

Recommended Stage 1 download, pending approval: `RAW_lowresRGB.zip` plus
`Integral Images.zip`, exactly 8.895 GiB transferred, 14.302 GiB extracted and
23.197 GiB archive-plus-extracted peak; reserve at least 30 GiB. Current free
space was approximately 131 GiB. Defer the 7.409-GiB legacy `YOLO.zip` and the
44.388-GiB high-resolution RGB archive. The current JKU repository code is
non-commercial research/evaluation licensed, unlike the CC BY 4.0 Zenodo data;
company/supervisor confirmation is required before reusing repository code.

Artifact:

- `docs/reports/aos_dataset_geometry_multiview_audit_2026-10-18.md`

## 2026-10-19 — AOS Stage 1 Download and Geometry/Annotation Verification

Downloaded only the approved official archives: `RAW_lowresRGB.zip`
(4,516,263,778 bytes) and `Integral Images.zip` (5,034,548,706 bytes). Both
official MD5 hashes matched and both archives passed `unzip -tq`. Archives were
retained after extraction; high-resolution RGB and `YOLO.zip` were not
downloaded. Approximately 107 GiB remains free.

Verified 18 scenes with 6,271 rectified RGB, 6,271 original thermal and 6,271
rectified thermal frames. Every frame maps exactly to a released pose; all
RGB/thermal pose counts match, and the fixed rig transform is stable to
`5.995e-15`. Verified 9,612 RGB and 9,612 thermal integral products and their
labels. RGB/thermal label pairs are byte-identical, and every canonical scene
label count matches the paper (37 train, 5 validation, 57 test people).

For F0, a pose/registration-selected three-view example confirmed overlapping
visual coverage. Two source-to-reference ORB/RANSAC registrations had 11 and
53 inliers with median errors 0.83 and 1.41 px. These are background/canopy
alignment diagnostics only and do not establish person or ground-plane
correspondence.

The release contains no direct raw-RGB person labels, original 3D contours, or
the exact virtual-camera/DEM configuration needed to map raw/fused outputs to
official integral-image labels. Raw-RGB and geometry-fusion recall therefore
remain blocked; official RGB/thermal integral labels are valid. Next action:
request the original 3D contours and exact integral-generation geometry from
the AOS authors before running the paired DINOv3 A/B/C experiment.

Artifacts:

- `docs/reports/aos_stage1_download_geometry_annotation_verification_2026-10-19.md`
- `scripts/audit_aos_stage1.py`
- `data/aos/README.md`
- `outputs/aos/stage1_verification/`

## 2026-10-19 — AOS RGB Integral Zero-Shot DINOv3 Evaluation

Ran the frozen ForestPersons DINOv3 ViT-S/16 rank-4 q/v LoRA + stride-16
FCOS checkpoint on one predeclared canonical RGB integral product per official
AOS test scene (10 images, 57 repeated-scene-independent GT people). No train
or validation products, threshold tuning, downloads, or model changes were
used. Frozen settings were short side 672, confidence 0.55, prediction floor
0.001, and NMS IoU 0.50.

At IoU 0.50: 2 TP, 12 FP, 55 FN, recall 0.0351, precision 0.1429,
FP/image 1.20, AP50 0.0357, and AR@100 at IoU 0.50 0.1930. Standard COCO
AR@100 averaged over IoU 0.50:0.95 was 0.0474. At the separately labeled
IoU 0.25 context protocol: 7 TP, 7 FP, 50 FN, recall 0.1228. This is a
zero-shot RGB-integral cross-dataset result and is not directly comparable to
the paper's trained thermal-integral detector.

Artifacts:

- `scripts/evaluate_aos_rgb_integral_zero_shot.py`
- `outputs/aos/rgb_integral_zero_shot_canonical_test_20261019/`

## 2026-10-19 — AOS RGB Integral Zero-Shot Failure Diagnostic

Reproduced all 830 saved post-NMS predictions exactly and audited raw FCOS,
top-300, NMS, and fixed-confidence stages without tuning. Image/label pairing,
512-to-672 scaling, and preprocessing are correct. Of 57 GT, 31 have no raw
IoU>=0.50 candidate, 8 valid candidates are lost at top-300, 7 at NMS, 9 fall
below confidence 0.55, and 2 remain detected. The dominant issue is
cross-domain localization, compounded by ranking/NMS, not coordinate scaling.
Next: verify the exact AOS virtual-camera/DEM/3D-label mapping on train/valid
scenes before implementing geometry-aware DINOv3 feature fusion.

Artifacts:

- `docs/reports/aos_rgb_integral_zero_shot_failure_diagnostic_2026-10-19.md`
- `scripts/diagnose_aos_rgb_integral_zero_shot.py`
- `outputs/aos/rgb_integral_zero_shot_canonical_test_20261019/diagnostics/`

## 2026-10-19 — AOS F8 Geometry Feasibility

Verified on forest training scene F8 that released RGB poses reproduce COLMAP
world-to-camera transforms to within `3.14e-9`; rectified intrinsics are
`fx=fy=538.947368`, `cx=cy=256` at `512x512`. Three neighboring RGB views
registered reliably in image space (124/156 RANSAC inliers; 1.37/1.09 px
median errors). Exact official virtual-camera reprojection is blocked because
the downloaded release lacks the F8 DEM/OBJ, its COLMAP-frame transform, and
the exact virtual-camera/focal-surface configuration. No test data, detection,
training, or downloads were used.

Artifacts:

- `docs/reports/aos_geometry_feasibility_f8_2026-10-19.md`
- `scripts/validate_aos_f8_geometry.py`
- `outputs/aos/geometry_feasibility_f8_20261019/`

## 2026-10-19 — Repository Review Preparation

Added `docs/CODE_REVIEW_GUIDE.md` with the maintained DINOv3, LoRA, FCOS,
evaluation, and multi-view execution flow plus a five-day study plan. Added a
non-destructive cleanup candidate list; no artifacts were deleted or moved.
Updated README navigation and documented `PYTHONPATH="$PWD/src:$PWD"` for the
learnable fusion entry point. Focused multi-view tests passed (18/18).

## 2026-10-20 — DTSS Real-Image Pilot Label Audit

Stopped the requested ForestPersons person-present/person-absent DTSS pilot at
the predeclared label-validity gate. The COCO splits contain 13 zero-box train
images and 7 zero-box test images, but no exhaustive-annotation or confirmed
absence field, so these records were not treated as negatives. The official
paper documents a separate 193-image verified no-human directory,
`379_FPV_No_Person_summer_forest`, but it is not present under local `data/`.
No model code, training, inference, or checkpoints were changed.

Artifact:

- `docs/reports/dtss_real_image_pilot_feasibility_2026-10-20.md`

## 2026-10-20 — DTSS ForestPersons Detection Pilot

Implemented an isolated 35,713-parameter image patch embedding + one-block
Spiking Transformer + dense objectness/box head. A four-image overfit gate
passed (loss 1.2160 to 0.00834), and matched Fixed-T/learned-DTSS runs completed
on 32 train and 16 validation images. At confidence 0.05, Fixed T achieved
AP50 0.0004, recall 0.0625, R20 0, and 25.06 FP/image; learned DTSS achieved
AP50 0.0011, recall 0.1250, R20 0, and 25.63 FP/image. All learned timestep
parameters changed, but every mask remained at 4/4 active steps. These are
inconclusive feasibility results, not a detector comparison. Focused DTSS tests
passed (17/17). The first output without normalized annotation metadata is
preserved but explicitly invalid; `rerun1` is canonical.

Artifacts:

- `src/forestpersons/dtss_detector.py`
- `scripts/run_dtss_forestpersons_detection_pilot.py`
- `tests/test_dtss_detector.py`
- `docs/reports/dtss_forestpersons_detection_pilot_2026-10-20.md`
- `outputs/dtss_detection/train32_val16_seed20261020_rerun1/`

## 2026-10-20 — AOS Multimodal/Multi-View Feasibility

Audited F0, F8, and O6 using only downloaded official metadata and images.
Same-index RGB/thermal pose pairs have a constant relative transform to within
`6.0e-15`; one-slot shifted pairing does not. The released model co-locates the
cameras and specifies a constant 0.8896-degree rotation under common rectified
intrinsics, enabling a reproducible rotation homography while not proving exact
physical pixel registration. Six metadata-selected RGB overlaps passed the
existing ORB/RANSAC checks. Raw frames still lack person boxes; direct labels
are valid only for official RGB/thermal integral products. Recommended next is
a frozen-DINOv3 feature-correspondence pilot on canonical paired integral
images before implementing any fusion.

Artifacts:

- `scripts/audit_aos_multimodal_feasibility.py`
- `docs/reports/aos_multimodal_multiview_feasibility_2026-10-20.md`
- `outputs/aos/multimodal_feasibility_20261020_rerun1/`

## 2026-10-20 — AOS Frozen DINOv3 Correspondence Prepared

Implemented, but did not execute, a frozen DINOv3 RGB-thermal patch-feature
correspondence experiment for all 18 official canonical integral-image pairs.
The protocol measures same-location cosine, fixed 3x3 local correspondence,
global patch retrieval, person/background patch similarity, and a predeclared
wrong-scene control. It loads the existing ViT-S/16 LoRA checkpoint, freezes
the backbone, and does not use the FCOS head or implement fusion. Three focused
synthetic tests passed; syntax validation passed. No AOS inference or metrics
were produced in this task.

Artifacts:

- `src/forestpersons/aos_correspondence.py`
- `scripts/evaluate_aos_dinov3_rgb_thermal_correspondence.py`
- `tests/test_aos_correspondence.py`
- `docs/reports/aos_dinov3_rgb_thermal_correspondence_implementation_2026-10-20.md`

## 2026-10-20 — AOS RGB/Thermal Detection Comparison Prepared

Prepared, but did not execute, a matched frozen-feature comparison of RGB-only,
thermal-only, and fixed mean RGB/thermal fusion. Official TRAIN/VALID/TEST
scenes are disjoint and paired RGB/thermal labels are byte-identical. The
deterministic protocol selects 224/32/10 paired products with 1,184/160/57 box
instances. However, the augmented TRAIN and VALID products represent only 37
and 5 unique people, so this is a limited feasibility experiment. All arms use
the same frozen ForestPersons DINOv3 ViT-S/16 + LoRA encoder and identically
initialized fresh FCOS heads; only O6 validation selects operating thresholds,
and canonical TEST scenes remain untouched. Four focused tests passed. No
features were extracted and no training or evaluation was run.

Artifacts:

- `src/forestpersons/aos_multimodal_detection.py`
- `scripts/run_aos_dinov3_multimodal_detection.py`
- `tests/test_aos_multimodal_detection.py`
- `docs/reports/aos_dinov3_rgb_thermal_detection_preparation_2026-10-20.md`

## 2026-10-20 — AOS Thermal vs Mean-Fusion Result Diagnosis

Analyzed the completed canonical TEST run entirely from saved post-NMS
predictions. At the validation-locked 0.70 thresholds, thermal detected 26/57
people while mean fusion detected 5/57, with both producing two false
positives. Fusion lost 21 thermal true positives and gained none. Eighteen lost
GTs still had a valid fusion candidate at IoU >= 0.50 whose score was below
0.70; two had best saved IoU between 0.25 and 0.50, and one had no useful
nearby saved candidate. Across thermal-detected GTs, fusion reduced the best
candidate score by 0.282 on average. All ten scene overlays and per-GT
localization diagnostics were saved. Because the dumps contain only post-NMS
top-100 candidates above 0.001, pre-NMS/top-k/NMS attribution is unavailable.

Artifacts:

- `scripts/analyze_aos_multimodal_detection_results.py`
- `docs/reports/aos_thermal_vs_mean_fusion_analysis_2026-10-20.md`
- `outputs/aos/dinov3_rgb_thermal_detection_20261020/analysis_thermal_vs_mean_fusion_20261020/`

## 2026-10-20 — AOS Thermal-Weighted Fusion Prepared

Extended the established cached-feature workflow with fixed thermal-weighted
fusion at alpha 0.75 and 0.90 while preserving all original arms and defaults.
The controlled run retrains fresh, identically initialized FCOS heads for
thermal-only and both weighted arms. O6 VALID exclusively selects each
threshold and the preferred arm before canonical TEST evaluation; evaluation
can require this persisted selection record. Four focused tests passed and the
CLI compiled. Per instruction, no training or evaluation was executed.

Artifacts:

- `src/forestpersons/aos_multimodal_detection.py`
- `scripts/run_aos_dinov3_multimodal_detection.py`
- `tests/test_aos_multimodal_detection.py`
- `docs/reports/aos_thermal_weighted_fusion_experiment_preparation_2026-10-20.md`

## 2026-10-20 — DTSS train5000/val600 Full-Run Infrastructure Prepared

Prepared a resumable 20-epoch MPS runner for matched Fixed-T=4 and learned-DTSS
arms without changing the detector architecture or pilot. It saves atomic
per-epoch model/optimizer checkpoints and evaluates AP50, AR@100, recall,
R20, precision, FP/image, and active timesteps on val600 after completion.
Image-ID-based loading verified 5,000/6,415 train images/annotations and
600/749 evaluable validation images/annotations, excluding one `unknown_0`
annotation; split-qualified overlap is zero. Sixteen focused tests passed. No
training was run and no full-run output directory was created.

Artifacts:

- `scripts/train_dtss_forestpersons_full.py`
- `tests/test_dtss_detector.py`
- `docs/reports/dtss_train5000_val600_training_readiness_2026-10-20.md`

## 2026-10-21 — Frozen DINOv3 MVR + FCOS Prepared

Implemented, but did not run, a matched train5000/val600 comparison between
the canonical frozen final-layer DINOv3 FCOS head and a 398,976-parameter
multi-layer/multi-resolution residual readout feeding the same FCOS head. The
MVR arm uses frozen blocks 3/6/9 plus the existing block-12 cache, with native,
2x, and half-resolution context. Its zero-initialized residual exactly matches
the baseline feature at initialization; both FCOS heads share an identical
fresh state and established 20-epoch settings. New extraction, resumable
training, and paired evaluation CLIs were added. Six focused tests passed. No
features were extracted and no training/evaluation outputs were created.

Artifacts:

- `src/forestpersons/dinov3_detector.py`
- `scripts/cache_dinov3_mvr_features.py`
- `scripts/train_dinov3_mvr_fcos.py`
- `scripts/evaluate_dinov3_mvr_fcos.py`
- `tests/test_dinov3_mvr.py`
- `docs/reports/dinov3_frozen_mvr_fcos_training_preparation_2026-10-21.md`

## 2026-10-22 — Frozen DINOv3 MVR Completed and ANN2SNN Pilot Prepared

The matched train5000/val600 experiment completed. Baseline FCOS achieved AP
0.3267, AP50 0.7915, AP75 0.2022, and AR@100 0.4550. MVR + FCOS improved these
to AP 0.3623, AP50 0.8162, AP75 0.2675, and AR@100 0.4900. At the exploratory
0.15 FP/image budget, baseline recall/R20 were 0.6916/0.4425 and MVR recall/R20
were 0.7210/0.4885.

Prepared a bounded hybrid ANN2SNN inference pilot without running detector
evaluation. Frozen signed DINOv3 features remain analog; deterministic
train-feature calibration converts only the six MVR ReLU sites to soft-reset IF
rate codes at T=4/8/16. Residual normalization, the GroupNorm/GELU FCOS tower,
classification/centerness, box regression, decoding, and NMS remain analog.
Spike activity is reported only as an activity proxy, never as measured energy.
Seventeen focused MVR, ANN2SNN, and evaluation tests passed, including MPS.

Artifacts:

- `src/forestpersons/dinov3_ann2snn.py`
- `scripts/evaluate_dinov3_mvr_ann2snn.py`
- `tests/test_dinov3_ann2snn.py`
- `docs/reports/dinov3_mvr_ann2snn_pilot_preparation_2026-10-22.md`
- `outputs/dinov3_mvr/evaluation_val600_threshold_sweep_20261022`

## 2026-10-22 — MVR ANN2SNN Pilot Sampling Audit

Audited the completed 100-image hybrid ANN2SNN pilot. Its manifest-prefix
sampling selected only the first visibility-20 block: 100 images, 116 R20 GT,
and no R40/R70/R100 GT. This explains why overall recall equaled R20 for each
mode. The modes did not have identical recall (ANN 0.6034, T4 0.4569, T8
0.5862, T16 0.6207). The ANN pilot's 9,213 saved detections exactly matched the
same image IDs in the prior val600 MVR prediction artifact. Its AP50 0.6151
versus full-val600 0.8162 is therefore a hard-stratum sampling effect, not an
inference or metric error.

Added opt-in deterministic visibility-stratified evaluation, full R20/R40/R70/
R100 reporting, and exact reuse of the original train-only calibration
thresholds. Original outputs and default prefix behavior remain unchanged.

Report: `docs/reports/dinov3_mvr_ann2snn_pilot_audit_2026-10-22.md`.

## 2026-10-22 — MVR Hybrid ANN2SNN val600 Completed

Evaluated the unchanged trained MVR + FCOS checkpoint on all 600 val600 images
using the original train-only calibration and fixed confidence 0.63. ANN AP50/
recall/R20 were 0.8162/0.8077/0.5862. Hybrid SNN results were: T4
0.7622/0.7156/0.4425, T8 0.8002/0.7891/0.5632, and T16
0.8083/0.7997/0.5862. T16 retained 99.0% of ANN AP50 and matched ANN R20, but
was 3.77x slower in the dense MPS simulation. Activity ratios ranged from
0.0246 to 0.0356 and are not hardware energy measurements.

The GitHub-safe summary is
`docs/reports/dinov3_mvr_ann2snn_val600_2026-10-22.md`; raw checkpoints,
predictions, caches, and full output directories remain local and excluded from
version control.

## 2026-10-22 — MVR Native Readout Ablation Implemented

Added an optional `mvr_native` arm between the unchanged final-layer baseline
and full native/fine/coarse MVR. The 197,248-parameter native readout projects
blocks 3/6/9/12 at the original patch grid and adds a zero-initialized residual
to block 12; together with FCOS it has 721,894 trainable parameters. Existing
baseline and full-MVR checkpoint state dictionaries strict-load unchanged.

Focused MVR/ANN2SNN tests passed 14/14 and the maintained suite passed 190 with
3 skipped. A two-image real-cache MPS forward/backward smoke produced finite
losses 1.25248 and 1.04025, nonzero projection gradients after the first update,
0.927 s runtime, and 1.043 GiB observed driver memory. No checkpoint or
training output was written and full training was not started.

The trainer and evaluator now support the three-arm comparison while preserving
legacy two-arm behavior. Exact non-overwriting commands are documented in
`docs/reports/dinov3_mvr_native_ablation_implementation_2026-10-22.md`.

## 2026-10-23 — MVR Native Readout Ablation Completed

Completed the matched 20-epoch `mvr_native` train5000 run and three-arm val600
evaluation. Native multi-layer readout achieved AP 0.3303, AP50 0.7926, AP75
0.2082, and AR@100 0.4615, compared with baseline 0.3267/0.7915/0.2022/0.4550
and full MVR 0.3623/0.8162/0.2675/0.4900. At the exploratory 0.425 FP/image
budget, recall/R20 were baseline 0.8198/0.6207, native 0.8318/0.6322, and full
MVR 0.8278/0.6092.

Native training loss fell from 0.85827 to 0.61632 over 20 epochs in 7,511.16 s
(2 h 05 m 11 s). The ablation shows that native-grid intermediate layers yield
only small AP/AR gains; most full-MVR improvement, especially AP75, comes from
the added fine/coarse multi-resolution context. Operating points are
exploratory because thresholds were selected on val600.

Artifacts and interpretation are recorded in
`docs/reports/dinov3_mvr_native_ablation_results_2026-10-23.md`.

Repeated the exact three-arm cached-feature evaluation into the separate
`evaluation_val600_readout_ablation_seed20261012_rerun1` directory. All three
prediction JSON files, the confidence sweep, and matched-budget CSV matched the
original byte-for-byte; metric values were exact after excluding runtime-only
fields. No model was retrained and no prior output was overwritten.

## 2026-10-23 — Full MVR Label-Efficiency Study Prepared

Prepared, but did not train, a controlled Full MVR label-efficiency study using
exact nested train100/train400/train1000/train5000 prefixes. The historical
train400 was not nested (397/400 overlap with train5000), and naive train5000
prefixes were visibility-biased, so a deterministic visibility-interleaved,
sequence-folder-round-robin order was created from exactly the existing
train5000 pool.

The subsets contain 141/519/1300/6415 annotations and 89/222/258/258 sequence
folders. Each has exactly 25/100/250/1250 target images per visibility. Train
and val have zero filename and zero sequence-folder overlap; 45 split-local
numeric image IDs overlap and are deliberately not used for identity. All
5,000 images and both 5,000-item feature caches were verified present.

The trainer now supports an explicit expected sample count and records strict
manifest, sample-count, FCOS-initialization, and full-model-initialization
metadata for resume/evaluation checks. A thin evaluator reuses the canonical
val600 metric implementation. Focused tests and exact non-overwriting commands
are documented in
`docs/reports/dinov3_mvr_label_efficiency_readiness_2026-10-23.md`.

## 2026-10-23 — Full MVR Label-Efficiency Study Completed

Completed matched 20-epoch Full MVR training on exact nested train100/400/1000/
train5000 subsets and evaluated all four checkpoints on val600. All runs used
the same FCOS and complete-model initialization hashes and had finite 20-epoch
histories.

AP increased monotonically from 0.0844 to 0.1735 to 0.2673 to 0.3353; AP75 from
0.0151 to 0.0547 to 0.1290 to 0.2169; and AR@100 from 0.2430 to 0.3387 to
0.4099 to 0.4780. At the exploratory 0.425 FP/image budget, recall increased
from 0.3231 to 0.5607 to 0.7250 to 0.8224, while R20 increased from 0.1264 to
0.2644 to 0.4713 to 0.5977.

Training runtimes were 165/612/1665/8119 seconds with approximately 1.10 GiB
peak MPS driver memory. The trend supports additional seeds, but the experiment
uses one seed, fixed epochs imply different optimizer-step counts, val600 is a
reused pilot subset, and train1000 already spans all 258 training sequences.
The matched train5000 AP is also 0.0270 below the earlier differently ordered
Full MVR run, demonstrating order sensitivity.

Full results and limitations are documented in
`docs/reports/dinov3_mvr_label_efficiency_results_2026-10-23.md`.

## 2026-10-23 — Frozen versus LoRA Full MVR Experiment Prepared

Prepared the matched online Experiment 3 infrastructure without starting
training. Both arms strict-load the same strongest completed Full MVR + FCOS
checkpoint (AP 0.3623) and use the same train5000 manifest, two epochs, image
order, losses, accumulation, optimizer-step budget, and val600 protocol. The
control trains only the 923,622-parameter MVR/FCOS head with DINOv3 frozen; the
experimental arm additionally trains 73,728 fresh rank-4 q/v LoRA parameters.

Added differentiable online extraction for blocks 3/6/9/12, an overwrite-safe
resumable trainer, and an evaluator that rejects mismatched initialization,
manifest, settings, optimizer steps, or per-epoch image order. Focused LoRA/MVR
and label-efficiency tests passed 25/25, both CLI entry points import successfully, and the actual
initial checkpoint strict-loaded. No training, inference, cache generation,
commit, or push occurred. Commands, estimates, and limitations are in
`docs/reports/dinov3_mvr_frozen_vs_lora_readiness_2026-10-23.md`.

## 2026-10-23 — Frozen versus LoRA Full MVR MPS Smoke Passed

Added a non-resumable `--smoke-optimizer-steps` mode that preserves the full
Experiment 3 configuration but writes only `smoke.pt` and
`smoke_summary.json`; it never writes `latest.pt` or `final.pt`. Two matched
MPS runs processed 8 real train5000 images and performed 2 optimizer steps.

Frozen and fresh-LoRA losses were finite at 0.632151 and 0.635843. Runtime was
6.51/11.25 seconds and observed peak MPS driver memory was 1.69/5.56 GiB. No
gradient clipping occurred. All 32 MVR/FCOS tensors received nonzero gradients
and changed in both arms. In the LoRA arm, 24/48 adapter gradients were nonzero
at step 1 as expected from zero-initialized B matrices; all 48 were nonzero at
step 2, and all 24 B matrices changed. Sample-order hashes matched exactly.

Focused tests passed 26/26. Full training was not started. Commands and artifacts
are documented in
`docs/reports/dinov3_mvr_frozen_vs_lora_mps_smoke_2026-10-23.md`.

## 2026-10-24 — Comprehensive Evidence Audit and Experiment 3 Results

Completed a documentation-only audit of the repository registry, reports, and
saved metrics under `$LOGXY_ARTIFACT_ROOT`. No training, inference,
downloads, checkpoint changes, or Git operations were performed.

The matched Frozen-versus-LoRA Full-MVR continuation has now completed and the
registry was corrected from `prepared` to `completed`. Both arms continued for
two epochs from the same trained Full-MVR checkpoint on the same ordered
train5000 stream. Frozen AP/AP50/AP75/AR@100 were
0.3533/0.8082/0.2362/0.4846; LoRA achieved
0.3652/0.8144/0.2722/0.4873. At the exploratory 0.425 FP/image budget, frozen
recall/R20 were 0.8505/0.6264 at 0.4167 FP/image, while LoRA reached
0.8531/0.6552 at 0.4000 FP/image. Frozen and LoRA training took 4,916 and
13,232 seconds with 2.33 and 11.89 GiB observed peak MPS driver memory.

Added missing registry records for completed DTSS train5000 work and AOS
integral-image/RGB-thermal studies. The audit also records the corrected
image-ID-safe train5000 annotation count of 6,415 versus a historical
YOLO-conversion artifact count of 6,431. The difference is preserved as a
reproducibility limitation.

The complete supervisor-facing synthesis, protocol boundaries, evidence paths,
and prioritized next experiments are in
`docs/reports/logxy_comprehensive_experiment_comparison_2026-10-24.md`.

## 2026-10-24 — GitHub Release Preparation

Prepared the code/documentation tree for manual GitHub review without staging,
committing, pushing, moving, or deleting files. Expanded `.gitignore` coverage
for datasets, generated outputs, model/cache formats, archives, credentials,
experiment trackers, profiler traces, and machine-local files while retaining
lightweight subset-manifest exceptions. Expanded `README.md` with the actual
research pipeline, datasets, verified results, environment, artifact-root
convention, reproducible MVR commands, evaluation protocol, and limitations.

Sanitized personal absolute paths in public project documentation by using
`$LOGXY_REPO` and `$LOGXY_ARTIFACT_ROOT`; intentional Colab
`/content/drive/...` paths remain. Updated
`docs/GITHUB_RELEASE_CHECKLIST.md` with publication blockers and exact manual
review commands. No training, inference, full test run, or artifact mutation
was performed.

