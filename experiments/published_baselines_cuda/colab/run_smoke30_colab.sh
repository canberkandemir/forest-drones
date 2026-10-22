#!/usr/bin/env bash
set -euo pipefail

LOGXY_ROOT="${LOGXY_ROOT:-/content/drive/MyDrive/LogXY}"
MMDET_ROOT="${MMDET_ROOT:-/content/third_party/mmdetection}"
ENV_NAME="${LOGXY_MAMBA_ENV:-logxy-mmdet2}"
RESULT_DIR="${LOGXY_ROOT}/experiments/published_baselines_cuda/results/colab_smoke30"
CHECKPOINT_DIR="${LOGXY_ROOT}/experiments/published_baselines_cuda/checkpoints"
export PATH="${HOME}/.local/bin:${PATH}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
mkdir -p "${RESULT_DIR}"

run_in_env() {
  micromamba run -n "${ENV_NAME}" "$@"
}

cd "${LOGXY_ROOT}"
export PYTHONPATH="${LOGXY_ROOT}/src:${LOGXY_ROOT}"

run_in_env python scripts/verify_mmdet_cuda_env.py | tee "${RESULT_DIR}/environment_verification.json"

run_in_env python scripts/mmdet_infer_forestpersons.py \
  --config "${MMDET_ROOT}/configs/detr/detr_r50_8x2_150e_coco.py" \
  --checkpoint "${CHECKPOINT_DIR}/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth" \
  --annotation outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json \
  --image-root data/images \
  --output "${RESULT_DIR}/detr_r50_coco_val600_smoke30.json" \
  --device cuda:0 \
  --max-images 30 \
  --score-thr 0.001 \
  --source-class-index 0

run_in_env python scripts/mmdet_infer_forestpersons.py \
  --config "${MMDET_ROOT}/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py" \
  --checkpoint "${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth" \
  --annotation outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json \
  --image-root data/images \
  --output "${RESULT_DIR}/deformable_faster_rcnn_r50_dcn_coco_val600_smoke30.json" \
  --device cuda:0 \
  --max-images 30 \
  --score-thr 0.001 \
  --source-class-index 0

run_in_env python scripts/evaluate_external_coco_results.py \
  --root data \
  --split val \
  --manifest outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json \
  --predictions "${RESULT_DIR}/detr_r50_coco_val600_smoke30.json" \
  --output-dir "${RESULT_DIR}/detr_r50_coco_val600_smoke30_eval"

run_in_env python scripts/evaluate_external_coco_results.py \
  --root data \
  --split val \
  --manifest outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json \
  --predictions "${RESULT_DIR}/deformable_faster_rcnn_r50_dcn_coco_val600_smoke30.json" \
  --output-dir "${RESULT_DIR}/deformable_faster_rcnn_r50_dcn_coco_val600_smoke30_eval"

echo "Saved smoke-test predictions and evaluations under ${RESULT_DIR}"
