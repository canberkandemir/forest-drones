#!/usr/bin/env bash
set -euo pipefail

LOGXY_ROOT="${LOGXY_ROOT:-/content/drive/MyDrive/LogXY}"
CHECKPOINT_DIR="${LOGXY_ROOT}/experiments/published_baselines_cuda/checkpoints"
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "${CHECKPOINT_DIR}"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [ -s "${output}" ]; then
    echo "Checkpoint exists: ${output}"
  else
    echo "Downloading ${url}"
    wget -O "${output}.tmp" "${url}"
    mv "${output}.tmp" "${output}"
  fi
}

download_if_missing \
  "https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth" \
  "${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth"

download_if_missing \
  "https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth" \
  "${CHECKPOINT_DIR}/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"

ls -lh "${CHECKPOINT_DIR}"
