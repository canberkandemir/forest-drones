#!/usr/bin/env bash
set -euo pipefail

# Prepare a reproducible MMDetection 2.x CUDA environment on Google Colab.
#
# This script intentionally creates an isolated micromamba environment instead
# of modifying Colab's default Python runtime. That keeps the workflow robust
# when Colab's system Python/PyTorch versions change.

ENV_NAME="${LOGXY_MAMBA_ENV:-logxy-mmdet2}"
PYTHON_VERSION="${LOGXY_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${LOGXY_TORCH_VERSION:-1.13.1}"
TORCHVISION_VERSION="${LOGXY_TORCHVISION_VERSION:-0.14.1}"
CUDA_VERSION="${LOGXY_CUDA_VERSION:-11.7}"
MMCV_VERSION="${LOGXY_MMCV_VERSION:-1.7.1}"
MMDET_VERSION="${LOGXY_MMDET_VERSION:-v2.28.2}"
MKL_CONSTRAINT="${LOGXY_MKL_CONSTRAINT:-mkl<2024.1}"
INTEL_OPENMP_CONSTRAINT="${LOGXY_INTEL_OPENMP_CONSTRAINT:-intel-openmp<2024.1}"

export PATH="${HOME}/.local/bin:${PATH}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
export MAMBA_NO_BANNER=1

find_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
    return 0
  fi
  for candidate in \
    "${HOME}/.local/bin/micromamba" \
    "${HOME}/bin/micromamba" \
    "/usr/local/bin/micromamba"; do
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. In Colab, enable Runtime > Change runtime type > GPU." >&2
  exit 1
fi

nvidia-smi

if [ ! -d /content/drive/MyDrive ]; then
  echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "Run this in a notebook first: from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 1
fi

if ! MICROMAMBA_BIN="$(find_micromamba)"; then
  echo "Installing micromamba..."
  "${SHELL:-/bin/bash}" -c "$(curl -L micro.mamba.pm/install.sh)" < /dev/null
  hash -r
  MICROMAMBA_BIN="$(find_micromamba)"
fi

echo "Using micromamba: ${MICROMAMBA_BIN}"
echo "Using MAMBA_ROOT_PREFIX: ${MAMBA_ROOT_PREFIX}"
eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"

if ! "${MICROMAMBA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating ${ENV_NAME}..."
  "${MICROMAMBA_BIN}" create -y -n "${ENV_NAME}" \
    -c pytorch -c nvidia -c conda-forge \
    "python=${PYTHON_VERSION}" \
    "pytorch=${TORCH_VERSION}" \
    "torchvision=${TORCHVISION_VERSION}" \
    "pytorch-cuda=${CUDA_VERSION}" \
    "${MKL_CONSTRAINT}" \
    "${INTEL_OPENMP_CONSTRAINT}" \
    pip
else
  echo "Environment ${ENV_NAME} already exists; reusing it."
fi

# PyTorch 1.13.x can fail to import with newer MKL/ITT runtimes:
#   undefined symbol: iJIT_NotifyEvent
# Keep this repair step outside environment creation so a partial failed Colab
# setup can be safely rerun.
"${MICROMAMBA_BIN}" install -y -n "${ENV_NAME}" \
  -c pytorch -c nvidia -c conda-forge \
  "${MKL_CONSTRAINT}" \
  "${INTEL_OPENMP_CONSTRAINT}"

"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python --version
"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())
PY
"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install -U pip
"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  "openmim" \
  "pycocotools" \
  "opencv-python-headless" \
  "tqdm" \
  "numpy<2"

echo "Installing mmcv-full ${MMCV_VERSION} via OpenMIM..."
"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" mim install "mmcv-full==${MMCV_VERSION}"

mkdir -p /content/third_party
if [ ! -d /content/third_party/mmdetection/.git ]; then
  git clone --branch "${MMDET_VERSION}" --depth 1 \
    https://github.com/open-mmlab/mmdetection.git \
    /content/third_party/mmdetection
else
  echo "MMDetection checkout already exists; refreshing requested tag."
  git -C /content/third_party/mmdetection fetch --depth 1 origin "refs/tags/${MMDET_VERSION}:refs/tags/${MMDET_VERSION}" || true
  git -C /content/third_party/mmdetection checkout "${MMDET_VERSION}"
fi

"${MICROMAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install -v -e /content/third_party/mmdetection

cat <<EOF

MMDetection Colab environment prepared.

Use commands like:
  ${MICROMAMBA_BIN} run -n ${ENV_NAME} python scripts/verify_mmdet_cuda_env.py

Recommended variables:
  export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
  export MMDET_ROOT=/content/third_party/mmdetection
  export LOGXY_ROOT=/content/drive/MyDrive/LogXY

EOF
