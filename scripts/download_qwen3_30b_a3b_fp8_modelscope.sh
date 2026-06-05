#!/usr/bin/env bash
set -euo pipefail

# export http_proxy="${http_proxy:-http://127.0.0.1:18080}"
# export https_proxy="${https_proxy:-http://127.0.0.1:18080}"
# export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18080}"
# export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18080}"
# unset ALL_PROXY all_proxy

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
LOCAL_DIR="${LOCAL_DIR:-models/Qwen3-4B-modelscope}"
VENV_DIR="${VENV_DIR:-.venv-modelscope-download}"

if command -v modelscope >/dev/null 2>&1; then
  MODELSCOPE_BIN="$(command -v modelscope)"
else
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python -m venv --clear "${VENV_DIR}"
  fi
  "${VENV_DIR}/bin/python" -m pip install -U pip modelscope
  MODELSCOPE_BIN="${VENV_DIR}/bin/modelscope"
fi

mkdir -p "${LOCAL_DIR}"

"${MODELSCOPE_BIN}" download \
  --model "${MODEL_ID}" \
  --local_dir "${LOCAL_DIR}"
