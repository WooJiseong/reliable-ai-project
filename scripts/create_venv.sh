#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"

case "${BACKEND}" in
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  cu121)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu128)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  *)
    echo "Usage: bash scripts/create_venv.sh [cpu|cu121|cu128]" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchaudio==${TORCH_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements-common.txt
if [ -d "CompareNoiseWaveMel" ]; then
  COMPARE_NOISE_WAVE_MEL_DIR="CompareNoiseWaveMel"
else
  COMPARE_NOISE_WAVE_MEL_DIR="reliable-ai-project/CompareNoiseWaveMel"
fi
python -m pip install -e "${COMPARE_NOISE_WAVE_MEL_DIR}"

python - <<'PY'
import datasets
import soundfile
import torch
import torchaudio
import transformers
from noise_robust_asr.models.conformer_ctc import build_conformer_ctc

print("environment-ok")
print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
