# Shared Virtual Environment

Use one workspace-level virtual environment for both repositories:

- `ReliableAI_team_project`
- `CompareNoiseWaveMel`

The main version constraint is that `torch` and `torchaudio` must use the same
version because TorchAudio wheels are built against specific Torch versions.
This workspace pins both to `2.7.1`.

## Create The Environment

CPU-only:

```bash
bash scripts/create_venv.sh cpu
```

CUDA 12.1:

```bash
bash scripts/create_venv.sh cu121
```

CUDA 12.8:

```bash
bash scripts/create_venv.sh cu128
```

The script creates `.venv`, installs the matched Torch stack first, installs the
shared Hugging Face/audio dependencies, installs `CompareNoiseWaveMel` in
editable mode, and runs a small import check.

## Activate Later

```bash
source .venv/bin/activate
```

## Manual Install

If you do not use the helper script, install Torch first with the wheel index
that matches your machine, then install the shared requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-common.txt
python -m pip install -e CompareNoiseWaveMel
```

## Smoke Test

```bash
python - <<'PY'
import datasets
import torch
import torchaudio
import transformers
from noise_robust_asr.models.conformer_ctc import build_conformer_ctc

print(torch.__version__)
print(torchaudio.__version__)
print(transformers.__version__)
print(datasets.__version__)
print(torch.cuda.is_available())
PY
```
