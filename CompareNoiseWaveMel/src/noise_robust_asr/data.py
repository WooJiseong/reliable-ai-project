import json
from pathlib import Path
from typing import List, Union

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from noise_robust_asr.text import CharTokenizer


TARGET_SAMPLE_RATE = 16_000


def read_manifest(path: Union[str, Path]) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"audio", "text", "language"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(item)
    return rows


class SpeechManifestDataset(Dataset):
    def __init__(self, manifest_path: Union[str, Path], tokenizer: CharTokenizer):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.project_root = Path(__file__).resolve().parents[2]
        self.items = read_manifest(self.manifest_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_audio_path(self, audio_path: str) -> Path:
        path = Path(audio_path)
        if not path.is_absolute():
            return self.manifest_dir / path
        if path.exists():
            return path

        marker = "reliable-ai-project/CompareNoiseWaveMel/"
        path_text = path.as_posix()
        if marker in path_text:
            return self.project_root / path_text.split(marker, 1)[1]
        return path

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        audio_path = self._resolve_audio_path(item["audio"])
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
        waveform = waveform.clamp(-1.0, 1.0)
        tokens = self.tokenizer.encode(item["text"])
        return {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "tokens": tokens,
            "token_length": torch.tensor(tokens.numel(), dtype=torch.long),
            "text": item["text"],
            "language": item["language"],
            "audio": str(audio_path),
        }


def collate_batch(batch: List[dict]) -> dict:
    waveforms = pad_sequence([item["waveform"] for item in batch], batch_first=True)
    tokens = torch.cat([item["tokens"] for item in batch])
    return {
        "waveform": waveforms,
        "waveform_length": torch.stack([item["waveform_length"] for item in batch]),
        "tokens": tokens,
        "token_length": torch.stack([item["token_length"] for item in batch]),
        "text": [item["text"] for item in batch],
        "language": [item["language"] for item in batch],
        "audio": [item["audio"] for item in batch],
    }
