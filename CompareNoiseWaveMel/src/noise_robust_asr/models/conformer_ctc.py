from torch import nn
from typing import Optional, Tuple

import torch


MEL_FRONTEND_STATE_PREFIX = "frontend.mel."


def conformer_ctc_checkpoint_state(model: nn.Module) -> dict:
    state = model.state_dict()
    return {key: value for key, value in state.items() if not key.startswith(MEL_FRONTEND_STATE_PREFIX)}


def load_conformer_ctc_checkpoint_state(model: nn.Module, state: dict):
    state = {key: value for key, value in state.items() if not key.startswith(MEL_FRONTEND_STATE_PREFIX)}
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith(MEL_FRONTEND_STATE_PREFIX)]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Error(s) in loading ConformerCTC state_dict: "
            f"missing_keys={missing}, unexpected_keys={incompatible.unexpected_keys}"
        )
    return incompatible


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel = None

    def _mel_device(self) -> Optional[torch.device]:
        if self.mel is None:
            return None
        tensor = next(self.mel.buffers(), None)
        if tensor is None:
            tensor = next(self.mel.parameters(), None)
        return None if tensor is None else tensor.device

    def _build_mel(self, device: torch.device):
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            normalized=False,
        ).to(device)

    def forward(self, waveform: torch.Tensor, waveform_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mel is None or self._mel_device() != waveform.device:
            self._build_mel(waveform.device)

        features = self.mel(waveform).transpose(1, 2)
        features = torch.log(features.clamp_min(1e-5))
        features = (features - features.mean(dim=(1, 2), keepdim=True)) / features.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        feature_length = torch.div(waveform_length, self.hop_length, rounding_mode="floor") + 1
        feature_length = feature_length.clamp(max=features.size(1))
        return features, feature_length


class ConformerCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ff_dim: int = 1024,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from torchaudio.models import Conformer
        except ImportError as exc:
            raise ImportError("torchaudio is required for ConformerCTC") from exc

        self.frontend = LogMelFrontend(n_mels=input_dim)
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.encoder = Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ff_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)

    def forward(self, waveform: torch.Tensor, waveform_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, feature_length = self.frontend(waveform, waveform_length)
        x = self.input_proj(features)
        encoded, encoded_length = self.encoder(x, feature_length)
        logits = self.ctc_head(encoded)
        return logits, encoded_length


def build_conformer_ctc(vocab_size: int, size: str = "small") -> ConformerCTC:
    if size == "tiny":
        return ConformerCTC(vocab_size=vocab_size, encoder_dim=144, num_layers=4, num_heads=4, ff_dim=576)
    if size == "small":
        return ConformerCTC(vocab_size=vocab_size, encoder_dim=256, num_layers=12, num_heads=4, ff_dim=1024)
    if size == "medium":
        return ConformerCTC(vocab_size=vocab_size, encoder_dim=384, num_layers=16, num_heads=6, ff_dim=1536)
    raise ValueError(f"Unknown Conformer size: {size}")
