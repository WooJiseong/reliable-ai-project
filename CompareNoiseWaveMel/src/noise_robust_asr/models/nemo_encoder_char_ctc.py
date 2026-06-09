import inspect
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        residual = encoded
        encoded = self.norm(encoded)
        encoded = F.gelu(self.down(encoded))
        encoded = self.dropout(encoded)
        encoded = self.up(encoded)
        return residual + encoded


class NemoEncoderCharCTC(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        vocab_size: int,
        freeze_encoder: bool = False,
        adapter_languages=None,
        adapter_dim: int = 0,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NeMo is required for NemoEncoderCharCTC. "
                "Install the ASR extra with: python -m pip install 'nemo_toolkit[asr]'"
            ) from exc

        self.pretrained_model = pretrained_model
        self.encoder_model = ASRModel.from_pretrained(model_name=pretrained_model)
        self._enable_input_gradients()

        encoder_dim = getattr(getattr(self.encoder_model, "encoder", None), "_feat_out", None)
        if encoder_dim is None:
            raise RuntimeError(f"Could not infer encoder output dimension from {pretrained_model}.")

        adapter_languages = adapter_languages or []
        self.adapter_languages = list(adapter_languages)
        self.adapters = nn.ModuleDict()
        if adapter_dim > 0:
            self.adapters = nn.ModuleDict(
                {
                    language: BottleneckAdapter(
                        hidden_size=int(encoder_dim),
                        bottleneck_size=adapter_dim,
                        dropout=adapter_dropout,
                    )
                    for language in self.adapter_languages
                }
            )

        self.ctc_head = nn.Linear(int(encoder_dim), vocab_size)
        if freeze_encoder:
            for parameter in self.encoder_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor, waveform_length: torch.Tensor, language=None):
        encoded, encoded_lengths = self.encoder_model(input_signal=waveform, input_signal_length=waveform_length)
        encoded = encoded.transpose(1, 2)
        encoded = self._apply_adapters(encoded, language)
        logits = self.ctc_head(encoded)
        return logits, encoded_lengths

    def log_probs(self, waveform: torch.Tensor, waveform_length: torch.Tensor, language=None):
        logits, encoded_lengths = self(waveform, waveform_length, language=language)
        return F.log_softmax(logits, dim=-1), encoded_lengths

    def _apply_adapters(self, encoded: torch.Tensor, language):
        if not self.adapters:
            return encoded

        if language is None:
            raise ValueError("language is required when language adapters are enabled.")

        if isinstance(language, str):
            if language not in self.adapters:
                raise ValueError(f"No adapter configured for language {language!r}.")
            return self.adapters[language](encoded)

        adapted = []
        for item, item_language in zip(encoded, language):
            if item_language in self.adapters:
                adapted.append(self.adapters[item_language](item.unsqueeze(0)).squeeze(0))
            else:
                adapted.append(item)
        return torch.stack(adapted, dim=0)

    def _enable_input_gradients(self) -> None:
        preprocessor = getattr(self.encoder_model, "preprocessor", None)
        if preprocessor is None:
            return

        self._unwrap_no_grad_forward(preprocessor)
        featurizer = getattr(preprocessor, "featurizer", None)
        if featurizer is None:
            return

        if hasattr(featurizer, "use_grads"):
            featurizer.use_grads = True
        self._unwrap_no_grad_forward(featurizer)

    @staticmethod
    def _unwrap_no_grad_forward(module: nn.Module) -> None:
        if not hasattr(module.forward, "__wrapped__"):
            return
        wrapped = inspect.unwrap(module.forward)
        if getattr(wrapped, "__self__", None) is module:
            module.forward = wrapped
        else:
            module.forward = MethodType(wrapped, module)
