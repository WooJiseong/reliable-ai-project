import inspect
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


class NemoEncoderCharCTC(nn.Module):
    def __init__(self, pretrained_model: str, vocab_size: int, freeze_encoder: bool = False):
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

        self.ctc_head = nn.Linear(int(encoder_dim), vocab_size)
        if freeze_encoder:
            for parameter in self.encoder_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor, waveform_length: torch.Tensor):
        encoded, encoded_lengths = self.encoder_model(input_signal=waveform, input_signal_length=waveform_length)
        encoded = encoded.transpose(1, 2)
        logits = self.ctc_head(encoded)
        return logits, encoded_lengths

    def log_probs(self, waveform: torch.Tensor, waveform_length: torch.Tensor):
        logits, encoded_lengths = self(waveform, waveform_length)
        return F.log_softmax(logits, dim=-1), encoded_lengths

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
