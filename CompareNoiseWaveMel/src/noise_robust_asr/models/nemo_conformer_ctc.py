import inspect
from typing import List
from types import MethodType

import torch
from torch import nn


class NemoConformerCTC(nn.Module):
    def __init__(self, pretrained_model: str, freeze: bool = True):
        super().__init__()
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NeMo is required for NemoConformerCTC. "
                "Install the ASR extra with: python -m pip install 'nemo_toolkit[asr]'"
            ) from exc

        self.pretrained_model = pretrained_model
        self.model = ASRModel.from_pretrained(model_name=pretrained_model)
        self._enable_input_gradients()
        self.model.eval()
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _enable_input_gradients(self) -> None:
        preprocessor = getattr(self.model, "preprocessor", None)
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

    @property
    def blank_id(self) -> int:
        decoder = self._ctc_decoder()
        vocabulary = getattr(decoder, "vocabulary", None)
        if vocabulary is not None:
            return len(vocabulary)

        num_classes_with_blank = getattr(decoder, "num_classes_with_blank", None)
        if num_classes_with_blank is not None:
            return int(num_classes_with_blank) - 1

        raise RuntimeError("Could not infer NeMo CTC blank id from the decoder.")

    def forward(self, waveform: torch.Tensor, waveform_length: torch.Tensor):
        output = self.model(input_signal=waveform, input_signal_length=waveform_length)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                f"{self.pretrained_model} did not return a NeMo ASR output tuple."
            )
        first, encoded_lengths = output[0], output[1]
        if hasattr(self.model, "ctc_decoder"):
            log_probs = self.model.ctc_decoder(encoder_output=first)
        else:
            log_probs = first
        return log_probs, encoded_lengths

    def encode_text(self, text: str) -> torch.Tensor:
        normalized = text.strip().lower()
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            token_ids = tokenizer.text_to_ids(normalized)
            return torch.tensor(token_ids, dtype=torch.long)

        vocabulary = getattr(self._ctc_decoder(), "vocabulary", None)
        if vocabulary is None:
            raise RuntimeError("Could not find a NeMo tokenizer or decoder vocabulary.")

        token_to_id = {token: idx for idx, token in enumerate(vocabulary)}
        token_ids = [token_to_id[token] for token in normalized if token in token_to_id]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode_log_probs(self, log_probs: torch.Tensor, encoded_lengths: torch.Tensor) -> List[str]:
        predictions = log_probs.argmax(dim=-1)
        results = []
        for sequence, length in zip(predictions.cpu().tolist(), encoded_lengths.cpu().tolist()):
            collapsed = []
            previous = None
            for idx in sequence[:length]:
                if idx != previous and idx != self.blank_id:
                    collapsed.append(idx)
                previous = idx
            results.append(self.decode_ids(collapsed))
        return results

    def decode_ids(self, token_ids: List[int]) -> str:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            if hasattr(tokenizer, "ids_to_text"):
                return tokenizer.ids_to_text(token_ids).strip()
            if hasattr(tokenizer, "ids_to_tokens"):
                tokens = tokenizer.ids_to_tokens(token_ids)
                return "".join(tokens).replace("▁", " ").strip()

        vocabulary = getattr(self._ctc_decoder(), "vocabulary", None)
        if vocabulary is None:
            raise RuntimeError("Could not find a NeMo tokenizer or decoder vocabulary.")
        return "".join(vocabulary[idx] for idx in token_ids if idx < len(vocabulary)).strip()

    def _ctc_decoder(self):
        decoder = getattr(self.model, "ctc_decoder", None)
        if decoder is not None:
            return decoder
        decoder = getattr(self.model, "decoder", None)
        if decoder is not None:
            return decoder
        raise RuntimeError("The loaded NeMo model does not expose a CTC decoder.")
