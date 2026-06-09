from __future__ import annotations

from itertools import groupby
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn


PROJECT_LANGUAGE_TO_OWSM = {
    "en": "<eng>",
    "ko": "<kor>",
    "zh": "<zho>",
    "ru": "<rus>",
}


class OWSMCTC(nn.Module):
    """Thin differentiable wrapper around ESPnet OWSM-CTC models."""

    def __init__(
        self,
        model_tag: str = "espnet/owsm_ctc_v4_1B",
        device: str = "cpu",
        dtype: str = "float16",
        task_symbol: str = "<asr>",
        fixed_audio_seconds: Optional[float] = 30.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        try:
            from espnet2.tasks.s2t_ctc import S2TTask
            from espnet2.text.build_tokenizer import build_tokenizer
            from espnet2.text.token_id_converter import TokenIDConverter
            from espnet_model_zoo.downloader import ModelDownloader
        except ImportError as exc:
            raise ImportError(
                "ESPnet and espnet_model_zoo are required for OWSMCTC. "
                "Install with: python -m pip install -r requirements-espnet.txt"
            ) from exc

        self.model_tag = model_tag
        self.device_name = device
        self.dtype_name = dtype
        self.task_symbol = task_symbol
        self.fixed_audio_seconds = fixed_audio_seconds

        downloaded = ModelDownloader().download_and_unpack(model_tag)
        model, train_args = S2TTask.build_model_from_file(
            downloaded["s2t_train_config"],
            downloaded["s2t_model_file"],
            device,
        )
        model = model.to(dtype=getattr(torch, dtype)).eval()
        for module in model.modules():
            if hasattr(module, "use_flash_attn"):
                setattr(module, "use_flash_attn", use_flash_attn)

        token_type = getattr(train_args, "token_type", None)
        bpemodel = getattr(train_args, "bpemodel", None)
        if token_type is None:
            tokenizer = None
        elif token_type in {"bpe", "hugging_face"} or "whisper" in token_type:
            tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        self.s2t_model = model
        self.train_args = train_args
        self.tokenizer = tokenizer
        self.converter = TokenIDConverter(token_list=model.token_list)
        self.blank_id = int(model.blank_id)
        self.vocab_size = len(model.token_list)

        sample_rate = getattr(train_args, "frontend_conf", {}).get("fs", 16000)
        self.sample_rate = int(sample_rate) if not isinstance(sample_rate, str) else 16000
        if self.fixed_audio_seconds is None:
            self.fixed_audio_samples = None
        else:
            self.fixed_audio_samples = int(round(self.sample_rate * self.fixed_audio_seconds))

    def resolve_language_symbol(self, language: str) -> str:
        symbol = PROJECT_LANGUAGE_TO_OWSM.get(language, language)
        if not (symbol.startswith("<") and symbol.endswith(">")):
            symbol = f"<{symbol}>"
        if symbol not in self.converter.token2id:
            raise ValueError(
                f"OWSM model {self.model_tag!r} does not support language symbol "
                f"{symbol!r} for project language {language!r}."
            )
        return symbol

    def supported_project_languages(self, languages: Iterable[str]) -> list[str]:
        supported = []
        for language in languages:
            try:
                self.resolve_language_symbol(language)
            except ValueError:
                continue
            supported.append(language)
        return supported

    def encode_text(self, text: str, language: Optional[str] = None) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError(f"OWSM model {self.model_tag!r} does not expose a text tokenizer.")
        token_ids = self.converter.tokens2ids(self.tokenizer.text2tokens(text.strip()))
        return torch.tensor(token_ids, dtype=torch.long)

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
        language=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, waveform_length = self._prepare_waveform(waveform, waveform_length)
        batch = self._build_batch(waveform, waveform_length, language)
        encoded, encoded_lengths = self.s2t_model.encode(**batch)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        log_probs = self.s2t_model.ctc.log_softmax(encoded)
        return log_probs.float(), encoded_lengths

    def log_probs(self, waveform: torch.Tensor, waveform_length: torch.Tensor, language=None):
        return self(waveform, waveform_length, language=language)

    def extract_features(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, waveform_length = self._prepare_waveform(waveform, waveform_length)
        with torch.cuda.amp.autocast(False):
            features, feature_lengths = self.s2t_model._extract_feats(
                waveform,
                waveform_length,
            )
        return features, feature_lengths

    def log_probs_from_features(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        language=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        prompt = self._build_prompt(batch_size, features.device, language)

        with torch.cuda.amp.autocast(False):
            if self.s2t_model.normalize is not None:
                features, feature_lengths = self.s2t_model.normalize(
                    features,
                    feature_lengths,
                )

        encoded, encoded_lengths, _ = self.s2t_model.encoder(
            features,
            feature_lengths,
            ctc=self.s2t_model.ctc,
            prefix_embeds=prompt["prefix_embeds"],
            memory=prompt["memory"],
            memory_mask=prompt["memory_mask"],
        )
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        log_probs = self.s2t_model.ctc.log_softmax(encoded)
        return log_probs.float(), encoded_lengths

    @torch.no_grad()
    def ctc_decode(self, log_probs: torch.Tensor) -> list[str]:
        predictions = torch.argmax(log_probs, dim=-1)
        results = []
        for sequence in predictions.cpu().tolist():
            token_ids = [
                token_id
                for token_id, _ in groupby(sequence)
                if token_id != self.blank_id
            ]
            tokens = self.converter.ids2tokens(token_ids)
            tokens = [
                token
                for token in tokens
                if not (token.startswith("<") and token.endswith(">"))
            ]
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(tokens)
            else:
                text = "".join(tokens)
            results.append((text or "").strip())
        return results

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = waveform.to(dtype=getattr(torch, self.dtype_name))
        if self.fixed_audio_samples is None:
            return waveform, waveform_length

        current = waveform.size(1)
        target = self.fixed_audio_samples
        if current < target:
            waveform = F.pad(waveform, (0, target - current))
        elif current > target:
            waveform = waveform[:, :target]
        waveform_length = waveform_length.new_full((waveform.size(0),), target)
        return waveform, waveform_length

    def _build_batch(self, waveform: torch.Tensor, waveform_length: torch.Tensor, language):
        batch_size = waveform.size(0)
        prompt_tokens = self._build_prompt_tokens(batch_size, waveform.device, language)
        return {
            "speech": waveform,
            "speech_lengths": waveform_length,
            **prompt_tokens,
        }

    def _build_prompt_tokens(self, batch_size: int, device: torch.device, language):
        if language is None:
            language = ["en"] * batch_size
        elif isinstance(language, str):
            language = [language] * batch_size

        text_prev = torch.tensor(
            [[self.s2t_model.na]],
            dtype=torch.long,
            device=device,
        ).repeat(batch_size, 1)
        text_prev_lengths = torch.full(
            (batch_size,),
            text_prev.size(1),
            dtype=torch.long,
            device=device,
        )

        task_id = self.converter.token2id[self.task_symbol]
        prefix = torch.tensor(
            [
                [self.converter.token2id[self.resolve_language_symbol(item)], task_id]
                for item in language
            ],
            dtype=torch.long,
            device=device,
        )
        prefix_lengths = torch.full(
            (batch_size,),
            prefix.size(1),
            dtype=torch.long,
            device=device,
        )

        return {
            "text_prev": text_prev,
            "text_prev_lengths": text_prev_lengths,
            "prefix": prefix,
            "prefix_lengths": prefix_lengths,
        }

    def _build_prompt(self, batch_size: int, device: torch.device, language):
        from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

        tokens = self._build_prompt_tokens(batch_size, device, language)
        text_prev = tokens["text_prev"]
        text_prev_lengths = tokens["text_prev_lengths"]
        prefix = tokens["prefix"]

        text_prev = text_prev.masked_fill(text_prev == -1, self.s2t_model.eos)
        memory, memory_lengths, _ = self.s2t_model.prompt_encoder(
            self.s2t_model.pos_enc(self.s2t_model.embed(text_prev)),
            text_prev_lengths,
        )
        memory_mask = (~make_pad_mask(memory_lengths)[:, None, :]).to(memory.device)
        return {
            "memory": self.s2t_model.prompt_proj(memory),
            "memory_mask": memory_mask,
            "prefix_embeds": self.s2t_model.embed_proj(self.s2t_model.embed(prefix)),
        }
