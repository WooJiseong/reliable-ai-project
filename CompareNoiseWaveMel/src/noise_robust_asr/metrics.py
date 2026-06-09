import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional

from noise_robust_asr.text import normalize_text


PHONEMIZER_LANGUAGE_CODES = {
    "en": "en-us",
    "ko": "ko",
    "zh": "cmn",
    "ru": "ru",
}


def _edit_distance(reference: List[str], hypothesis: List[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _is_chinese_language(language: Optional[str]) -> bool:
    if not language:
        return False
    return language.lower().replace("-", "_") in {"zh", "zh_cn", "cmn", "cmn_hans_cn"}


def segment_text_for_metrics(text: str, language: Optional[str] = None) -> str:
    normalized = normalize_text(text)
    if not normalized or not _is_chinese_language(language):
        return normalized

    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError(
            "Chinese WER/PER metrics require Jieba segmentation. "
            "Install it with `pip install jieba`."
        ) from exc

    tokens = []
    for token in jieba.cut(normalized):
        token = token.strip()
        if token and re.search(r"[\w\u4e00-\u9fff]", token, flags=re.UNICODE):
            tokens.append(token)
    return " ".join(tokens)


def wer(reference: str, hypothesis: str, language: Optional[str] = None) -> float:
    ref_words = segment_text_for_metrics(reference, language).split()
    hyp_words = segment_text_for_metrics(hypothesis, language).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


class PhonemeMetric:
    """Language-aware phonemizer wrapper for phoneme error rate."""

    def __init__(
        self,
        backend: str = "espeak",
        language_map: Optional[Dict[str, str]] = None,
    ):
        self.backend = backend
        self.language_map = dict(PHONEMIZER_LANGUAGE_CODES)
        if language_map:
            self.language_map.update(language_map)
        self._cache = {}

        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator
        except ImportError as exc:
            raise RuntimeError(
                "phonemizer is required for phoneme metrics. Install the Python "
                "package and an eSpeak/eSpeak-NG backend before running PER evaluation."
            ) from exc

        self._configure_espeak_loader()
        self._phonemize = phonemize
        self._espeak_backend_cls = EspeakBackend
        self._backends = {}
        self._separator = Separator(phone=" ", word=" | ", syllable="")

    @staticmethod
    def _configure_espeak_loader() -> None:
        if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            return
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return

        try:
            import espeakng_loader
        except ImportError:
            return

        if espeakng_loader.load_library() is None:
            return

        os.environ.setdefault(
            "PHONEMIZER_ESPEAK_LIBRARY",
            espeakng_loader.get_library_path(),
        )
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        espeakng_loader.make_library_available()

    def language_code(self, language: str) -> str:
        return self.language_map.get(language, language)

    def phonemize_text(self, text: str, language: str) -> str:
        text = segment_text_for_metrics(text, language)
        if not text:
            return ""

        cache_key = (language, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            language_code = self.language_code(language)
            if self.backend == "espeak":
                backend = self._backends.get(language_code)
                if backend is None:
                    backend = self._espeak_backend_cls(
                        language=language_code,
                        preserve_punctuation=False,
                    )
                    self._backends[language_code] = backend
                phonemes = backend.phonemize(
                    [text],
                    separator=self._separator,
                    strip=True,
                    njobs=1,
                )
            else:
                phonemes = self._phonemize(
                    text,
                    language=language_code,
                    backend=self.backend,
                    separator=self._separator,
                    strip=True,
                    preserve_punctuation=False,
                    njobs=1,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to phonemize text for language={language!r} "
                f"with backend={self.backend!r}. Install espeak/eSpeak-NG or "
                "set PHONEMIZER_ESPEAK_LIBRARY and ESPEAK_DATA_PATH to a "
                "compatible eSpeak-NG build."
            ) from exc

        if isinstance(phonemes, list):
            phonemes = phonemes[0] if phonemes else ""
        phonemes = re.sub(r"\s+", " ", str(phonemes)).strip()
        self._cache[cache_key] = phonemes
        return phonemes

    def tokens(self, text: str, language: str) -> List[str]:
        return [token for token in self.phonemize_text(text, language).split() if token != "|"]

    def per(self, reference: str, hypothesis: str, language: str) -> float:
        ref_tokens = self.tokens(reference, language)
        hyp_tokens = self.tokens(hypothesis, language)
        if not ref_tokens:
            return 0.0 if not hyp_tokens else 1.0
        return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def phoneme_error_rate(
    reference: str,
    hypothesis: str,
    language: str,
    phoneme_metric: Optional[PhonemeMetric] = None,
) -> float:
    metric = phoneme_metric or PhonemeMetric()
    return metric.per(reference, hypothesis, language)


def summarize_by_language(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    summary = {}
    for language, items in sorted(grouped.items()):
        clean_wer = sum(item["clean_wer"] for item in items) / len(items)
        attacked_wer = sum(item["attacked_wer"] for item in items) / len(items)
        clean_cer = sum(item["clean_cer"] for item in items) / len(items)
        attacked_cer = sum(item["attacked_cer"] for item in items) / len(items)
        summary[language] = {
            "count": float(len(items)),
            "clean_wer": clean_wer,
            "attacked_wer": attacked_wer,
            "wer_degradation": attacked_wer - clean_wer,
            "clean_cer": clean_cer,
            "attacked_cer": attacked_cer,
            "cer_degradation": attacked_cer - clean_cer,
        }
        if all("clean_per" in item and "attacked_per" in item for item in items):
            clean_per = sum(item["clean_per"] for item in items) / len(items)
            attacked_per = sum(item["attacked_per"] for item in items) / len(items)
            summary[language].update(
                {
                    "clean_per": clean_per,
                    "attacked_per": attacked_per,
                    "per_degradation": attacked_per - clean_per,
                }
            )

    if summary:
        wer_degradations = [metrics["wer_degradation"] for metrics in summary.values()]
        cer_degradations = [metrics["cer_degradation"] for metrics in summary.values()]
        gap = {
            "wer_robustness_gap": max(wer_degradations) - min(wer_degradations),
            "cer_robustness_gap": max(cer_degradations) - min(cer_degradations),
        }
        per_degradations = [
            metrics["per_degradation"]
            for metrics in summary.values()
            if "per_degradation" in metrics
        ]
        if per_degradations:
            gap["per_robustness_gap"] = max(per_degradations) - min(per_degradations)
        for metrics in summary.values():
            metrics.update(gap)

    return summary
