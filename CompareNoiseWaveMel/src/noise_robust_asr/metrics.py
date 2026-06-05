from collections import defaultdict
from typing import Dict, List

from noise_robust_asr.text import normalize_text


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


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


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

    if summary:
        wer_degradations = [metrics["wer_degradation"] for metrics in summary.values()]
        cer_degradations = [metrics["cer_degradation"] for metrics in summary.values()]
        gap = {
            "wer_robustness_gap": max(wer_degradations) - min(wer_degradations),
            "cer_robustness_gap": max(cer_degradations) - min(cer_degradations),
        }
        for metrics in summary.values():
            metrics.update(gap)

    return summary
