#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.metrics import PhonemeMetric, cer, wer


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ReliableAI_team_project").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Wav MMS JSONL results and Mel Conformer CSV results with shared metrics."
    )
    parser.add_argument(
        "--wave-jsonl",
        default=str(
            find_workspace_root()
            / "ReliableAI_team_project"
            / "data"
            / "attack_results"
            / "all_results.jsonl"
        ),
    )
    parser.add_argument("--mel-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--phonemizer-backend", default="espeak")
    parser.add_argument(
        "--phonemizer-language-map",
        default=None,
        help="JSON object or JSON file mapping project language tags to phonemizer language codes.",
    )
    parser.add_argument(
        "--disable-phoneme-metric",
        action="store_true",
        help="Skip phonemizer PER summary. Use only for legacy/debug runs.",
    )
    return parser.parse_args()


def mean(values):
    values = list(values)
    return sum(values) / max(len(values), 1)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    summary = {}
    for language, items in sorted(grouped.items()):
        clean_wer = mean(item["clean_wer"] for item in items)
        attacked_wer = mean(item["attacked_wer"] for item in items)
        clean_cer = mean(item["clean_cer"] for item in items)
        attacked_cer = mean(item["attacked_cer"] for item in items)
        summary[language] = {
            "count": len(items),
            "clean_wer": clean_wer,
            "attacked_wer": attacked_wer,
            "wer_degradation": attacked_wer - clean_wer,
            "clean_cer": clean_cer,
            "attacked_cer": attacked_cer,
            "cer_degradation": attacked_cer - clean_cer,
        }
        if all("clean_per" in item and "attacked_per" in item for item in items):
            clean_per = mean(item["clean_per"] for item in items)
            attacked_per = mean(item["attacked_per"] for item in items)
            summary[language].update(
                {
                    "clean_per": clean_per,
                    "attacked_per": attacked_per,
                    "per_degradation": attacked_per - clean_per,
                }
            )

    if summary:
        wer_degradations = [item["wer_degradation"] for item in summary.values()]
        cer_degradations = [item["cer_degradation"] for item in summary.values()]
        gap = {
            "wer_robustness_gap": max(wer_degradations) - min(wer_degradations),
            "cer_robustness_gap": max(cer_degradations) - min(cer_degradations),
        }
        per_degradations = [
            item["per_degradation"]
            for item in summary.values()
            if "per_degradation" in item
        ]
        if per_degradations:
            gap["per_robustness_gap"] = max(per_degradations) - min(per_degradations)
        for item in summary.values():
            item.update(gap)
    return summary


def add_phoneme_metrics(row, phoneme_metric):
    if phoneme_metric is None:
        return row
    language = row["language"]
    reference = row["reference"]
    clean_prediction = row["clean_prediction"]
    attacked_prediction = row["attacked_prediction"]
    clean_per = phoneme_metric.per(reference, clean_prediction, language)
    attacked_per = phoneme_metric.per(reference, attacked_prediction, language)
    row.update(
        {
            "reference_phonemes": phoneme_metric.phonemize_text(reference, language),
            "clean_prediction_phonemes": phoneme_metric.phonemize_text(clean_prediction, language),
            "attacked_prediction_phonemes": phoneme_metric.phonemize_text(attacked_prediction, language),
            "clean_per": clean_per,
            "attacked_per": attacked_per,
            "per_degradation": attacked_per - clean_per,
        }
    )
    return row


def optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def load_language_map(value):
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def read_wave_rows(path, phoneme_metric, languages=None):
    languages = set(languages or [])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if languages and item["lang_tag"] not in languages:
                continue
            reference = item["ground_truth"]
            clean_prediction = item["clean_pred"]
            attacked_prediction = item["adv_pred"]
            row = {
                "model_family": "wav_mms_ctc",
                "language": item["lang_tag"],
                "reference": reference,
                "clean_prediction": clean_prediction,
                "attacked_prediction": attacked_prediction,
                "clean_wer": wer(reference, clean_prediction, item["lang_tag"]),
                "attacked_wer": wer(reference, attacked_prediction, item["lang_tag"]),
                "clean_cer": cer(reference, clean_prediction),
                "attacked_cer": cer(reference, attacked_prediction),
            }
            rows.append(add_phoneme_metrics(row, phoneme_metric))
    return rows


def read_mel_rows(path, phoneme_metric, languages=None):
    languages = set(languages or [])
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            if languages and item["language"] not in languages:
                continue
            reference = item["reference"]
            clean_prediction = item["clean_prediction"]
            attacked_prediction = item["attacked_prediction"]
            row = {
                "model_family": item.get("model_family") or "mel_conformer_ctc",
                "language": item["language"],
                "reference": reference,
                "clean_prediction": clean_prediction,
                "attacked_prediction": attacked_prediction,
                "clean_wer": wer(reference, clean_prediction, item["language"]),
                "attacked_wer": wer(reference, attacked_prediction, item["language"]),
                "clean_cer": float(item["clean_cer"]),
                "attacked_cer": float(item["attacked_cer"]),
            }
            clean_per = optional_float(item.get("clean_per"))
            attacked_per = optional_float(item.get("attacked_per"))
            if phoneme_metric is not None:
                row = add_phoneme_metrics(row, phoneme_metric)
            elif clean_per is not None and attacked_per is not None:
                row.update(
                    {
                        "reference_phonemes": item.get("reference_phonemes", ""),
                        "clean_prediction_phonemes": item.get("clean_prediction_phonemes", ""),
                        "attacked_prediction_phonemes": item.get("attacked_prediction_phonemes", ""),
                        "clean_per": clean_per,
                        "attacked_per": attacked_per,
                        "per_degradation": attacked_per - clean_per,
                    }
                )
            rows.append(row)
    return rows


def main():
    args = parse_args()
    phoneme_metric = None
    if not args.disable_phoneme_metric:
        phoneme_metric = PhonemeMetric(
            backend=args.phonemizer_backend,
            language_map=load_language_map(args.phonemizer_language_map),
        )
    wave_rows = read_wave_rows(args.wave_jsonl, phoneme_metric, languages=args.languages)
    mel_rows = read_mel_rows(args.mel_csv, phoneme_metric, languages=args.languages)
    result = {
        "wav_mms_ctc": summarize(wave_rows),
        "mel_conformer_ctc": summarize(mel_rows),
        "controls": {
            "dataset": "ReliableAI_team_project FLEURS save_to_disk export",
            "sample_rate": 16000,
            "attack": "untargeted PGD-5, L-infinity waveform perturbation",
            "epsilon": 0.005,
            "alpha": 0.001,
            "random_start": False,
            "primary_metric": "phoneme_error_rate",
            "phonemizer_backend": None if phoneme_metric is None else args.phonemizer_backend,
            "languages": args.languages,
        },
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    sys.exit(main())
