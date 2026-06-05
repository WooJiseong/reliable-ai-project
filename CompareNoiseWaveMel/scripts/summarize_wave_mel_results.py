#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.metrics import cer, wer


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
    return summary


def read_wave_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            reference = item["ground_truth"]
            clean_prediction = item["clean_pred"]
            attacked_prediction = item["adv_pred"]
            rows.append(
                {
                    "model_family": "wav_mms_ctc",
                    "language": item["lang_tag"],
                    "reference": reference,
                    "clean_prediction": clean_prediction,
                    "attacked_prediction": attacked_prediction,
                    "clean_wer": wer(reference, clean_prediction),
                    "attacked_wer": wer(reference, attacked_prediction),
                    "clean_cer": cer(reference, clean_prediction),
                    "attacked_cer": cer(reference, attacked_prediction),
                }
            )
    return rows


def read_mel_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            rows.append(
                {
                    "model_family": item.get("model_family") or "mel_conformer_ctc",
                    "language": item["language"],
                    "reference": item["reference"],
                    "clean_prediction": item["clean_prediction"],
                    "attacked_prediction": item["attacked_prediction"],
                    "clean_wer": float(item["clean_wer"]),
                    "attacked_wer": float(item["attacked_wer"]),
                    "clean_cer": float(item["clean_cer"]),
                    "attacked_cer": float(item["attacked_cer"]),
                }
            )
    return rows


def main():
    args = parse_args()
    wave_rows = read_wave_rows(args.wave_jsonl)
    mel_rows = read_mel_rows(args.mel_csv)
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
