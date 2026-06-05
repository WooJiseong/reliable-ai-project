#!/usr/bin/env python
import argparse
import json
import re
import sys
from pathlib import Path


LANGUAGES = ["ko", "en", "zh", "ru"]
TARGET_SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ReliableAI_team_project").exists():
            return parent
    return PROJECT_ROOT.parent


WORKSPACE_ROOT = find_workspace_root()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert ReliableAI_team_project's saved Hugging Face FLEURS "
            "dataset into CompareNoiseWaveMel JSONL manifests."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(WORKSPACE_ROOT / "ReliableAI_team_project" / "data" / "waveform"),
        help="Path to the dataset saved by ReliableAI_team_project/src/data_load.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "team_fleurs"),
        help="Directory for wav files and JSONL manifests.",
    )
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--samples-per-lang", type=int, default=1000)
    parser.add_argument("--train-per-lang", type=int, default=800)
    parser.add_argument("--valid-per-lang", type=int, default=100)
    parser.add_argument("--test-per-lang", type=int, default=100)
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Rewrite wav files even when they already exist.",
    )
    return parser.parse_args()


def safe_text_id(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^0-9a-zA-Z가-힣一-龥а-яА-Я_]+", "", text)
    return text[:32] or "sample"


def as_mono_waveform(audio: dict):
    import torch

    waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    return waveform.flatten().clamp(-1.0, 1.0)


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    import torchaudio
    from datasets import load_from_disk
    from tqdm import tqdm

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    split_total = args.train_per_lang + args.valid_per_lang + args.test_per_lang
    if split_total > args.samples_per_lang:
        raise ValueError(
            "train-per-lang + valid-per-lang + test-per-lang must be <= samples-per-lang"
        )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"{dataset_dir} does not exist. Run ReliableAI_team_project/src/data_load.py first."
        )

    dataset = load_from_disk(str(dataset_dir))
    required = {"audio", "raw_transcription", "lang_tag"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{dataset_dir} missing dataset columns: {sorted(missing)}")

    grouped = {language: [] for language in args.languages}
    for dataset_index, sample in enumerate(dataset):
        language = sample["lang_tag"]
        if language not in grouped:
            continue
        if len(grouped[language]) >= args.samples_per_lang:
            continue
        grouped[language].append((dataset_index, sample))
        if all(len(items) >= args.samples_per_lang for items in grouped.values()):
            break

    for language, items in grouped.items():
        if len(items) < args.samples_per_lang:
            raise ValueError(
                f"Only found {len(items)} samples for {language}; "
                f"requested {args.samples_per_lang}."
            )

    all_rows = []
    splits = {"train": [], "valid": [], "test": []}
    split_bounds = {
        "train": (0, args.train_per_lang),
        "valid": (args.train_per_lang, args.train_per_lang + args.valid_per_lang),
        "test": (
            args.train_per_lang + args.valid_per_lang,
            args.train_per_lang + args.valid_per_lang + args.test_per_lang,
        ),
    }

    for language in args.languages:
        for rank, (dataset_index, sample) in enumerate(
            tqdm(grouped[language], desc=f"export {language}")
        ):
            audio = sample["audio"]
            waveform = as_mono_waveform(audio)
            sample_rate = int(audio["sampling_rate"])
            if sample_rate != TARGET_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, TARGET_SAMPLE_RATE
                )

            text = sample["raw_transcription"]
            wav_path = audio_dir / f"{language}_{rank:05d}_{dataset_index}_{safe_text_id(text)}.wav"
            if args.overwrite_audio or not wav_path.exists():
                torchaudio.save(str(wav_path), waveform.unsqueeze(0), TARGET_SAMPLE_RATE)

            row = {
                "audio": str(Path("audio") / wav_path.name),
                "text": text,
                "language": language,
                "source_dataset_index": dataset_index,
                "language_rank": rank,
                "sample_rate": TARGET_SAMPLE_RATE,
            }
            all_rows.append(row)

            for split_name, (start, end) in split_bounds.items():
                if start <= rank < end:
                    splits[split_name].append(row)

    write_jsonl(output_dir / "all.jsonl", all_rows)
    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    metadata = {
        "source_dataset": str(dataset_dir),
        "output_dir": str(output_dir),
        "languages": args.languages,
        "samples_per_lang": args.samples_per_lang,
        "train_per_lang": args.train_per_lang,
        "valid_per_lang": args.valid_per_lang,
        "test_per_lang": args.test_per_lang,
        "sample_rate": TARGET_SAMPLE_RATE,
        "selection": "preserves ReliableAI_team_project dataset order within each language",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
