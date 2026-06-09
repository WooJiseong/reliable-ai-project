#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.attacks.pgd import pgd_attack
from noise_robust_asr.data import SpeechManifestDataset, collate_batch
from noise_robust_asr.metrics import PhonemeMetric, cer, summarize_by_language, wer
from noise_robust_asr.models.conformer_ctc import build_conformer_ctc, load_conformer_ctc_checkpoint_state
from noise_robust_asr.text import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional ReliableAI_team_project-compatible JSONL output path.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--phonemizer-backend", default="espeak")
    parser.add_argument(
        "--phonemizer-language-map",
        default=None,
        help="JSON object or JSON file mapping project language tags to phonemizer language codes.",
    )
    parser.add_argument(
        "--disable-phoneme-metric",
        action="store_true",
        help="Skip phonemizer PER columns. Use only for legacy/debug runs.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Enable random PGD initialization. Disabled by default to match ReliableAI_team_project.",
    )
    parser.add_argument(
        "--no-random-start",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_language_map(value):
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA, but this PyTorch process cannot see a CUDA device; using CPU.",
            flush=True,
        )
        return "cpu"
    return requested


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def decode_batch(model, tokenizer, batch):
    logits, _ = model(batch["waveform"], batch["waveform_length"])
    return tokenizer.ctc_decode(logits)


def filter_dataset_by_language(dataset, languages):
    if not languages:
        return dataset
    languages = set(languages)
    indices = [
        index for index, item in enumerate(dataset.items)
        if item["language"] in languages
    ]
    return Subset(dataset, indices)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.no_random_start:
        args.random_start = False
    tokenizer = load_tokenizer(args.vocab)
    phoneme_metric = None
    if not args.disable_phoneme_metric:
        phoneme_metric = PhonemeMetric(
            backend=args.phonemizer_backend,
            language_map=load_language_map(args.phonemizer_language_map),
        )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_conformer_ctc(tokenizer.vocab_size, size=checkpoint.get("size", "small")).to(args.device)
    load_conformer_ctc_checkpoint_state(model, checkpoint["model_state"])
    model.eval()

    dataset = filter_dataset_by_language(
        SpeechManifestDataset(args.manifest, tokenizer),
        args.languages,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive when provided")
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    rows = []
    for batch in tqdm(loader, desc="pgd-eval"):
        batch = move_batch(batch, args.device)

        clean_hyps = decode_batch(model, tokenizer, batch)
        attacked_waveform = pgd_attack(
            model,
            batch,
            blank_id=tokenizer.blank_id,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.pgd_steps,
            random_start=args.random_start,
        )
        attacked_batch = dict(batch)
        attacked_batch["waveform"] = attacked_waveform
        attacked_hyps = decode_batch(model, tokenizer, attacked_batch)

        for idx, (ref, clean_hyp, attacked_hyp) in enumerate(zip(batch["text"], clean_hyps, attacked_hyps)):
            language = batch["language"][idx]
            clean_wer = wer(ref, clean_hyp, language)
            attacked_wer = wer(ref, attacked_hyp, language)
            clean_cer = cer(ref, clean_hyp)
            attacked_cer = cer(ref, attacked_hyp)
            row = {
                "audio": batch["audio"][idx],
                "language": language,
                "reference": ref,
                "clean_prediction": clean_hyp,
                "attacked_prediction": attacked_hyp,
                "clean_wer": clean_wer,
                "attacked_wer": attacked_wer,
                "wer_degradation": attacked_wer - clean_wer,
                "clean_cer": clean_cer,
                "attacked_cer": attacked_cer,
                "cer_degradation": attacked_cer - clean_cer,
                "pgd_steps": args.pgd_steps,
                "epsilon": args.epsilon,
                "alpha": args.alpha,
                "random_start": args.random_start,
                "model_family": "mel_conformer_ctc",
            }
            if phoneme_metric is not None:
                clean_per = phoneme_metric.per(ref, clean_hyp, language)
                attacked_per = phoneme_metric.per(ref, attacked_hyp, language)
                row.update(
                    {
                        "reference_phonemes": phoneme_metric.phonemize_text(ref, language),
                        "clean_prediction_phonemes": phoneme_metric.phonemize_text(clean_hyp, language),
                        "attacked_prediction_phonemes": phoneme_metric.phonemize_text(attacked_hyp, language),
                        "clean_per": clean_per,
                        "attacked_per": attacked_per,
                        "per_degradation": attacked_per - clean_per,
                    }
                )
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio",
        "language",
        "reference",
        "clean_prediction",
        "attacked_prediction",
        "clean_wer",
        "attacked_wer",
        "wer_degradation",
        "clean_cer",
        "attacked_cer",
        "cer_degradation",
        "reference_phonemes",
        "clean_prediction_phonemes",
        "attacked_prediction_phonemes",
        "clean_per",
        "attacked_per",
        "per_degradation",
        "pgd_steps",
        "epsilon",
        "alpha",
        "random_start",
        "model_family",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                item = {
                    "lang_tag": row["language"],
                    "ground_truth": row["reference"],
                    "clean_pred": row["clean_prediction"],
                    "adv_pred": row["attacked_prediction"],
                    "model_family": row["model_family"],
                    "clean_wer": row["clean_wer"],
                    "attacked_wer": row["attacked_wer"],
                    "clean_cer": row["clean_cer"],
                    "attacked_cer": row["attacked_cer"],
                }
                if "clean_per" in row:
                    item.update(
                        {
                            "reference_phonemes": row["reference_phonemes"],
                            "clean_pred_phonemes": row["clean_prediction_phonemes"],
                            "adv_pred_phonemes": row["attacked_prediction_phonemes"],
                            "clean_per": row["clean_per"],
                            "attacked_per": row["attacked_per"],
                            "per_degradation": row["per_degradation"],
                        }
                    )
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = summarize_by_language(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
