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
from noise_robust_asr.metrics import cer, summarize_by_language, wer
from noise_robust_asr.models.conformer_ctc import build_conformer_ctc, load_conformer_ctc_checkpoint_state
from noise_robust_asr.text import CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
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


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def decode_batch(model, tokenizer, batch):
    logits, _ = model(batch["waveform"], batch["waveform_length"])
    return tokenizer.ctc_decode(logits)


def main():
    args = parse_args()
    if args.no_random_start:
        args.random_start = False
    tokenizer = CharTokenizer.load(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_conformer_ctc(tokenizer.vocab_size, size=checkpoint.get("size", "small")).to(args.device)
    load_conformer_ctc_checkpoint_state(model, checkpoint["model_state"])
    model.eval()

    dataset = SpeechManifestDataset(args.manifest, tokenizer)
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
            clean_wer = wer(ref, clean_hyp)
            attacked_wer = wer(ref, attacked_hyp)
            clean_cer = cer(ref, clean_hyp)
            attacked_cer = cer(ref, attacked_hyp)
            rows.append(
                {
                    "audio": batch["audio"][idx],
                    "language": batch["language"][idx],
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
            )

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

    summary = summarize_by_language(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
