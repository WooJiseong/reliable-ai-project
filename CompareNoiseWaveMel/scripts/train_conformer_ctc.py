#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.data import SpeechManifestDataset, collate_batch, read_manifest
from noise_robust_asr.metrics import cer, wer
from noise_robust_asr.models.conformer_ctc import build_conformer_ctc, conformer_ctc_checkpoint_state
from noise_robust_asr.text import CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def validate(model, loader, tokenizer, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits, _ = model(batch["waveform"], batch["waveform_length"])
        hyps = tokenizer.ctc_decode(logits)
        for ref, hyp in zip(batch["text"], hyps):
            rows.append({"wer": wer(ref, hyp), "cer": cer(ref, hyp)})
    return {
        "wer": sum(row["wer"] for row in rows) / max(len(rows), 1),
        "cer": sum(row["cer"] for row in rows) / max(len(rows), 1),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = read_manifest(args.train_manifest)
    tokenizer = CharTokenizer.build([item["text"] for item in train_items])
    tokenizer.save(output_dir / "vocab.json")

    train_dataset = SpeechManifestDataset(args.train_manifest, tokenizer)
    valid_dataset = SpeechManifestDataset(args.valid_manifest, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = build_conformer_ctc(tokenizer.vocab_size, size=args.size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_wer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in progress:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths = model(batch["waveform"], batch["waveform_length"])
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs,
                batch["tokens"],
                output_lengths,
                batch["token_length"],
                blank=tokenizer.blank_id,
                zero_infinity=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        metrics = validate(model, valid_loader, tokenizer, args.device)
        epoch_state = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "valid": metrics,
        }
        print(json.dumps(epoch_state, ensure_ascii=False))

        if metrics["wer"] < best_wer:
            best_wer = metrics["wer"]
            torch.save(
                {
                    "model_state": conformer_ctc_checkpoint_state(model),
                    "size": args.size,
                    "vocab_size": tokenizer.vocab_size,
                    "epoch": epoch,
                    "valid": metrics,
                },
                output_dir / "checkpoint.pt",
            )


if __name__ == "__main__":
    main()
