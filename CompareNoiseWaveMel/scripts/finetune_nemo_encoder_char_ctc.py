#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.data import SpeechManifestDataset, collate_batch, read_manifest
from noise_robust_asr.metrics import cer, wer
from noise_robust_asr.models.nemo_encoder_char_ctc import NemoEncoderCharCTC
from noise_robust_asr.text import CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-model", default="stt_multilingual_fastconformer_hybrid_large_pc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def maybe_subset(dataset, limit):
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("dataset limit must be positive when provided")
    return Subset(dataset, range(min(limit, len(dataset))))


def ctc_loss(model, tokenizer, batch):
    logits, output_lengths = model(batch["waveform"], batch["waveform_length"])
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        batch["tokens"],
        output_lengths,
        batch["token_length"],
        blank=tokenizer.blank_id,
        zero_infinity=True,
    )


@torch.no_grad()
def validate(model, loader, tokenizer, device):
    model.eval()
    losses = []
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        losses.append(ctc_loss(model, tokenizer, batch).item())
        logits, _ = model(batch["waveform"], batch["waveform_length"])
        hyps = tokenizer.ctc_decode(logits)
        for ref, hyp in zip(batch["text"], hyps):
            rows.append({"wer": wer(ref, hyp), "cer": cer(ref, hyp)})
    return {
        "loss": sum(losses) / max(len(losses), 1),
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

    model = NemoEncoderCharCTC(
        pretrained_model=args.pretrained_model,
        vocab_size=tokenizer.vocab_size,
        freeze_encoder=args.freeze_encoder,
    ).to(args.device)

    train_dataset = maybe_subset(SpeechManifestDataset(args.train_manifest, tokenizer), args.limit_train)
    valid_dataset = maybe_subset(SpeechManifestDataset(args.valid_manifest, tokenizer), args.limit_valid)
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

    encoder_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("ctc_head."):
            head_params.append(parameter)
        else:
            encoder_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_wer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"nemo-char-ft epoch {epoch}")
        for batch in progress:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss(model, tokenizer, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_postfix(loss=loss.item())

        metrics = validate(model, valid_loader, tokenizer, args.device)
        epoch_state = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "valid": metrics,
            "pretrained_model": args.pretrained_model,
            "vocab_size": tokenizer.vocab_size,
        }
        print(json.dumps(epoch_state, ensure_ascii=False))

        checkpoint = {
            "model_state": model.state_dict(),
            "pretrained_model": args.pretrained_model,
            "vocab_size": tokenizer.vocab_size,
            "epoch": epoch,
            "valid": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["wer"] < best_wer:
            best_wer = metrics["wer"]
            torch.save(checkpoint, output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
