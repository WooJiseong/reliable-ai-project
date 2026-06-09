#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.data import TARGET_SAMPLE_RATE, read_manifest
from noise_robust_asr.metrics import cer, wer
from noise_robust_asr.models.nemo_conformer_ctc import NemoConformerCTC


class NemoSpeechManifestDataset(Dataset):
    def __init__(self, manifest_path, model: NemoConformerCTC):
        self.items = read_manifest(manifest_path)
        self.model = model

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torchaudio

        item = self.items[idx]
        waveform, sample_rate = torchaudio.load(item["audio"])
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
        waveform = waveform.clamp(-1.0, 1.0)
        tokens = self.model.encode_text(item["text"], item["language"])
        return {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "tokens": tokens,
            "token_length": torch.tensor(tokens.numel(), dtype=torch.long),
            "text": item["text"],
            "language": item["language"],
            "audio": item["audio"],
        }


def collate_batch(batch):
    return {
        "waveform": pad_sequence([item["waveform"] for item in batch], batch_first=True),
        "waveform_length": torch.stack([item["waveform_length"] for item in batch]),
        "tokens": torch.cat([item["tokens"] for item in batch]),
        "token_length": torch.stack([item["token_length"] for item in batch]),
        "text": [item["text"] for item in batch],
        "language": [item["language"] for item in batch],
        "audio": [item["audio"] for item in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-model", default="nvidia/stt_multilingual_fastconformer_hybrid_large_pc")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
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


def ctc_loss(model, batch):
    log_probs, output_lengths = model(batch["waveform"], batch["waveform_length"])
    return F.ctc_loss(
        log_probs.transpose(0, 1),
        batch["tokens"],
        output_lengths,
        batch["token_length"],
        blank=model.blank_id,
        zero_infinity=True,
    )


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = []
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss = ctc_loss(model, batch)
        losses.append(loss.item())
        log_probs, output_lengths = model(batch["waveform"], batch["waveform_length"])
        hyps = model.decode_log_probs(log_probs, output_lengths)
        for ref, hyp, language in zip(batch["text"], hyps, batch["language"]):
            rows.append({"wer": wer(ref, hyp, language), "cer": cer(ref, hyp)})

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "wer": sum(row["wer"] for row in rows) / max(len(rows), 1),
        "cer": sum(row["cer"] for row in rows) / max(len(rows), 1),
    }


def maybe_subset(dataset, limit):
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("dataset limit must be positive when provided")
    return Subset(dataset, range(min(limit, len(dataset))))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NemoConformerCTC(args.pretrained_model, freeze=False).to(args.device)
    if args.freeze_encoder:
        encoder = getattr(model.model, "encoder", None)
        if encoder is not None:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    train_dataset = maybe_subset(NemoSpeechManifestDataset(args.train_manifest, model), args.limit_train)
    valid_dataset = maybe_subset(NemoSpeechManifestDataset(args.valid_manifest, model), args.limit_valid)
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

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_wer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"nemo-ft epoch {epoch}")
        for batch in progress:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_postfix(loss=loss.item())

        metrics = validate(model, valid_loader, args.device)
        epoch_state = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "valid": metrics,
            "pretrained_model": args.pretrained_model,
        }
        print(json.dumps(epoch_state, ensure_ascii=False))

        torch.save(
            {
                "model_state": model.state_dict(),
                "pretrained_model": args.pretrained_model,
                "epoch": epoch,
                "valid": metrics,
            },
            output_dir / "last.pt",
        )
        if metrics["wer"] < best_wer:
            best_wer = metrics["wer"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "pretrained_model": args.pretrained_model,
                    "epoch": epoch,
                    "valid": metrics,
                },
                output_dir / "checkpoint.pt",
            )


if __name__ == "__main__":
    main()
