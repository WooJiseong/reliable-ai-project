#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.attacks.pgd import pgd_attack
from noise_robust_asr.data import TARGET_SAMPLE_RATE, read_manifest
from noise_robust_asr.metrics import cer, summarize_by_language, wer
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
        tokens = self.model.encode_text(item["text"])
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pretrained-model", default="nvidia/stt_en_conformer_ctc_small")
    parser.add_argument("--finetuned-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Enable random PGD initialization. Disabled by default to match ReliableAI_team_project.",
    )
    return parser.parse_args()


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def decode_batch(model: NemoConformerCTC, batch):
    log_probs, encoded_lengths = model(batch["waveform"], batch["waveform_length"])
    return model.decode_log_probs(log_probs, encoded_lengths)


def main():
    args = parse_args()
    model = NemoConformerCTC(args.pretrained_model).to(args.device)
    if args.finetuned_checkpoint is not None:
        checkpoint = torch.load(args.finetuned_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = NemoSpeechManifestDataset(args.manifest, model)
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
    for batch in tqdm(loader, desc="nemo-pgd-eval"):
        batch = move_batch(batch, args.device)

        clean_hyps = decode_batch(model, batch)
        attacked_waveform = pgd_attack(
            model,
            batch,
            blank_id=model.blank_id,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.pgd_steps,
            random_start=args.random_start,
        )
        attacked_batch = dict(batch)
        attacked_batch["waveform"] = attacked_waveform
        attacked_hyps = decode_batch(model, attacked_batch)

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
                    "model_family": "nemo_conformer_ctc",
                    "pretrained_model": args.pretrained_model,
                    "finetuned_checkpoint": args.finetuned_checkpoint or "",
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
        "pretrained_model",
        "finetuned_checkpoint",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_by_language(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
