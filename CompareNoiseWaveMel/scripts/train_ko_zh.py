#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.data import SpeechManifestDataset, collate_batch
from noise_robust_asr.metrics import cer, wer
from noise_robust_asr.models.nemo_encoder_char_ctc import NemoEncoderCharCTC
from noise_robust_asr.text import load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue training a multilingual NeMo encoder + shared subword CTC checkpoint "
            "on ko/zh samples, then evaluate on the full test set."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-manifest",
        default=str(PROJECT_ROOT / "data" / "team_fleurs" / "train.jsonl"),
    )
    parser.add_argument(
        "--valid-manifest",
        default=str(PROJECT_ROOT / "data" / "team_fleurs" / "valid.jsonl"),
    )
    parser.add_argument(
        "--test-manifest",
        default=str(PROJECT_ROOT / "data" / "team_fleurs" / "test.jsonl"),
    )
    parser.add_argument("--train-languages", nargs="+", default=["ko", "zh"])
    parser.add_argument("--eval-languages", nargs="+", default=["ko", "en", "zh", "ru"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--adapter-dim",
        type=int,
        default=None,
        help="Adapter bottleneck size. Defaults to checkpoint adapter_dim, or 256.",
    )
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=0.0,
        help="Optional shared CTC head LR. Default 0 trains ko/zh adapters only.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--phonemizer-backend", default="espeak")
    parser.add_argument("--phonemizer-language-map", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--wave-jsonl",
        default=str(
            PROJECT_ROOT.parent
            / "ReliableAI_team_project"
            / "data"
            / "attack_results"
            / "all_results.jsonl"
        ),
    )
    return parser.parse_args()


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
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def filter_dataset_by_language(dataset, languages):
    languages = set(languages)
    indices = [
        index
        for index, item in enumerate(dataset.items)
        if item["language"] in languages
    ]
    return Subset(dataset, indices)


def maybe_limit(dataset, limit):
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("limit must be positive when provided")
    return Subset(dataset, range(min(limit, len(dataset))))


def ctc_loss(model, tokenizer, batch):
    logits, output_lengths = model(
        batch["waveform"],
        batch["waveform_length"],
        language=batch.get("language"),
    )
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
        logits, _ = model(
            batch["waveform"],
            batch["waveform_length"],
            language=batch.get("language"),
        )
        hyps = tokenizer.ctc_decode(logits)
        for ref, hyp, language in zip(batch["text"], hyps, batch["language"]):
            rows.append(
                {
                    "language": language,
                    "wer": wer(ref, hyp, language),
                    "cer": cer(ref, hyp),
                }
            )
    by_language = {}
    for language in sorted({row["language"] for row in rows}):
        items = [row for row in rows if row["language"] == language]
        by_language[language] = {
            "count": len(items),
            "wer": sum(item["wer"] for item in items) / max(len(items), 1),
            "cer": sum(item["cer"] for item in items) / max(len(items), 1),
        }
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "wer": sum(row["wer"] for row in rows) / max(len(rows), 1),
        "cer": sum(row["cer"] for row in rows) / max(len(rows), 1),
        "by_language": by_language,
    }


def build_model(checkpoint, tokenizer, args):
    checkpoint_languages = checkpoint.get("adapter_languages") or []
    adapter_languages = sorted(set(checkpoint_languages) | set(args.train_languages))
    adapter_dim = args.adapter_dim or checkpoint.get("adapter_dim") or 256
    model = NemoEncoderCharCTC(
        pretrained_model=checkpoint["pretrained_model"],
        vocab_size=tokenizer.vocab_size,
        freeze_encoder=True,
        adapter_languages=adapter_languages,
        adapter_dim=adapter_dim,
    ).to(args.device)

    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("adapters.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Could not load checkpoint cleanly: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
    return model


def configure_trainable_parameters(model, args):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_languages = set(args.train_languages)
    adapter_params = []
    for language, adapter in model.adapters.items():
        requires_grad = language in train_languages
        for parameter in adapter.parameters():
            parameter.requires_grad_(requires_grad)
            if requires_grad:
                adapter_params.append(parameter)

    parameter_groups = []
    if adapter_params:
        parameter_groups.append({"params": adapter_params, "lr": args.adapter_lr})

    if args.head_lr > 0:
        head_params = list(model.ctc_head.parameters())
        for parameter in head_params:
            parameter.requires_grad_(True)
        parameter_groups.append({"params": head_params, "lr": args.head_lr})

    if not parameter_groups:
        raise ValueError("No trainable parameters selected.")
    return parameter_groups


def run(command, cwd: Path):
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(checkpoint, tokenizer, args)

    train_dataset = filter_dataset_by_language(
        SpeechManifestDataset(args.train_manifest, tokenizer),
        args.train_languages,
    )
    valid_dataset = filter_dataset_by_language(
        SpeechManifestDataset(args.valid_manifest, tokenizer),
        args.train_languages,
    )
    train_dataset = maybe_limit(train_dataset, args.limit_train)
    valid_dataset = maybe_limit(valid_dataset, args.limit_valid)

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
        configure_trainable_parameters(model, args),
        weight_decay=args.weight_decay,
    )

    best_cer = float("inf")
    best_path = output_dir / "checkpoint.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"ko-zh-adapter epoch {epoch}")
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
            "continued_from": str(Path(args.checkpoint).resolve()),
            "train_languages": args.train_languages,
            "adapter_dim": args.adapter_dim or checkpoint.get("adapter_dim") or 256,
            "adapter_lr": args.adapter_lr,
            "head_lr": args.head_lr,
        }
        print(json.dumps(epoch_state, ensure_ascii=False), flush=True)

        checkpoint_out = {
            "model_state": model.state_dict(),
            "pretrained_model": checkpoint["pretrained_model"],
            "vocab_size": tokenizer.vocab_size,
            "tokenizer": checkpoint.get("tokenizer", "subword"),
            "tokenizer_vocab_size": checkpoint.get("tokenizer_vocab_size"),
            "tokenizer_character_coverage": checkpoint.get("tokenizer_character_coverage"),
            "adapter_mode": "language",
            "adapter_dim": args.adapter_dim or checkpoint.get("adapter_dim") or 256,
            "adapter_languages": model.adapter_languages,
            "continued_from": str(Path(args.checkpoint).resolve()),
            "train_languages": args.train_languages,
            "epoch": epoch,
            "valid": metrics,
        }
        torch.save(checkpoint_out, last_path)
        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            torch.save(checkpoint_out, best_path)

    output_vocab = output_dir / "vocab.json"
    output_vocab.write_text(Path(args.vocab).read_text(encoding="utf-8"), encoding="utf-8")
    model_path = Path(args.vocab).parent / "tokenizer.model"
    if model_path.exists():
        target_model = output_dir / "tokenizer.model"
        target_model.write_bytes(model_path.read_bytes())

    if args.skip_eval:
        return

    output_csv = output_dir / "pgd5_eval.csv"
    output_jsonl = output_dir / "all_results.jsonl"
    eval_command = [
        sys.executable,
        "scripts/eval_attack_nemo_encoder_char_ctc.py",
        "--manifest",
        args.test_manifest,
        "--checkpoint",
        str(best_path),
        "--vocab",
        str(output_vocab),
        "--output-csv",
        str(output_csv),
        "--output-jsonl",
        str(output_jsonl),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--pgd-steps",
        str(args.pgd_steps),
        "--epsilon",
        str(args.epsilon),
        "--alpha",
        str(args.alpha),
        "--phonemizer-backend",
        args.phonemizer_backend,
        "--languages",
        *args.eval_languages,
    ]
    if args.limit_test is not None:
        eval_command.extend(["--limit", str(args.limit_test)])
    if args.phonemizer_language_map:
        eval_command.extend(["--phonemizer-language-map", args.phonemizer_language_map])
    run(eval_command, PROJECT_ROOT)

    if args.skip_summary:
        return

    summary_command = [
        sys.executable,
        "scripts/summarize_wave_mel_results.py",
        "--wave-jsonl",
        args.wave_jsonl,
        "--mel-csv",
        str(output_csv),
        "--output-json",
        str(output_dir / "wave_mel_summary.json"),
        "--phonemizer-backend",
        args.phonemizer_backend,
        "--languages",
        *args.eval_languages,
    ]
    if args.phonemizer_language_map:
        summary_command.extend(["--phonemizer-language-map", args.phonemizer_language_map])
    run(summary_command, PROJECT_ROOT)


if __name__ == "__main__":
    main()
