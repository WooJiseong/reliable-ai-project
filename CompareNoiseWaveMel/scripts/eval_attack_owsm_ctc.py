#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_robust_asr.attacks.pgd import pgd_attack
from noise_robust_asr.data import TARGET_SAMPLE_RATE, read_manifest
from noise_robust_asr.metrics import PhonemeMetric, cer, summarize_by_language, wer
from noise_robust_asr.models.owsm_ctc import OWSMCTC


class OWSMSpeechManifestDataset(Dataset):
    def __init__(self, manifest_path, model: OWSMCTC, languages=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.items = read_manifest(manifest_path)
        if languages:
            languages = set(languages)
            self.items = [item for item in self.items if item["language"] in languages]
        self.model = model

    def __len__(self):
        return len(self.items)

    def _resolve_audio_path(self, audio_path: str) -> Path:
        path = Path(audio_path)
        if path.is_absolute():
            return path
        return self.manifest_dir / path

    def __getitem__(self, idx):
        import torchaudio

        item = self.items[idx]
        audio_path = self._resolve_audio_path(item["audio"])
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                TARGET_SAMPLE_RATE,
            )
        waveform = waveform.clamp(-1.0, 1.0)
        tokens = self.model.encode_text(item["text"], item["language"])
        return {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "tokens": tokens,
            "token_length": torch.tensor(tokens.numel(), dtype=torch.long),
            "text": item["text"],
            "language": item["language"],
            "audio": str(audio_path),
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
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional ReliableAI_team_project-compatible JSONL output path.",
    )
    parser.add_argument("--model-tag", default="espnet/owsm_ctc_v4_1B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--fixed-audio-seconds", type=float, default=30.0)
    parser.add_argument("--attack-space", choices=["mel", "waveform"], default="mel")
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for clean/attacked mel tensors. Defaults to OUTPUT_CSV.parent/artifacts/mel.",
    )
    parser.add_argument(
        "--mel-save-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=None,
        help="Evaluate the first N samples for each requested language.",
    )
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
        help="Skip phonemizer PER columns. Use only for legacy/debug runs.",
    )
    parser.add_argument(
        "--allow-empty-predictions",
        action="store_true",
        help="Do not fail when all clean predictions are empty.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Rewrite partial CSV/JSONL outputs every N rows so long runs are resumable.",
    )
    parser.add_argument("--random-start", action="store_true")
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


def manifest_languages(manifest_path: str) -> list[str]:
    return sorted({item["language"] for item in read_manifest(manifest_path)})


def validate_languages(model: OWSMCTC, languages: list[str]) -> None:
    unsupported = [
        language
        for language in languages
        if language not in model.supported_project_languages([language])
    ]
    if unsupported:
        raise ValueError(
            f"{model.model_tag} cannot run requested project languages: {unsupported}. "
            "Expected OWSM language symbols for this project are en=<eng>, ko=<kor>, "
            "zh=<zho>, ru=<rus>."
        )


def limit_dataset_per_language(dataset, languages, limit_per_language):
    if limit_per_language is None:
        return dataset
    if limit_per_language < 1:
        raise ValueError("--limit-per-language must be positive when provided")

    requested = list(languages)
    counts = {language: 0 for language in requested}
    indices = []
    for index, item in enumerate(dataset.items):
        language = item["language"]
        if language not in counts:
            continue
        if counts[language] >= limit_per_language:
            continue
        indices.append(index)
        counts[language] += 1
        if all(count >= limit_per_language for count in counts.values()):
            break

    missing = {
        language: limit_per_language - count
        for language, count in counts.items()
        if count < limit_per_language
    }
    if missing:
        raise ValueError(f"Not enough samples for --limit-per-language: {missing}")
    return Subset(dataset, indices)


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def decode_batch(model: OWSMCTC, batch):
    log_probs, _ = model(
        batch["waveform"],
        batch["waveform_length"],
        language=batch.get("language"),
    )
    return model.ctc_decode(log_probs)


@torch.no_grad()
def decode_features(model: OWSMCTC, features, feature_lengths, batch):
    log_probs, _ = model.log_probs_from_features(
        features,
        feature_lengths,
        language=batch.get("language"),
    )
    return model.ctc_decode(log_probs)


def ctc_loss_from_log_probs(model, features, feature_lengths, batch):
    log_probs, output_lengths = model.log_probs_from_features(
        features,
        feature_lengths,
        language=batch.get("language"),
    )
    return F.ctc_loss(
        log_probs.transpose(0, 1),
        batch["tokens"],
        output_lengths,
        batch["token_length"],
        blank=model.blank_id,
        zero_infinity=True,
    )


def pgd_attack_mel(
    model: OWSMCTC,
    batch: dict,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool,
):
    model.eval()
    clean_mel, mel_lengths = model.extract_features(
        batch["waveform"],
        batch["waveform_length"],
    )
    clean_mel = clean_mel.detach()

    if random_start:
        delta = torch.empty_like(clean_mel).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(clean_mel)
    attacked = (clean_mel + delta).detach()
    diagnostics = {
        "attack_initial_loss": None,
        "attack_final_loss": None,
        "attack_max_grad_linf": 0.0,
        "attack_max_grad_l2": 0.0,
    }

    for _ in range(steps):
        attacked.requires_grad_(True)
        loss = ctc_loss_from_log_probs(model, attacked, mel_lengths, batch)
        grad = torch.autograd.grad(loss, attacked, only_inputs=True)[0]
        loss_value = float(loss.detach().cpu())
        if diagnostics["attack_initial_loss"] is None:
            diagnostics["attack_initial_loss"] = loss_value
        diagnostics["attack_final_loss"] = loss_value
        diagnostics["attack_max_grad_linf"] = max(
            diagnostics["attack_max_grad_linf"],
            float(grad.detach().abs().max().cpu()),
        )
        diagnostics["attack_max_grad_l2"] = max(
            diagnostics["attack_max_grad_l2"],
            float(grad.detach().flatten(start_dim=1).norm(dim=1).max().cpu()),
        )

        attacked = attacked.detach() + alpha * grad.sign()
        delta = (attacked - clean_mel).clamp(-epsilon, epsilon)
        attacked = (clean_mel + delta).detach()

    delta = attacked - clean_mel
    diagnostics.update(
        {
            "mel_delta_linf": float(delta.detach().abs().max().cpu()),
            "mel_delta_l2": float(delta.detach().flatten(start_dim=1).norm(dim=1).max().cpu()),
            "mel_delta_mean_abs": float(delta.detach().abs().mean().cpu()),
        }
    )
    return clean_mel, mel_lengths, attacked, diagnostics


def save_mel_artifact(path, mel, mel_length, row, kind, save_dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, save_dtype)
    torch.save(
        {
            "mel": mel[: int(mel_length)].detach().cpu().to(dtype=dtype),
            "mel_length": int(mel_length),
            "audio": row["audio"],
            "language": row["language"],
            "reference": row["reference"],
            "kind": kind,
        },
        path,
    )


def output_fieldnames():
    return [
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
        "attack_space",
        "mel_delta_linf",
        "mel_delta_l2",
        "mel_delta_mean_abs",
        "attack_initial_loss",
        "attack_final_loss",
        "attack_max_grad_linf",
        "attack_max_grad_l2",
        "clean_mel_path",
        "adv_mel_path",
        "model_family",
        "pretrained_model",
    ]


def jsonl_item_from_row(row):
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
        "pretrained_model": row["pretrained_model"],
        "attack_space": row["attack_space"],
        "mel_delta_linf": row["mel_delta_linf"],
        "mel_delta_l2": row["mel_delta_l2"],
        "mel_delta_mean_abs": row["mel_delta_mean_abs"],
        "attack_initial_loss": row["attack_initial_loss"],
        "attack_final_loss": row["attack_final_loss"],
        "attack_max_grad_linf": row["attack_max_grad_linf"],
        "attack_max_grad_l2": row["attack_max_grad_l2"],
        "clean_mel_path": row["clean_mel_path"],
        "adv_mel_path": row["adv_mel_path"],
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
    return item


def write_outputs(output_csv: Path, output_jsonl, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    if output_jsonl:
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(jsonl_item_from_row(row), ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    max_step_reach = args.alpha * args.pgd_steps
    if not args.random_start and max_step_reach < args.epsilon:
        print(
            "Warning: --random-start is disabled, so PGD can move at most "
            f"alpha * steps = {max_step_reach:g}, which is smaller than "
            f"epsilon = {args.epsilon:g}. Increase --alpha/--pgd-steps or enable "
            "--random-start to explore the full L-infinity ball.",
            flush=True,
        )
    if args.device == "cpu" and args.dtype != "float32":
        print("CPU execution requires float32 for OWSM-CTC; using --dtype float32.", flush=True)
        args.dtype = "float32"

    phoneme_metric = None
    if not args.disable_phoneme_metric:
        phoneme_metric = PhonemeMetric(
            backend=args.phonemizer_backend,
            language_map=load_language_map(args.phonemizer_language_map),
        )

    model = OWSMCTC(
        model_tag=args.model_tag,
        device=args.device,
        dtype=args.dtype,
        fixed_audio_seconds=args.fixed_audio_seconds,
        use_flash_attn=args.use_flash_attn,
    ).to(args.device)
    model.eval()

    selected_languages = args.languages or manifest_languages(args.manifest)
    validate_languages(model, selected_languages)

    dataset = OWSMSpeechManifestDataset(args.manifest, model, languages=selected_languages)
    if len(dataset) == 0:
        raise ValueError(f"No manifest rows found for requested languages: {selected_languages}")
    dataset = limit_dataset_per_language(
        dataset,
        selected_languages,
        args.limit_per_language,
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

    output_csv = Path(args.output_csv)
    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir is not None
        else output_csv.parent / "artifacts" / "mel"
    )

    rows = []
    for batch in tqdm(loader, desc="owsm-ctc-pgd-eval"):
        batch = move_batch(batch, args.device)
        clean_mel = None
        adv_mel = None
        mel_lengths = None
        attack_diagnostics = {
            "mel_delta_linf": "",
            "mel_delta_l2": "",
            "mel_delta_mean_abs": "",
            "attack_initial_loss": "",
            "attack_final_loss": "",
            "attack_max_grad_linf": "",
            "attack_max_grad_l2": "",
        }

        if args.attack_space == "mel":
            clean_mel, mel_lengths, adv_mel, attack_diagnostics = pgd_attack_mel(
                model,
                batch,
                epsilon=args.epsilon,
                alpha=args.alpha,
                steps=args.pgd_steps,
                random_start=args.random_start,
            )
            clean_hyps = decode_features(model, clean_mel, mel_lengths, batch)
            attacked_hyps = decode_features(model, adv_mel, mel_lengths, batch)
        else:
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

        for idx, (ref, clean_hyp, attacked_hyp) in enumerate(
            zip(batch["text"], clean_hyps, attacked_hyps)
        ):
            language = batch["language"][idx]
            row_index = len(rows)
            clean_mel_path = ""
            adv_mel_path = ""
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
                "attack_space": args.attack_space,
                **attack_diagnostics,
                "model_family": "owsm_ctc",
                "pretrained_model": args.model_tag,
            }
            if clean_mel is not None and adv_mel is not None and mel_lengths is not None:
                stem = f"{row_index:06d}_{language}"
                clean_mel_path = artifact_dir / f"{stem}_clean_mel.pt"
                adv_mel_path = artifact_dir / f"{stem}_adv_mel.pt"
                save_mel_artifact(
                    clean_mel_path,
                    clean_mel[idx],
                    mel_lengths[idx],
                    row,
                    "clean",
                    args.mel_save_dtype,
                )
                save_mel_artifact(
                    adv_mel_path,
                    adv_mel[idx],
                    mel_lengths[idx],
                    row,
                    "adversarial",
                    args.mel_save_dtype,
                )
                row.update(
                    {
                        "clean_mel_path": str(clean_mel_path),
                        "adv_mel_path": str(adv_mel_path),
                    }
                )
            else:
                row.update({"clean_mel_path": clean_mel_path, "adv_mel_path": adv_mel_path})
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
            if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                write_outputs(output_csv, args.output_jsonl, rows)

    empty_clean = sum(not row["clean_prediction"] for row in rows)
    empty_attacked = sum(not row["attacked_prediction"] for row in rows)
    if rows and empty_clean == len(rows) and not args.allow_empty_predictions:
        raise RuntimeError(
            "OWSM-CTC decoded empty clean predictions for every evaluated sample. "
            "This usually indicates an invalid runtime configuration rather than model "
            "performance. Re-run with --dtype float32, avoid --use-flash-attn unless it "
            "is installed and verified, and smoke-test with --limit 1 --pgd-steps 0."
        )
    if empty_clean or empty_attacked:
        print(
            f"Empty predictions: clean={empty_clean}/{len(rows)}, "
            f"attacked={empty_attacked}/{len(rows)}",
            flush=True,
        )

    write_outputs(output_csv, args.output_jsonl, rows)

    summary = summarize_by_language(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
