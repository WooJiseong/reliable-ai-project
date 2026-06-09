#!/usr/bin/env python
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ReliableAI_team_project").exists():
            return parent
    return PROJECT_ROOT.parent


WORKSPACE_ROOT = find_workspace_root()
TEAM_ROOT = WORKSPACE_ROOT / "ReliableAI_team_project"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the controlled multilingual PGD experiment with phonemizer PER metrics."
    )
    parser.add_argument("--samples-per-lang", type=int, default=1000)
    parser.add_argument("--train-per-lang", type=int, default=0)
    parser.add_argument("--valid-per-lang", type=int, default=0)
    parser.add_argument("--test-per-lang", type=int, default=1000)
    parser.add_argument("--languages", nargs="+", default=["ko", "en", "zh", "ru"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for mel evaluation. Use auto to select cuda when available, otherwise cpu.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional mel evaluation smoke-test limit.")
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=None,
        help="Optional balanced mel evaluation limit per requested language.",
    )
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--phonemizer-backend", default="espeak")
    parser.add_argument("--phonemizer-language-map", default=None)
    parser.add_argument(
        "--disable-phoneme-metric",
        action="store_true",
        help="Skip phonemizer PER columns for faster smoke tests.",
    )
    parser.add_argument(
        "--wave-baseline",
        choices=["auto", "always", "never"],
        default="auto",
        help="Run ReliableAI_team_project/exe.py when needed.",
    )
    parser.add_argument(
        "--mel-mode",
        choices=[
            "nemo-pretrained-ctc",
            "nemo-encoder-char-ctc",
            "nemo-encoder-checkpoint",
            "owsm-ctc",
            "local-checkpoint",
        ],
        default="nemo-encoder-char-ctc",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stt_multilingual_fastconformer_hybrid_large_pc",
        help="NeMo model name for pretrained/evaluated 600M-class Conformer runs.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument(
        "--finetune-output-dir",
        default=str(PROJECT_ROOT / "runs" / "nemo_encoder_subword_ctc"),
    )
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-batch-size", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-head-lr", type=float, default=5e-4)
    parser.add_argument("--tokenizer", choices=["subword", "char"], default="subword")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-character-coverage", type=float, default=0.995)
    parser.add_argument("--adapter-mode", choices=["none", "language"], default="language")
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument(
        "--owsm-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
    )
    parser.add_argument("--owsm-use-flash-attn", action="store_true")
    parser.add_argument("--owsm-fixed-audio-seconds", type=float, default=30.0)
    parser.add_argument("--owsm-attack-space", choices=["mel", "waveform"], default="mel")
    parser.add_argument("--owsm-artifact-dir", default=None)
    parser.add_argument(
        "--owsm-mel-save-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--owsm-flush-every", type=int, default=25)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs" / "ctc_conformer_600m_phoneme"),
    )
    parser.add_argument(
        "--manifest-dir",
        default=str(PROJECT_ROOT / "data" / "team_fleurs"),
    )
    parser.add_argument(
        "--team-dataset-dir",
        default=str(TEAM_ROOT / "data" / "waveform"),
    )
    parser.add_argument(
        "--wave-jsonl",
        default=str(TEAM_ROOT / "data" / "attack_results" / "all_results.jsonl"),
    )
    parser.add_argument(
        "--mel-jsonl",
        default=str(PROJECT_ROOT / "data" / "attack_results" / "all_results.jsonl"),
    )
    parser.add_argument("--skip-team-data-build", action="store_true")
    parser.add_argument("--skip-manifest-build", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing Python module {module_name!r}. {install_hint}")


def preflight_phonemizer(args) -> None:
    from noise_robust_asr.metrics import PhonemeMetric

    metric = PhonemeMetric(backend=args.phonemizer_backend)
    metric.phonemize_text("hello", "en")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA, but this PyTorch install has no available CUDA support; using CPU.",
            flush=True,
        )
        return "cpu"
    return requested


def run(command, cwd: Path) -> None:
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed with exit code {exc.returncode} in {cwd}: "
            f"{' '.join(str(part) for part in command)}",
            file=sys.stderr,
            flush=True,
        )
        raise


def write_plan(args, paths):
    plan = {
        "objective": "multilingual PGD-5 robustness comparison with phonemizer PER",
        "controls": {
            "languages": args.languages,
            "sample_rate": 16000,
            "pgd_steps": args.pgd_steps,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "random_start": args.random_start,
            "primary_metric": "phoneme_error_rate",
            "phonemizer_backend": args.phonemizer_backend,
            "device": args.device,
            "owsm_dtype": args.owsm_dtype,
            "owsm_fixed_audio_seconds": args.owsm_fixed_audio_seconds,
            "owsm_use_flash_attn": args.owsm_use_flash_attn,
            "owsm_attack_space": args.owsm_attack_space,
            "owsm_mel_save_dtype": args.owsm_mel_save_dtype,
            "owsm_flush_every": args.owsm_flush_every,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "mel_mode": args.mel_mode,
        "pretrained_model": args.pretrained_model,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_team_data_if_needed(args):
    dataset_dir = Path(args.team_dataset_dir)
    if dataset_dir.exists() or args.skip_team_data_build:
        return
    run([sys.executable, "src/data_load.py"], TEAM_ROOT)


def build_manifest_if_needed(args):
    manifest_dir = Path(args.manifest_dir)
    test_manifest = manifest_dir / "test.jsonl"
    if test_manifest.exists() or args.skip_manifest_build:
        return
    command = [
        sys.executable,
        "scripts/prepare_team_fleurs_manifest.py",
        "--dataset-dir",
        str(Path(args.team_dataset_dir)),
        "--output-dir",
        str(manifest_dir),
        "--samples-per-lang",
        str(args.samples_per_lang),
        "--train-per-lang",
        str(args.train_per_lang),
        "--valid-per-lang",
        str(args.valid_per_lang),
        "--test-per-lang",
        str(args.test_per_lang),
        "--languages",
        *args.languages,
    ]
    run(command, PROJECT_ROOT)


def run_wave_baseline_if_needed(args):
    wave_jsonl = Path(args.wave_jsonl)
    should_run = args.wave_baseline == "always" or (
        args.wave_baseline == "auto" and not wave_jsonl.exists()
    )
    if should_run:
        run([sys.executable, "exe.py"], TEAM_ROOT)


def run_mel_eval(args, test_manifest: Path, output_csv: Path):
    common = [
        "--manifest",
        str(test_manifest),
        "--output-csv",
        str(output_csv),
        "--output-jsonl",
        str(Path(args.mel_jsonl)),
        "--batch-size",
        str(args.batch_size),
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
        *args.languages,
    ]
    if args.phonemizer_language_map:
        common.extend(["--phonemizer-language-map", args.phonemizer_language_map])
    if args.disable_phoneme_metric:
        common.append("--disable-phoneme-metric")
    if args.limit is not None:
        common.extend(["--limit", str(args.limit)])
    if args.random_start:
        common.append("--random-start")

    if args.mel_mode == "nemo-pretrained-ctc":
        command = [
            sys.executable,
            "scripts/eval_attack_nemo_conformer_ctc.py",
            *common,
            "--pretrained-model",
            args.pretrained_model,
        ]
    elif args.mel_mode in {"nemo-encoder-char-ctc", "nemo-encoder-checkpoint"}:
        if not args.checkpoint or not args.vocab:
            raise ValueError(
                "--checkpoint and --vocab are required for nemo-encoder-checkpoint."
            )
        command = [
            sys.executable,
            "scripts/eval_attack_nemo_encoder_char_ctc.py",
            *common,
            "--checkpoint",
            args.checkpoint,
            "--vocab",
            args.vocab,
        ]
    elif args.mel_mode == "owsm-ctc":
        command = [
            sys.executable,
            "scripts/eval_attack_owsm_ctc.py",
            *common,
            "--model-tag",
            args.pretrained_model,
            "--dtype",
            args.owsm_dtype,
            "--fixed-audio-seconds",
            str(args.owsm_fixed_audio_seconds),
            "--attack-space",
            args.owsm_attack_space,
            "--mel-save-dtype",
            args.owsm_mel_save_dtype,
            "--flush-every",
            str(args.owsm_flush_every),
        ]
        if args.limit_per_language is not None:
            command.extend(["--limit-per-language", str(args.limit_per_language)])
        if args.owsm_artifact_dir:
            command.extend(["--artifact-dir", args.owsm_artifact_dir])
        if args.owsm_use_flash_attn:
            command.append("--use-flash-attn")
    else:
        if not args.checkpoint or not args.vocab:
            raise ValueError("--checkpoint and --vocab are required for local-checkpoint.")
        command = [
            sys.executable,
            "scripts/eval_attack_conformer_ctc.py",
            *common,
            "--checkpoint",
            args.checkpoint,
            "--vocab",
            args.vocab,
        ]

    run(command, PROJECT_ROOT)


def finetune_nemo_encoder_char_if_needed(args):
    if args.mel_mode != "nemo-encoder-char-ctc":
        return

    output_dir = Path(args.finetune_output_dir)
    checkpoint = output_dir / "checkpoint.pt"
    vocab = output_dir / "vocab.json"
    if checkpoint.exists() and vocab.exists():
        args.checkpoint = str(checkpoint)
        args.vocab = str(vocab)
        return

    command = [
        sys.executable,
        "scripts/finetune_nemo_encoder_char_ctc.py",
        "--train-manifest",
        str(Path(args.manifest_dir) / "train.jsonl"),
        "--valid-manifest",
        str(Path(args.manifest_dir) / "valid.jsonl"),
        "--output-dir",
        str(output_dir),
        "--pretrained-model",
        args.pretrained_model,
        "--epochs",
        str(args.finetune_epochs),
        "--batch-size",
        str(args.finetune_batch_size),
        "--lr",
        str(args.finetune_lr),
        "--head-lr",
        str(args.finetune_head_lr),
        "--tokenizer",
        args.tokenizer,
        "--tokenizer-vocab-size",
        str(args.tokenizer_vocab_size),
        "--tokenizer-character-coverage",
        str(args.tokenizer_character_coverage),
        "--adapter-mode",
        args.adapter_mode,
        "--adapter-dim",
        str(args.adapter_dim),
        "--adapter-lr",
        str(args.adapter_lr),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.limit_train is not None:
        command.extend(["--limit-train", str(args.limit_train)])
    if args.limit_valid is not None:
        command.extend(["--limit-valid", str(args.limit_valid)])
    if args.freeze_encoder:
        command.append("--freeze-encoder")

    run(command, PROJECT_ROOT)
    args.checkpoint = str(checkpoint)
    args.vocab = str(vocab)


def run_summary(args, output_csv: Path, summary_json: Path):
    if args.skip_summary:
        return
    command = [
        sys.executable,
        "scripts/summarize_wave_mel_results.py",
        "--wave-jsonl",
        args.wave_jsonl,
        "--mel-csv",
        str(output_csv),
        "--output-json",
        str(summary_json),
        "--phonemizer-backend",
        args.phonemizer_backend,
        "--languages",
        *args.languages,
    ]
    if args.phonemizer_language_map:
        command.extend(["--phonemizer-language-map", args.phonemizer_language_map])
    if args.disable_phoneme_metric:
        command.append("--disable-phoneme-metric")
    run(command, PROJECT_ROOT)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    require_module(
        "phonemizer",
        "Install it in the workspace venv and ensure eSpeak/eSpeak-NG is available.",
    )
    if not args.disable_phoneme_metric:
        preflight_phonemizer(args)
    if args.mel_mode.startswith("nemo"):
        require_module("nemo", "Install CompareNoiseWaveMel/requirements-nemo.txt.")
    if args.mel_mode == "owsm-ctc":
        require_module("espnet2", "Install CompareNoiseWaveMel/requirements-espnet.txt.")
        require_module(
            "espnet_model_zoo",
            "Install CompareNoiseWaveMel/requirements-espnet.txt.",
        )

    output_dir = Path(args.output_dir)
    output_csv = output_dir / "pgd5_eval.csv"
    summary_json = output_dir / "wave_mel_summary.json"
    test_manifest = Path(args.manifest_dir) / "test.jsonl"
    paths = {
        "team_dataset_dir": Path(args.team_dataset_dir),
        "test_manifest": test_manifest,
        "wave_jsonl": Path(args.wave_jsonl),
        "mel_csv": output_csv,
        "mel_jsonl": Path(args.mel_jsonl),
        "summary_json": summary_json,
    }
    write_plan(args, paths)

    build_team_data_if_needed(args)
    build_manifest_if_needed(args)
    run_wave_baseline_if_needed(args)
    finetune_nemo_encoder_char_if_needed(args)
    run_mel_eval(args, test_manifest, output_csv)
    run_summary(args, output_csv, summary_json)


if __name__ == "__main__":
    main()
