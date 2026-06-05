# Compare Noise Robustness: Waveform vs Mel ASR

This repository contains the Mel-Spectrogram ASR side for multilingual noise-attack robustness experiments:

- Model: Conformer encoder + CTC head
- Input to attack: raw waveform
- Model frontend: differentiable log-Mel spectrogram
- Attack: untargeted PGD-5 on waveform
- Evaluation: language-wise WER/CER degradation and robustness gap

## Manifest Format

Use JSONL, one utterance per line:

```json
{"audio": "/abs/path/audio.wav", "text": "reference transcript", "language": "ko"}
{"audio": "/abs/path/audio.wav", "text": "reference transcript", "language": "en"}
```

Recommended language codes for this project:

- `en`: English
- `ko`: Korean
- `zh`: Chinese
- `ru`: Russian

Audio is resampled to 16 kHz by the dataset loader.

## Setup

```bash
bash ../scripts/create_venv.sh cpu
source ../.venv/bin/activate
```

Use `cu121` or `cu128` instead of `cpu` when you need CUDA wheels. The workspace
helper keeps `torch` and `torchaudio` on the same version and installs this
package in editable mode.

## Use ReliableAI Team DataLoader Output

This project can consume the dataset saved by
`ReliableAI_team_project/src/data_load.py` at
`ReliableAI_team_project/data/waveform`. The bridge script preserves the
ReliableAI dataset order within each language and exports 16 kHz wav files plus
JSONL manifests.

```bash
python3 scripts/prepare_team_fleurs_manifest.py \
  --samples-per-lang 1000 \
  --train-per-lang 800 \
  --valid-per-lang 100 \
  --test-per-lang 100
```

The script writes:

- `data/team_fleurs/all.jsonl`
- `data/team_fleurs/train.jsonl`
- `data/team_fleurs/valid.jsonl`
- `data/team_fleurs/test.jsonl`
- `data/team_fleurs/metadata.json`

For direct comparison with `ReliableAI_team_project`, keep the shared variables
fixed and treat the ASR model family as the main independent variable:

| Variable | Controlled value |
| --- | --- |
| Dataset | FLEURS rows from `ReliableAI_team_project/data/waveform` |
| Languages | `ko`, `en`, `zh`, `ru` |
| Sample rate | 16,000 Hz |
| Attack target | Waveform amplitude |
| Attack objective | Untargeted CTC loss maximization |
| PGD steps | `5` |
| L-infinity epsilon | `0.005` |
| PGD alpha | `0.001` |
| Random start | `false` |
| Metrics | WER, CER, attacked-clean degradation |

## Train Conformer-CTC

```bash
python3 scripts/train_conformer_ctc.py \
  --train-manifest data/team_fleurs/train.jsonl \
  --valid-manifest data/team_fleurs/valid.jsonl \
  --output-dir runs/conformer_ctc \
  --epochs 30
```

The script writes:

- `runs/conformer_ctc/checkpoint.pt`
- `runs/conformer_ctc/vocab.json`

## Evaluate Clean vs PGD-5

```bash
python3 scripts/eval_attack_conformer_ctc.py \
  --manifest data/team_fleurs/test.jsonl \
  --checkpoint runs/conformer_ctc/checkpoint.pt \
  --vocab runs/conformer_ctc/vocab.json \
  --output-csv runs/conformer_ctc/pgd5_eval.csv
```

The default attack parameters are intentionally aligned with
`ReliableAI_team_project`: `--pgd-steps 5`, `--epsilon 0.005`,
`--alpha 0.001`, and no random start. Pass `--random-start` only for an
additional ablation, not for the controlled Wav-vs-Mel comparison.

The script prints language-level metrics:

- `clean_wer`, `attacked_wer`, `wer_degradation`
- `clean_cer`, `attacked_cer`, `cer_degradation`
- `wer_robustness_gap = max(wer_degradation) - min(wer_degradation)`
- `cer_robustness_gap = max(cer_degradation) - min(cer_degradation)`

## Evaluate Pretrained NeMo Conformer-CTC

Install the optional NeMo dependency set after activating the shared environment:

```bash
python3 -m pip install -r requirements-nemo.txt
```

Then run the pretrained Conformer-CTC evaluation with the same dataset, PGD, and
metric variables:

```bash
python3 scripts/eval_attack_nemo_conformer_ctc.py \
  --manifest data/team_fleurs/test.jsonl \
  --pretrained-model nvidia/stt_en_conformer_ctc_small \
  --output-csv runs/nemo_conformer_ctc/pgd5_eval.csv
```

Use a language-specific NeMo checkpoint when evaluating non-English subsets, for
example `nvidia/stt_ru_conformer_ctc_large` for Russian.

## Fine-Tune Multilingual NeMo Encoder + Char CTC

For a controlled `ko/en/zh/ru` comparison, use one pretrained multilingual
FastConformer encoder and train a shared character-level CTC head on the project
manifests. This avoids comparing different language-specific decoders.

```bash
python3 scripts/finetune_nemo_encoder_char_ctc.py \
  --train-manifest data/team_fleurs/train.jsonl \
  --valid-manifest data/team_fleurs/valid.jsonl \
  --output-dir runs/nemo_encoder_char_ctc \
  --pretrained-model stt_multilingual_fastconformer_hybrid_large_pc \
  --epochs 10 \
  --batch-size 2
```

The script writes:

- `runs/nemo_encoder_char_ctc/checkpoint.pt`
- `runs/nemo_encoder_char_ctc/last.pt`
- `runs/nemo_encoder_char_ctc/vocab.json`

Evaluate the fine-tuned CTC model with the same PGD settings:

```bash
python3 scripts/eval_attack_nemo_encoder_char_ctc.py \
  --manifest data/team_fleurs/test.jsonl \
  --checkpoint runs/nemo_encoder_char_ctc/checkpoint.pt \
  --vocab runs/nemo_encoder_char_ctc/vocab.json \
  --output-csv runs/nemo_encoder_char_ctc/pgd5_eval.csv
```

## Compare Wav and Mel Results

After running the Mel evaluation, summarize both repositories' outputs with the
same WER/CER implementation:

```bash
python3 scripts/summarize_wave_mel_results.py \
  --wave-jsonl ../ReliableAI_team_project/data/attack_results/all_results.jsonl \
  --mel-csv runs/conformer_ctc/pgd5_eval.csv \
  --output-json runs/conformer_ctc/wave_mel_summary.json
```

## Notes

PGD is implemented as an untargeted attack that maximizes CTC loss against the ground-truth transcript. The perturbation is constrained by an L-infinity bound in waveform amplitude space and clamped to the valid waveform range.
