#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-data/team_fleurs/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/attack_results/epsilon_sweep_test}"
LIMIT_PER_LANGUAGE="${LIMIT_PER_LANGUAGE:-100}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_TAG="${MODEL_TAG:-espnet/owsm_ctc_v4_1B}"
DTYPE="${DTYPE:-float32}"
ATTACK_SPACE="${ATTACK_SPACE:-mel}"
FIXED_AUDIO_SECONDS="${FIXED_AUDIO_SECONDS:-30.0}"
MEL_SAVE_DTYPE="${MEL_SAVE_DTYPE:-float16}"
FLUSH_EVERY="${FLUSH_EVERY:-25}"
PHONEMIZER_BACKEND="${PHONEMIZER_BACKEND:-espeak}"
LANGUAGES="${LANGUAGES:-ko en zh ru}"
PGD_STEPS=5

if [[ $# -gt 0 ]]; then
  EPSILON_VALUES=("$@")
elif [[ -n "${EPSILONS:-}" ]]; then
  # Example: EPSILONS="0.001 0.005 0.01 0.05 0.1"
  read -r -a EPSILON_VALUES <<< "$EPSILONS"
else
  EPSILON_VALUES=(0.001 0.005 0.01 0.05 0.1)
fi

read -r -a LANGUAGE_VALUES <<< "$LANGUAGES"
mkdir -p "$OUTPUT_DIR"

COMBINED_SUMMARY_JSONL="$OUTPUT_DIR/epsilon_per_summary.jsonl"
: > "$COMBINED_SUMMARY_JSONL"

for EPSILON in "${EPSILON_VALUES[@]}"; do
  ALPHA="$("$PYTHON_BIN" - "$EPSILON" <<'PY'
import sys

epsilon = float(sys.argv[1])
print(f"{epsilon / 5.0:.12g}")
PY
)"

  RESULT_JSONL="$OUTPUT_DIR/result_epsilon_${EPSILON}.jsonl"
  RESULT_CSV="$OUTPUT_DIR/result_epsilon_${EPSILON}.csv"
  SUMMARY_JSON="$OUTPUT_DIR/summary_epsilon_${EPSILON}.json"
  ARTIFACT_DIR="$OUTPUT_DIR/artifacts/epsilon_${EPSILON}"

  echo "==> epsilon=${EPSILON}, alpha=${ALPHA}, pgd_steps=${PGD_STEPS}, limit_per_language=${LIMIT_PER_LANGUAGE}"

  "$PYTHON_BIN" scripts/eval_attack_owsm_ctc.py \
    --manifest "$MANIFEST" \
    --output-csv "$RESULT_CSV" \
    --output-jsonl "$RESULT_JSONL" \
    --model-tag "$MODEL_TAG" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --fixed-audio-seconds "$FIXED_AUDIO_SECONDS" \
    --attack-space "$ATTACK_SPACE" \
    --pgd-steps "$PGD_STEPS" \
    --epsilon "$EPSILON" \
    --alpha "$ALPHA" \
    --artifact-dir "$ARTIFACT_DIR" \
    --mel-save-dtype "$MEL_SAVE_DTYPE" \
    --limit-per-language "$LIMIT_PER_LANGUAGE" \
    --languages "${LANGUAGE_VALUES[@]}" \
    --phonemizer-backend "$PHONEMIZER_BACKEND" \
    --flush-every "$FLUSH_EVERY"

  "$PYTHON_BIN" - "$RESULT_JSONL" "$SUMMARY_JSON" "$COMBINED_SUMMARY_JSONL" "$EPSILON" "$ALPHA" "$PGD_STEPS" "$LIMIT_PER_LANGUAGE" "${LANGUAGE_VALUES[@]}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

jsonl_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
combined_path = Path(sys.argv[3])
epsilon = float(sys.argv[4])
alpha = float(sys.argv[5])
pgd_steps = int(sys.argv[6])
limit_per_language = int(sys.argv[7])
requested_languages = sys.argv[8:]

rows_by_language = defaultdict(list)
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        language = row.get("lang_tag") or row.get("language")
        if language:
            rows_by_language[language].append(row)

missing = [language for language in requested_languages if language not in rows_by_language]
if missing:
    raise SystemExit(f"Missing language results in {jsonl_path}: {missing}")

def mean(values):
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)

by_language = {}
for language in requested_languages:
    rows = rows_by_language[language]
    if len(rows) != limit_per_language:
        raise SystemExit(
            f"{jsonl_path}: language={language} has {len(rows)} rows, "
            f"expected limit_per_language={limit_per_language}"
        )
    if not all("clean_per" in row and "attacked_per" in row for row in rows):
        raise SystemExit(f"{jsonl_path}: PER fields are missing for language={language}")

    clean_per = mean(float(row["clean_per"]) for row in rows)
    attacked_per = mean(float(row["attacked_per"]) for row in rows)
    by_language[language] = {
        "count": len(rows),
        "clean_per": clean_per,
        "attacked_per": attacked_per,
        "per_degradation": attacked_per - clean_per,
        "mean_row_per_degradation": mean(
            float(row.get("per_degradation", float(row["attacked_per"]) - float(row["clean_per"])))
            for row in rows
        ),
    }

all_clean = [item["clean_per"] for item in by_language.values()]
all_attacked = [item["attacked_per"] for item in by_language.values()]
all_degradation = [item["per_degradation"] for item in by_language.values()]
summary = {
    "epsilon": epsilon,
    "alpha": alpha,
    "pgd_steps": pgd_steps,
    "limit_per_language": limit_per_language,
    "languages": requested_languages,
    "result_jsonl": str(jsonl_path),
    "by_language": by_language,
    "aggregate": {
        "mean_clean_per": mean(all_clean),
        "mean_attacked_per": mean(all_attacked),
        "mean_per_degradation": mean(all_degradation),
        "per_robustness_gap": max(all_degradation) - min(all_degradation),
    },
}

summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
with combined_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
PY
done

echo "Done. Results: $OUTPUT_DIR"
echo "Combined PER summary: $COMBINED_SUMMARY_JSONL"
