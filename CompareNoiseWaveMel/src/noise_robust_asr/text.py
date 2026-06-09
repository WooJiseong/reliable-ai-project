import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import torch


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = "|"


def normalize_text(text: str) -> str:
    """Light normalization that keeps multilingual characters intact."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class CharTokenizer:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.blank_id = token_to_id[BLANK_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]

    @classmethod
    def build(cls, texts: List[str]) -> "CharTokenizer":
        chars = set()
        for text in texts:
            chars.update(normalize_text(text).replace(" ", SPACE_TOKEN))

        tokens = [BLANK_TOKEN, UNK_TOKEN] + sorted(chars)
        if SPACE_TOKEN not in tokens:
            tokens.append(SPACE_TOKEN)
        return cls({token: idx for idx, token in enumerate(tokens)})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2, sort_keys=True)

    def encode(self, text: str) -> "torch.Tensor":
        normalized = normalize_text(text).replace(" ", SPACE_TOKEN)
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in normalized]
        return torch.tensor(ids, dtype=torch.long)

    def decode_ids(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            token = self.id_to_token.get(idx, UNK_TOKEN)
            if token in {BLANK_TOKEN, UNK_TOKEN}:
                continue
            tokens.append(" " if token == SPACE_TOKEN else token)
        return "".join(tokens).strip()

    def ctc_decode(self, logits: "torch.Tensor") -> List[str]:
        predictions = torch.argmax(logits, dim=-1)
        results = []
        for sequence in predictions.cpu().tolist():
            collapsed = []
            prev = None
            for idx in sequence:
                if idx != prev and idx != self.blank_id:
                    collapsed.append(idx)
                prev = idx
            results.append(self.decode_ids(collapsed))
        return results

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


class SentencePieceCTCTokenizer:
    def __init__(self, model_path: Union[str, Path]):
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.blank_id = 0
        self.unk_id = self.processor.unk_id() + 1

    @classmethod
    def build(
        cls,
        texts: List[str],
        output_dir: Union[str, Path],
        vocab_size: int = 1024,
        model_type: str = "unigram",
        character_coverage: float = 0.995,
    ) -> "SentencePieceCTCTokenizer":
        import sentencepiece as spm

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = output_dir / "tokenizer"
        vocab_size, character_coverage = cls._resolve_training_args(
            texts=texts,
            vocab_size=vocab_size,
            requested_coverage=character_coverage,
        )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            train_path = Path(f.name)
            for text in texts:
                normalized = normalize_text(text)
                if normalized:
                    f.write(normalized + "\n")

        try:
            spm.SentencePieceTrainer.train(
                input=str(train_path),
                model_prefix=str(model_prefix),
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                hard_vocab_limit=False,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )
        finally:
            train_path.unlink(missing_ok=True)

        return cls(model_prefix.with_suffix(".model"))

    @staticmethod
    def _resolve_training_args(
        texts: List[str],
        vocab_size: int,
        requested_coverage: float,
    ) -> tuple[int, float]:
        char_counts = Counter()
        for text in texts:
            char_counts.update(normalize_text(text))

        if not char_counts:
            return vocab_size, requested_coverage

        total = sum(char_counts.values())
        running = 0
        required_chars = 0
        for _, count in char_counts.most_common():
            running += count
            required_chars += 1
            if running / total >= requested_coverage:
                break

        # SentencePiece reserves one meta piece for <unk> with our settings.
        required_vocab_size = required_chars + 1
        if required_vocab_size <= vocab_size:
            return vocab_size, requested_coverage

        print(
            "Requested SentencePiece character_coverage="
            f"{requested_coverage} requires at least {required_vocab_size} vocab entries, "
            f"but tokenizer_vocab_size={vocab_size}. "
            f"Using tokenizer_vocab_size={required_vocab_size} instead. "
            "Lower --tokenizer-character-coverage to keep a smaller vocabulary.",
            flush=True,
        )
        return required_vocab_size, requested_coverage

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SentencePieceCTCTokenizer":
        config_path = Path(path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_path = Path(config["model"])
        if not model_path.is_absolute():
            model_path = config_path.parent / model_path
        return cls(model_path)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        try:
            model = self.model_path.relative_to(path.parent)
        except ValueError:
            model = self.model_path
        config = {
            "tokenizer_type": "sentencepiece_ctc",
            "model": str(model),
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)

    def encode(self, text: str) -> "torch.Tensor":
        piece_ids = self.processor.encode(normalize_text(text), out_type=int)
        return torch.tensor([idx + 1 for idx in piece_ids], dtype=torch.long)

    def decode_ids(self, ids: List[int]) -> str:
        piece_ids = [idx - 1 for idx in ids if idx != self.blank_id]
        piece_ids = [idx for idx in piece_ids if idx >= 0]
        if not piece_ids:
            return ""
        return self.processor.decode(piece_ids).strip()

    def ctc_decode(self, logits: "torch.Tensor") -> List[str]:
        predictions = torch.argmax(logits, dim=-1)
        results = []
        for sequence in predictions.cpu().tolist():
            collapsed = []
            prev = None
            for idx in sequence:
                if idx != prev and idx != self.blank_id:
                    collapsed.append(idx)
                prev = idx
            results.append(self.decode_ids(collapsed))
        return results

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size() + 1


def build_tokenizer(
    texts: List[str],
    output_dir: Union[str, Path],
    tokenizer_type: str = "char",
    vocab_size: int = 1024,
    character_coverage: float = 0.995,
):
    if tokenizer_type == "char":
        return CharTokenizer.build(texts)
    if tokenizer_type == "subword":
        return SentencePieceCTCTokenizer.build(
            texts,
            output_dir=output_dir,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
        )
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def load_tokenizer(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("tokenizer_type") == "sentencepiece_ctc":
        return SentencePieceCTCTokenizer.load(path)
    return CharTokenizer(config)
