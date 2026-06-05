import json
import re
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
