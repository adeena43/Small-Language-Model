"""Deterministic character-level tokenizer.

Vocabulary policy
-----------------
* The vocabulary is the sorted set of unique characters in the training corpus
  (65 characters for Tiny Shakespeare). Sorting makes it deterministic.
* There are NO special tokens (no <pad>, <bos>, <eos>, <unk>) during training: the corpus is a
  single long stream of characters and we cut fixed-length windows from it.
* At inference time a character that is not in the vocabulary (e.g. a digit typed in a prompt)
  is replaced by a fallback character (a space if available) so that the model never crashes.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str] | str):
        self.chars: list[str] = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.fallback_id = self.stoi.get(" ", 0)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str, strict: bool = False) -> list[int]:
        ids = []
        for c in text:
            if c in self.stoi:
                ids.append(self.stoi[c])
            elif strict:
                raise KeyError(f"Character {c!r} is not in the vocabulary")
            else:
                ids.append(self.fallback_id)
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def unknown_chars(self, text: str) -> list[str]:
        return sorted({c for c in text if c not in self.stoi})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
