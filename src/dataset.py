"""Data pipeline: download, tokenise, split, and build (input, shifted-target) batches.

For a window of tokens  t[i], t[i+1], ..., t[i+T]  we use
    x = t[i   : i+T]     (inputs)
    y = t[i+1 : i+T+1]   (targets)  -> y[k] is the token that FOLLOWS x[k].
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

from src.tokenizer import CharTokenizer
from src.utils import resolve

# Same Tiny Shakespeare text as the Hugging Face card (karpathy/tiny_shakespeare); this raw copy is easiest to fetch.
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: str | Path = "data/input.txt", download: bool = True) -> str:
    """Read the corpus; download Tiny Shakespeare if the file is missing."""
    p = resolve(path)
    if not p.exists():
        if not download:
            raise FileNotFoundError(f"{p} not found. Download Tiny Shakespeare into data/input.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {p} ...")
        urllib.request.urlretrieve(DATA_URL, p)
    return p.read_text(encoding="utf-8")


def build_splits(text: str, tokenizer: CharTokenizer, val_fraction: float = 0.1):
    """Contiguous split (no shuffling across the boundary): first 90% train, last 10% val."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(len(data) * (1.0 - val_fraction))
    return data[:n_train], data[n_train:]


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device="cpu"):
    """Random windows. Returns x, y of shape (B, T) with y shifted one step to the right."""
    max_start = len(data) - context_length - 1
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


def sequential_batches(data: torch.Tensor, batch_size: int, context_length: int, device="cpu"):
    """Non-overlapping windows covering the whole split (used for full-validation perplexity)."""
    n = (len(data) - 1) // context_length
    xs = torch.stack([data[i * context_length : (i + 1) * context_length] for i in range(n)])
    ys = torch.stack([data[i * context_length + 1 : (i + 1) * context_length + 1] for i in range(n)])
    for s in range(0, n, batch_size):
        yield xs[s : s + batch_size].to(device), ys[s : s + batch_size].to(device)


def dataset_stats(text: str, tokenizer: CharTokenizer, train: torch.Tensor, val: torch.Tensor) -> dict:
    return {
        "num_characters": len(text),
        "num_lines": text.count("\n") + 1,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
    }
