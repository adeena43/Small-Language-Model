"""Shared helpers: seeding, config loading/overriding, device selection, checkpoint loading."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tiny_gpt.yaml"


def set_seed(seed: int) -> None:
    """Make runs reproducible (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_value(s: str) -> Any:
    try:
        return yaml.safe_load(s)  # "4" -> 4, "0.1" -> 0.1, "true" -> True, "abc" -> "abc"
    except Exception:
        return s


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply CLI overrides of the form section.key=value, e.g. model.n_layers=2."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a checkpoint written by train.py. Returns (model, tokenizer, ckpt_dict).

    The checkpoint is self-contained: weights + model config + tokenizer vocabulary.
    """
    from src.gpt import GPT
    from src.tokenizer import CharTokenizer

    ckpt = torch.load(resolve(path), map_location=device, weights_only=False)
    model = GPT(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok = CharTokenizer(ckpt["chars"])
    return model, tok, ckpt
