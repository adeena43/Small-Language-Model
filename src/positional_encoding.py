"""Positional information: sinusoidal (required) and learned (optional ablation)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)),  PE[pos, 2i+1] = cos(pos / 10000^(2i/d)).  Shape (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)                    # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) encoding, stored as a buffer so it moves with .to(device) but has no parameters."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LearnedPositionalEmbedding(nn.Module):
    """One trainable vector per position (optional experiment)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.emb(pos).unsqueeze(0)
