"""Decoder-style Transformer block (pre-LayerNorm).

    x = x + MHA(LN1(x), causal_mask)      # attention sub-layer with residual connection
    x = x + FFN(LN2(x))                   # feed-forward sub-layer with residual connection
"""
from __future__ import annotations

import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(D -> ffn_dim) -> GELU -> Linear(ffn_dim -> D) -> Dropout."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, mask=None):                # x: (B, T, D) -> (B, T, D)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
