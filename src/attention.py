"""Scaled dot-product attention, single-head self-attention, and multi-head attention.

Everything here is written by hand with plain tensor ops (matmul, softmax, Linear).
Mask convention used throughout: a BOOLEAN mask where True = "may attend", False = "blocked".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) lower-triangular mask. mask[i, j] is True iff j <= i (token i may look at token j)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None, dropout: nn.Module | None = None):
    """softmax(Q K^T / sqrt(d_k)) V.

    q, k, v : (..., T, d_k)   (leading dims can be batch and/or heads)
    mask    : bool, broadcastable to (..., T, T); False positions get -inf BEFORE the softmax
    returns : (output (..., T, d_v), weights (..., T, T))
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)       # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))   # blocked -> -inf -> softmax weight exactly 0
    weights = torch.softmax(scores, dim=-1)                 # each row sums to 1
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights                             # (..., T, d_v)


class SelfAttention(nn.Module):
    """Single-head self-attention with learned Q/K/V projections (used for teaching and tests)."""

    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x, mask=None):
        return scaled_dot_product_attention(self.q_proj(x), self.k_proj(x), self.v_proj(x), mask)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with EXPLICIT head split / merge.

    Shapes (B = batch, T = sequence length, D = d_model, H = n_heads, d_h = D / H):
        x                      (B, T, D)
        q,k,v after Linear     (B, T, D)
        split heads            (B, H, T, d_h)     view(B,T,H,d_h) then transpose(1,2)
        attention per head     weights (B, H, T, T), out (B, H, T, d_h)
        merge heads (concat)   (B, T, D)          transpose(1,2) then view(B,T,D)
        output projection W_o  (B, T, D)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.keep_weights = False          # set True to store attention maps for visualisation
        self.last_weights = None

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, T, d_h)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, d_h = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * d_h)         # (B, T, D)

    def forward(self, x, mask=None, trace: bool = False):
        def log(name, tensor):
            if trace:
                print(f"  {name:<34s} {tuple(tensor.shape)}")

        log("input x (B, T, D)", x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        log("q/k/v after Linear (B, T, D)", q)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        log("split heads q (B, H, T, d_h)", q)
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        log("attention weights (B, H, T, T)", weights)
        log("per-head output (B, H, T, d_h)", out)
        out = self.merge_heads(out)
        log("merged / concatenated (B, T, D)", out)
        out = self.resid_dropout(self.out_proj(out))
        log("after output projection (B, T, D)", out)
        if self.keep_weights:
            self.last_weights = weights.detach()
        return out
