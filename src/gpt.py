"""MiniGPT: decoder-only Transformer language model (causal LM)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import MultiHeadAttention, causal_mask
from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class GPT(nn.Module):
    """token ids (B, T) -> token embedding (B, T, D) + position -> N blocks -> LayerNorm -> logits (B, T, V)."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",   # "sinusoidal" | "learned"
        weight_tying: bool = False,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.n_heads, self.n_layers, self.ffn_dim = n_heads, n_layers, ffn_dim
        self.dropout_p, self.pos_encoding_name, self.weight_tying = dropout, pos_encoding, weight_tying

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, context_length)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEmbedding(d_model, context_length)
        else:
            raise ValueError(f"unknown pos_encoding {pos_encoding!r}")
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        # (T_max, T_max) lower-triangular causal mask; a buffer so it follows .to(device)
        self.register_buffer("mask", causal_mask(context_length), persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(blk.ffn.fc2.weight, std=0.02 / math.sqrt(2 * n_layers))
        # token embeddings get std = D^-0.5 and are multiplied by sqrt(D) in forward (as in the original paper)
        nn.init.normal_(self.tok_emb.weight, std=d_model ** -0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def config(self) -> dict:
        return dict(
            vocab_size=self.vocab_size, context_length=self.context_length, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers, ffn_dim=self.ffn_dim, dropout=self.dropout_p,
            pos_encoding=self.pos_encoding_name, weight_tying=self.weight_tying,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, causal: bool = True):
        """idx, targets: (B, T) long. Returns logits (B, T, V) and loss (scalar) if targets are given.

        causal=False removes the mask (bidirectional attention) -- used ONLY by the MLM-vs-CLM experiment.
        """
        B, T = idx.shape
        assert T <= self.context_length, f"sequence length {T} > context_length {self.context_length}"
        x = self.tok_emb(idx) * math.sqrt(self.d_model)          # (B, T, D)
        x = self.drop(self.pos_enc(x))                           # add position information
        mask = self.mask[:T, :T] if causal else None             # (T, T) broadcast over (B, H, T, T)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                 # (B, T, V)
        loss = None
        if targets is not None:
            # next-token cross-entropy: flatten (B, T, V) -> (B*T, V) and (B, T) -> (B*T)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = None,
                 top_p: float | None = None, greedy: bool = False):
        """Autoregressive sampling. idx: (B, T0) prompt ids. Returns (B, T0 + max_new_tokens)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.context_length:]                  # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :]                            # only the last position predicts the next token
            if greedy or temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p   # keep the token that crosses p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx

    # ---- inspection helpers -------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        # parameters() de-duplicates tied weights automatically
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def model_summary(self) -> str:
        n = self.count_parameters
        blk = self.blocks[0]
        per_block = sum(p.numel() for p in blk.parameters())
        pos = sum(p.numel() for p in self.pos_enc.parameters())
        lines = [
            "MiniGPT model summary",
            "=" * 54,
            f"vocab_size        : {self.vocab_size}",
            f"context_length    : {self.context_length}",
            f"d_model           : {self.d_model}",
            f"n_heads           : {self.n_heads}  (head_dim = {self.d_model // self.n_heads})",
            f"n_layers          : {self.n_layers}",
            f"ffn_dim           : {self.ffn_dim}",
            f"dropout           : {self.dropout_p}",
            f"position encoding : {self.pos_encoding_name}",
            f"weight tying      : {self.weight_tying}",
            "-" * 54,
            f"token embedding   : {self.tok_emb.weight.numel():>10,}",
            f"position encoding : {pos:>10,}  ({'learned' if pos else 'fixed buffer, no parameters'})",
            f"each block (x{self.n_layers})   : {per_block:>10,}",
            f"  - attention     : {sum(p.numel() for p in blk.attn.parameters()):>10,}",
            f"  - feed-forward  : {sum(p.numel() for p in blk.ffn.parameters()):>10,}",
            f"  - 2 LayerNorms  : {sum(p.numel() for p in list(blk.ln1.parameters()) + list(blk.ln2.parameters())):>10,}",
            f"final LayerNorm   : {sum(p.numel() for p in self.ln_f.parameters()):>10,}",
            f"LM head           : {self.lm_head.weight.numel():>10,}{'  (tied with token embedding)' if self.weight_tying else ''}",
            "-" * 54,
            f"TOTAL parameters  : {n():>10,}  ({n() / 1e6:.2f}M)",
        ]
        return "\n".join(lines)

    @torch.no_grad()
    def get_attention_maps(self, idx: torch.Tensor):
        """Run a forward pass and return a list (one per layer) of attention weights (B, H, T, T)."""
        mhas = [m for m in self.modules() if isinstance(m, MultiHeadAttention)]
        for m in mhas:
            m.keep_weights = True
        was_training = self.training
        self.eval()
        self(idx)
        maps = [m.last_weights for m in mhas]
        for m in mhas:
            m.keep_weights, m.last_weights = False, None
        self.train(was_training)
        return maps
