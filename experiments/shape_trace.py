"""Figure 3 (multi-head tensor-shape trace) and Figure 5 (model summary / parameter count).

Run:  python experiments/shape_trace.py
"""
from _common import LOG_DIR  # noqa: F401

import torch

from src.attention import MultiHeadAttention, causal_mask
from src.gpt import GPT
from src.utils import load_config

torch.manual_seed(0)
lines = []
B, T, D, H = 2, 6, 32, 4
lines.append(f"MULTI-HEAD ATTENTION SHAPE TRACE   (B={B}, T={T}, d_model={D}, n_heads={H}, head_dim={D // H})")
mha = MultiHeadAttention(D, H)
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    y = mha(torch.randn(B, T, D), mask=causal_mask(T), trace=True)
lines.append(buf.getvalue().rstrip())
lines.append(f"W_q, W_k, W_v, W_o each: Linear({D} -> {D});  concatenating {H} heads of size {D // H} gives {D} again.")
lines.append("")
cfg = load_config()["model"]
model = GPT(vocab_size=65, **cfg)
lines.append(model.model_summary())
x = torch.randint(0, 65, (2, cfg["context_length"]))
logits, loss = model(x, x)
lines.append("")
lines.append(f"forward check: idx {tuple(x.shape)} -> logits {tuple(logits.shape)}, initial loss {loss.item():.3f} (uniform guess = ln 65 = 4.174)")
text = "\n".join(lines)
print(text)
(LOG_DIR / "shape_trace_and_summary.txt").write_text(text, encoding="utf-8")
