"""Lesson 01 evidence: sequential vs parallel depth. A recurrent cell must loop over T steps (sequential depth O(T));
self-attention processes all positions in one matrix multiply (sequential depth O(1)) but stores a T x T matrix (memory O(T^2)).

Run:  python experiments/why_transformers.py
"""
from _common import FIG_DIR, LOG_DIR, mpl

import json
import time

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention, causal_mask

torch.manual_seed(0)
D, H, B = 128, 4, 8
lengths = [32, 64, 128, 256, 512]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # run on the GPU when there is one
hw = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
print(f"device: {hw}")
cell = nn.RNNCell(D, D).to(device); mha = MultiHeadAttention(D, H).eval().to(device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()              # GPU calls are asynchronous: wait for them before reading the clock


def timeit(fn, reps=20):
    for _ in range(3):                        # warm-up
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps * 1000.0


rows = []
with torch.no_grad():
    for T in lengths:
        x = torch.randn(B, T, D, device=device)
        def rnn():
            h = torch.zeros(B, D, device=device)
            for t in range(T):               # T strictly sequential steps: h_t depends on h_{t-1}
                h = cell(x[:, t], h)
            return h
        mask = causal_mask(T, device=device)
        t_rnn = timeit(rnn); t_att = timeit(lambda: mha(x, mask))
        rows.append({"T": T, "rnn_ms": t_rnn, "attention_ms": t_att, "rnn_sequential_steps": T,
                     "attention_sequential_steps": 1, "attention_matrix_MB": B * H * T * T * 4 / 2**20})
        print(f"T={T:>4d} | RNN loop {t_rnn:8.2f} ms ({T} sequential steps) | attention {t_att:8.2f} ms (1 step) | T x T maps {rows[-1]['attention_matrix_MB']:.2f} MB")

table = """
| property (per layer)             | RNN            | Self-attention |
|----------------------------------|----------------|----------------|
| sequential operations            | O(T)           | O(1)           |
| compute per layer                | O(T * d^2)     | O(T^2 * d)     |
| memory for pairwise scores       | none           | O(T^2) per head|
| max path length between 2 tokens | O(T)           | O(1)           |
"""
print(table)
(LOG_DIR / "why_transformers.json").write_text(json.dumps({"rows": rows, "note": f"B={B}, d_model={D}, heads={H}, device={hw}, forward only"}, indent=2))
(LOG_DIR / "why_transformers_table.md").write_text(table)

plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(lengths, [r["rnn_ms"] for r in rows], "o-", label="RNN cell loop (sequential)")
axes[0].plot(lengths, [r["attention_ms"] for r in rows], "s-", label="causal self-attention (parallel)")
axes[0].set_xlabel("sequence length T"); axes[0].set_ylabel("forward time (ms)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title(f"Measured forward time ({hw})")
axes[1].plot(lengths, [r["attention_matrix_MB"] for r in rows], "o-", color="tab:red")
axes[1].set_xlabel("sequence length T"); axes[1].set_ylabel("attention-score memory (MB)"); axes[1].grid(alpha=0.3)
axes[1].set_title("The price of parallelism: O(T^2) memory")
fig.tight_layout(); fig.savefig(FIG_DIR / "why_transformers.png", dpi=150); plt.close(fig)
print("saved", FIG_DIR / "why_transformers.png")
