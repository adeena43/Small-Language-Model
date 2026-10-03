"""Lesson 06 vs 07: bidirectional (BERT-style MLM) masking vs causal (GPT) masking.

1. mask matrices (heatmaps)
2. input/target construction for MLM and CLM on a sample sentence
3. leak test: change a FUTURE token and see whether an EARLIER position's output changes
4. short training experiment: the SAME tiny model trained for next-token prediction with and without the causal mask.
   Without the mask the model can read the answer from the right-hand side, so its loss collapses (cheating).

Run:  python experiments/mask_comparison.py [--steps 800]
"""
from _common import FIG_DIR, LOG_DIR, mpl

import argparse
import json
import random

import numpy as np
import torch

from src.attention import causal_mask
from src.dataset import build_splits, get_batch, load_text
from src.gpt import GPT
from src.tokenizer import CharTokenizer
from src.utils import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()
report = {}

# ---- 1. mask matrices -------------------------------------------------------------------------
T = 8
bidir = np.ones((T, T)); causal = causal_mask(T).numpy().astype(float)
plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, m, title in zip(axes, [bidir, causal], ["BERT-style (bidirectional): every token sees all tokens", "GPT-style (causal): token t sees tokens <= t"]):
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1.3); ax.set_title(title, fontsize=9)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, int(v), ha="center", va="center")
fig.suptitle("Attention mask matrices (1 = allowed, 0 = blocked)")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig6b_mask_matrices_mlm_vs_clm.png", dpi=150); plt.close(fig)

# ---- 2. objective construction ----------------------------------------------------------------
sentence = "to be or not to be".split()
random.seed(3)
mask_pos = random.sample(range(len(sentence)), 2)
mlm_in = ["[MASK]" if i in mask_pos else w for i, w in enumerate(sentence)]
mlm_target = {i: sentence[i] for i in mask_pos}
clm_in, clm_tgt = sentence[:-1], sentence[1:]
txt = [
    "MLM (BERT): input  = " + " ".join(mlm_in),
    f"            targets = only at masked positions: {mlm_target}  (uses left AND right context)",
    "CLM (GPT) : input  = " + " ".join(clm_in),
    "            targets = " + " ".join(clm_tgt) + "   (target[t] = input[t+1]; uses LEFT context only)",
]
print("\n".join(txt)); report["objective_example"] = txt

# ---- 3. leak test on an untrained model -------------------------------------------------------
set_seed(args.seed)
m = GPT(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0).eval()
x = torch.randint(0, 30, (1, 16)); x2 = x.clone(); x2[:, 12:] = (x2[:, 12:] + 5) % 30     # change tokens 12..15 only
with torch.no_grad():
    d_causal = (m(x)[0][:, :8] - m(x2)[0][:, :8]).abs().max().item()
    d_bidir = (m(x, causal=False)[0][:, :8] - m(x2, causal=False)[0][:, :8]).abs().max().item()
line = f"Leak test: max change of logits at positions 0..7 after editing tokens 12..15 -> causal: {d_causal:.2e} | bidirectional: {d_bidir:.2e}"
print(line); report["leak_test"] = {"causal": d_causal, "bidirectional": d_bidir}

# ---- 4. training experiment -------------------------------------------------------------------
text = load_text("data/input.txt"); tok = CharTokenizer.from_text(text)
train, val = build_splits(text, tok)
curves, finals = {}, {}
for name, causal_flag in [("causal (GPT)", True), ("bidirectional (BERT-like attention)", False)]:
    set_seed(args.seed)
    model = GPT(vocab_size=tok.vocab_size, context_length=64, d_model=64, n_heads=4, n_layers=2, ffn_dim=256, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for s in range(args.steps):
        xb, yb = get_batch(train, 32, 64)
        _, loss = model(xb, yb, causal=causal_flag)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        vl = np.mean([model(*get_batch(val, 32, 64), causal=causal_flag)[1].item() for _ in range(10)])
    curves[name] = losses; finals[name] = {"final_train_loss": float(np.mean(losses[-10:])), "val_loss": float(vl)}
    print(f"{name:<40s} final train loss {finals[name]['final_train_loss']:.3f} | val loss {vl:.3f}")
report["training_experiment"] = {"steps": args.steps, "results": finals}

fig, ax = plt.subplots(figsize=(7, 4))
for name, c in curves.items():
    ax.plot(c, label=name)
ax.set_xlabel("step"); ax.set_ylabel("next-token loss"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Same next-token task, with vs without the causal mask\n(bidirectional model 'cheats' by reading the answer)")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig6c_mlm_vs_clm_training.png", dpi=150); plt.close(fig)
(LOG_DIR / "mask_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("saved figures and outputs/logs/mask_comparison.json")
