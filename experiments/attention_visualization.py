"""Figure 7: attention maps from a TRAINED model (every layer x head).

Run:  python experiments/attention_visualization.py --checkpoint checkpoints/best.pt --prompt "ROMEO:\nO, that I were a glove"
"""
from _common import FIG_DIR

import argparse

import torch

from src.utils import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/best.pt")
ap.add_argument("--prompt", default="ROMEO:\\nO, that I were a glove upon")
ap.add_argument("--out", default=None)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

prompt = args.prompt.encode().decode("unicode_escape")      # allow \n typed on the command line
model, tok, ckpt = load_checkpoint(args.checkpoint, args.device)
ids = torch.tensor([tok.encode(prompt)[: model.context_length]], device=args.device)
maps = model.get_attention_maps(ids)                         # list[n_layers] of (1, H, T, T)
labels = [{" ": "·", "\n": "¶"}.get(c, c) for c in tok.decode(ids[0].tolist())]
T = len(labels)

plt = mpl = None
from _common import mpl as _mpl
plt = _mpl()
L, H = len(maps), maps[0].shape[1]
fig, axes = plt.subplots(L, H, figsize=(3.4 * H, 3.4 * L), squeeze=False)
for l in range(L):
    for h in range(H):
        ax = axes[l][h]
        ax.imshow(maps[l][0, h].cpu(), cmap="viridis", vmin=0)
        ax.set_title(f"layer {l} head {h}", fontsize=9)
        if T <= 40:
            ax.set_xticks(range(T)); ax.set_xticklabels(labels, fontsize=5); ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=5)
        ax.tick_params(length=0)
fig.suptitle(f"Figure 7: attention weights of the trained MiniGPT (query rows, key columns) - prompt of {T} chars", y=1.0)
fig.tight_layout()
out = args.out or str(FIG_DIR / "fig7_attention_visualization.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("saved", out)

# also report numerically how local the heads are (mean attention distance) - handy for the written interpretation
import numpy as np
for l in range(L):
    dists = []
    for h in range(H):
        w = maps[l][0, h].cpu().numpy()
        d = np.arange(T)[:, None] - np.arange(T)[None, :]
        dists.append(float((w * d).sum() / T))
    print(f"layer {l}: mean attention distance per head = {[round(x, 2) for x in dists]}")
