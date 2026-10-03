"""Required hand calculation: scaled dot-product attention on a 3-token sequence (d_k = d_v = 2).

Run:  python experiments/hand_calculation.py
Produces: outputs/logs/hand_calculation.txt  and  outputs/figures/fig1_hand_attention.png, fig2_causal_mask.png
"""
from _common import FIG_DIR, LOG_DIR, mpl  # noqa: F401  (also fixes sys.path)

import numpy as np
import torch

from src.attention import causal_mask, scaled_dot_product_attention

np.set_printoptions(precision=4, suppress=True)
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 2.], [3., 4.], [5., 6.]])
d_k = Q.shape[1]


def softmax_rows(s):
    e = np.exp(s - s.max(axis=1, keepdims=True))      # (subtracting the row max does not change the result)
    return e / e.sum(axis=1, keepdims=True)


lines = []
def out(title, mat=None):
    lines.append(title)
    if mat is not None:
        lines.append(str(mat))
    lines.append("")

out("INPUTS (3 tokens, d_k = d_v = 2)")
out("Q =", Q); out("K =", K); out("V =", V)
S = Q @ K.T
out("Step 1: scores = Q K^T", S)
S_scaled = S / np.sqrt(d_k)
out(f"Step 2: divide by sqrt(d_k) = sqrt({d_k}) = {np.sqrt(d_k):.4f}", S_scaled)
W = softmax_rows(S_scaled)
out("Step 3: row-wise softmax -> attention weights (each row sums to 1)", W)
out("   row sums =", W.sum(axis=1))
O = W @ V
out("Step 4: output = weights @ V", O)

mask = np.tril(np.ones((3, 3), dtype=bool))
S_masked = np.where(mask, S_scaled, -np.inf)
out("Step 5: CAUSAL MASK -> masked score matrix (future positions = -inf)", S_masked)
Wm = softmax_rows(S_masked)
out("Step 6: softmax of masked scores (exp(-inf) = 0 so future weights are exactly 0)", Wm)
Om = Wm @ V
out("Step 7: masked output = masked_weights @ V", Om)

out("WHAT CHANGED (information available to each position):")
lines.append("  token 1 : without mask it mixes info from tokens 1,2,3; with mask it sees ONLY itself (output = V[1]).")
lines.append("  token 2 : without mask it sees 1,2,3; with mask it sees tokens 1 and 2 only.")
lines.append("  token 3 : sees tokens 1,2,3 in both cases (the last token may look at the whole past).")
lines.append("  => the masked output of position t depends only on tokens <= t, which is what lets a GPT")
lines.append("     predict token t+1 from tokens 1..t without 'cheating' by looking at the answer.")

# cross-check against the PyTorch implementation used in the model
o_t, w_t = scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V))
om_t, wm_t = scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V), mask=causal_mask(3))
assert np.allclose(w_t.numpy(), W) and np.allclose(o_t.numpy(), O)
assert np.allclose(wm_t.numpy(), Wm) and np.allclose(om_t.numpy(), Om)
lines.append("")
lines.append("Cross-check: src.attention.scaled_dot_product_attention reproduces every matrix above  [OK]")

text = "\n".join(lines)
print(text)
(LOG_DIR / "hand_calculation.txt").write_text(text, encoding="utf-8")

# ---- figures ---------------------------------------------------------------------------------
plt = mpl()
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (mat, title) in zip(axes, [(S, "Q K^T"), (S_scaled, "scaled: / sqrt(d_k)"), (W, "softmax weights"), (O, "output = W V")]):
    im = ax.imshow(mat, cmap="viridis")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=11)
    ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(3)); ax.set_yticklabels(["tok1", "tok2", "tok3"])
fig.suptitle("Figure 1: hand-calculated scaled dot-product attention (3 tokens, d_k = 2)")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_hand_attention.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
disp = np.where(mask, S_scaled, np.nan)
for ax, (mat, title) in zip(axes, [(mask.astype(float), "causal mask (1 = allowed)"), (disp, "masked scores (blank = -inf)"), (Wm, "weights after masking")]):
    ax.imshow(mat, cmap="Blues", vmin=0)
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, "-inf" if (title.startswith("masked") and np.isnan(v)) else f"{v:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["1", "2", "3"]); ax.set_yticklabels(["1", "2", "3"])
    ax.set_xlabel("key position (attended to)"); ax.set_ylabel("query position")
fig.suptitle("Figure 2: causal mask - future tokens are blocked")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_causal_mask.png", dpi=150); plt.close(fig)
print("\nsaved figures to", FIG_DIR)
