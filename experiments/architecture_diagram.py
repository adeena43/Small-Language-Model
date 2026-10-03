"""Model architecture diagram (drawn with matplotlib). Run: python experiments/architecture_diagram.py"""
from _common import FIG_DIR, mpl

plt = mpl()
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(6.5, 9.5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 20)


def box(x, y, w, h, text, color="#dbeafe", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#1e3a8a", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(y0, y1, x=5):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="->", lw=1.4))


box(2.5, 0.3, 5, 0.8, "input token ids  (B, T)", "#fef3c7")
arrow(1.1, 1.9)
box(2.5, 1.9, 5, 0.9, "token embedding x sqrt(d)\n(B, T, d_model)")
arrow(2.8, 3.5)
box(2.5, 3.5, 5, 0.9, "+ sinusoidal positional encoding", "#dcfce7")
arrow(4.4, 5.2)
ax.add_patch(FancyBboxPatch((1.2, 5.2), 7.6, 9.0, boxstyle="round,pad=0.05", fc="#f8fafc", ec="#64748b", lw=1.5, ls="--"))
ax.text(8.7, 14.0, "x N", ha="right", fontsize=12, weight="bold")
box(2.5, 5.5, 5, 0.7, "LayerNorm 1", "#ede9fe")
box(2.5, 6.6, 5, 2.2, "Masked (causal) Multi-Head\nSelf-Attention\nQ,K,V -> split heads -> softmax(QK^T/sqrt(d_h))V\n-> merge -> W_o", "#fee2e2", 9)
box(2.5, 9.2, 5, 0.6, "+  residual add", "#fde68a")
box(2.5, 10.2, 5, 0.7, "LayerNorm 2", "#ede9fe")
box(2.5, 11.3, 5, 1.3, "Feed-forward\nLinear(d, 4d) -> GELU -> Linear(4d, d)", "#fee2e2", 9)
box(2.5, 13.0, 5, 0.6, "+  residual add", "#fde68a")
arrow(14.2, 14.9)
box(2.5, 14.9, 5, 0.8, "final LayerNorm", "#ede9fe")
arrow(15.7, 16.4)
box(2.5, 16.4, 5, 0.9, "LM head: Linear(d_model, vocab)\nlogits (B, T, V)")
arrow(17.3, 18.0)
box(2.5, 18.0, 5, 0.9, "softmax -> next-token probabilities\nloss = cross-entropy(logits[t], x[t+1])", "#fef3c7", 9)
ax.set_title("MiniGPT architecture (decoder-only, pre-LN)", fontsize=12)
fig.tight_layout(); fig.savefig(FIG_DIR / "architecture_diagram.png", dpi=150); plt.close(fig)
print("saved", FIG_DIR / "architecture_diagram.png")
