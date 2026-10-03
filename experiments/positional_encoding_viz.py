"""Figure 4: positional encoding visualisation (heatmap, waves, position-similarity).

Run:  python experiments/positional_encoding_viz.py
"""
from _common import FIG_DIR, mpl

import numpy as np

from src.positional_encoding import sinusoidal_table

pe = sinusoidal_table(128, 128).numpy()
plt = mpl()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
im = axes[0].imshow(pe, aspect="auto", cmap="RdBu"); axes[0].set_title("PE matrix (position x dimension)")
axes[0].set_xlabel("dimension"); axes[0].set_ylabel("position"); fig.colorbar(im, ax=axes[0], fraction=0.046)
for d in (0, 1, 8, 9, 32, 33):
    axes[1].plot(pe[:, d], label=f"dim {d}")
axes[1].set_title("Selected dimensions across positions\n(low dims = fast waves, high dims = slow waves)")
axes[1].set_xlabel("position"); axes[1].legend(fontsize=8, ncol=2)
sim = pe @ pe.T
sim = sim / np.sqrt(np.outer(np.diag(sim), np.diag(sim)))
im2 = axes[2].imshow(sim, cmap="viridis"); axes[2].set_title("cosine similarity between positions\n(nearby positions are more similar)")
axes[2].set_xlabel("position"); axes[2].set_ylabel("position"); fig.colorbar(im2, ax=axes[2], fraction=0.046)
fig.suptitle("Figure 4: sinusoidal positional encoding (d_model = 128)")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_positional_encoding.png", dpi=150); plt.close(fig)
print("saved", FIG_DIR / "fig4_positional_encoding.png")
