"""Debugging gate: loss must collapse on a tiny repeated batch (proves gradients + target shifting + masking work)."""
import torch

from src.dataset import get_batch
from src.train import run_overfit
from src.utils import load_config


def test_targets_are_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=5, context_length=10)
    assert torch.equal(y, x + 1)            # for an arange stream the next token is exactly x + 1
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_tiny_overfit_loss_drops_substantially():
    cfg = load_config()
    cfg["model"].update(d_model=64, n_heads=4, n_layers=2, ffn_dim=256, context_length=32)
    losses = run_overfit(cfg, steps=150, lr=3e-3, batch_size=2, quiet=True)
    assert losses[-1] < 0.1 * losses[0], f"loss only went {losses[0]:.3f} -> {losses[-1]:.3f}"
