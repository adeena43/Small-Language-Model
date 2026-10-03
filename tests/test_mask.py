"""Causality tests: a token at position t must never use information from positions > t."""
import torch

from src.attention import causal_mask, scaled_dot_product_attention
from src.gpt import GPT


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    assert not m[0, 1] and m[1, 0] and m[2, 2]


def test_masked_attention_gives_zero_weight_to_future():
    q = torch.randn(2, 3, 6, 4)
    _, w = scaled_dot_product_attention(q, q, q, mask=causal_mask(6))
    future = ~causal_mask(6)
    assert torch.all(w[..., future] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 6), atol=1e-5)   # rows still sum to 1


def test_first_token_only_attends_to_itself():
    q = torch.randn(1, 5, 4)
    _, w = scaled_dot_product_attention(q, q, q, mask=causal_mask(5))
    assert torch.allclose(w[0, 0], torch.tensor([1., 0., 0., 0., 0.]))


def test_full_model_never_uses_future_tokens():
    """Change tokens AFTER position t: logits at positions <= t must be identical."""
    torch.manual_seed(0)
    model = GPT(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=3, dropout=0.0).eval()
    x = torch.randint(0, 30, (2, 16))
    t = 7
    x2 = x.clone()
    x2[:, t + 1:] = torch.randint(0, 30, (2, 16 - t - 1))        # corrupt the future
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = model(x2)
    assert torch.allclose(l1[:, : t + 1], l2[:, : t + 1], atol=1e-6)
    assert not torch.allclose(l1[:, t + 1:], l2[:, t + 1:])       # sanity: the future positions DO change


def test_without_mask_the_future_leaks():
    """Control experiment: with bidirectional attention the earlier positions change too."""
    torch.manual_seed(0)
    model = GPT(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0).eval()
    x = torch.randint(0, 30, (1, 16)); x2 = x.clone(); x2[:, 10:] = (x2[:, 10:] + 1) % 30
    with torch.no_grad():
        l1, _ = model(x, causal=False); l2, _ = model(x2, causal=False)
    assert not torch.allclose(l1[:, :5], l2[:, :5])
