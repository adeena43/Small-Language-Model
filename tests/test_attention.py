"""Attention tests, including the hand-calculation example (3 tokens, d_k = d_v = 2)."""
import math

import numpy as np
import torch

from src.attention import MultiHeadAttention, SelfAttention, scaled_dot_product_attention

Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])


def numpy_attention(q, k, v):
    s = q @ k.T / math.sqrt(q.shape[-1])
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    w = e / e.sum(axis=-1, keepdims=True)
    return w @ v, w


def test_hand_example_matches_independent_numpy():
    out, w = scaled_dot_product_attention(Q, K, V)
    ref_out, ref_w = numpy_attention(Q.numpy(), K.numpy(), V.numpy())
    assert np.allclose(w.numpy(), ref_w, atol=1e-6)
    assert np.allclose(out.numpy(), ref_out, atol=1e-6)


def test_qk_scores_are_what_we_computed_by_hand():
    scores = Q @ K.T
    assert torch.equal(scores, torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.]]))


def test_weights_sum_to_one_and_shapes():
    q = torch.randn(2, 4, 5, 8)
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 4, 5, 8) and w.shape == (2, 4, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_scaling_matters():
    """Without the 1/sqrt(d_k) factor the softmax would be sharper for large d_k."""
    q = torch.randn(1, 64) * 3
    k = torch.randn(6, 64) * 3
    _, w_scaled = scaled_dot_product_attention(q, k, k)
    w_unscaled = torch.softmax(q @ k.T, dim=-1)
    assert w_unscaled.max() >= w_scaled.max()


def test_self_attention_module_shapes():
    sa = SelfAttention(16, 8, 8)
    out, w = sa(torch.randn(3, 5, 16))
    assert out.shape == (3, 5, 8) and w.shape == (3, 5, 5)


def test_multihead_split_merge_roundtrip():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 7, 32)
    assert torch.equal(mha.merge_heads(mha.split_heads(x)), x)   # split then merge returns the original tensor
    assert mha.split_heads(x).shape == (2, 4, 7, 8)
