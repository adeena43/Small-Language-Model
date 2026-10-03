"""Shape checks for every major component."""
import pytest
import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT
from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding, sinusoidal_table
from src.transformer_block import TransformerBlock


def test_multihead_attention_shapes():
    mha = MultiHeadAttention(64, 8)
    x = torch.randn(3, 10, 64)
    assert mha(x).shape == (3, 10, 64)
    q = mha.split_heads(mha.q_proj(x))
    assert q.shape == (3, 8, 10, 8)


def test_d_model_must_be_divisible_by_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 4)


def test_positional_encoding_shape_and_values():
    pe = sinusoidal_table(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))  # sin(0)=0, cos(0)=1
    assert torch.all(pe.abs() <= 1.0 + 1e-6)
    x = torch.zeros(2, 10, 16)
    assert SinusoidalPositionalEncoding(16, 50)(x).shape == (2, 10, 16)
    assert LearnedPositionalEmbedding(16, 50)(x).shape == (2, 10, 16)


def test_sinusoidal_has_no_parameters():
    assert sum(p.numel() for p in SinusoidalPositionalEncoding(16, 50).parameters()) == 0


def test_block_preserves_shape():
    blk = TransformerBlock(32, 4, 128, 0.1)
    assert blk(torch.randn(2, 9, 32)).shape == (2, 9, 32)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tying", [False, True])
def test_gpt_logits_and_loss(pos, tying):
    m = GPT(vocab_size=40, context_length=32, d_model=32, n_heads=4, n_layers=2, pos_encoding=pos, weight_tying=tying)
    x = torch.randint(0, 40, (3, 20))
    logits, loss = m(x, x)
    assert logits.shape == (3, 20, 40) and loss.ndim == 0


def test_initial_loss_is_close_to_uniform():
    import math
    m = GPT(vocab_size=65, context_length=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    x = torch.randint(0, 65, (8, 32))
    assert abs(m(x, x)[1].item() - math.log(65)) < 0.3


def test_generate_length_and_context_cropping():
    m = GPT(vocab_size=20, context_length=8, d_model=16, n_heads=2, n_layers=1)
    out = m.generate(torch.zeros(1, 5, dtype=torch.long), max_new_tokens=20)   # longer than the context window
    assert out.shape == (1, 25)


def test_weight_tying_reduces_parameters():
    a = GPT(vocab_size=40, context_length=16, d_model=32, n_heads=2, n_layers=1, weight_tying=False)
    b = GPT(vocab_size=40, context_length=16, d_model=32, n_heads=2, n_layers=1, weight_tying=True)
    assert a.count_parameters() - b.count_parameters() == 40 * 32
