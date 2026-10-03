"""Tokenizer, checkpoint reload, and API tests."""
import os

import torch

from src.generate import generate_text
from src.gpt import GPT
from src.tokenizer import CharTokenizer
from src.utils import load_checkpoint


def _tiny_checkpoint(path, tok):
    model = GPT(vocab_size=tok.vocab_size, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    torch.save({"model_state": model.state_dict(), "model_config": model.config, "chars": tok.chars, "step": 0, "val_loss": 0.0}, path)
    return model


def test_tokenizer_roundtrip_and_determinism():
    text = "Hello, World!\nSecond line."
    a, b = CharTokenizer.from_text(text), CharTokenizer.from_text(text[::-1])
    assert a.chars == b.chars == sorted(set(text))              # deterministic, order-independent vocabulary
    assert a.decode(a.encode(text)) == text


def test_tokenizer_unknown_character_policy():
    tok = CharTokenizer.from_text("ab c")
    assert tok.decode(tok.encode("a9b")) == "a b"               # unseen char -> space fallback
    try:
        tok.encode("9", strict=True); raise AssertionError("strict mode should raise")
    except KeyError:
        pass


def test_checkpoint_reload_generates_identically(tmp_path):
    tok = CharTokenizer.from_text("abcdefghij \n")
    original = _tiny_checkpoint(tmp_path / "c.pt", tok).eval()
    model, tok2, _ = load_checkpoint(tmp_path / "c.pt", "cpu")  # a "fresh process" would do exactly this
    torch.manual_seed(5); a = generate_text(original, tok, "abc", 30, 0.8)
    torch.manual_seed(5); b = generate_text(model, tok2, "abc", 30, 0.8)
    assert a == b and len(a) == 33


def test_api_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api as api
    tok = CharTokenizer.from_text("abcdefghij \n")
    _tiny_checkpoint(tmp_path / "c.pt", tok)
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "c.pt"))
    api.STATE.clear()
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 10, "temperature": 0.8, "seed": 0})
        assert r.status_code == 200 and len(r.json()["generated"]) == 13
        assert client.post("/generate", json={"prompt": "abc", "max_new_tokens": 10_000}).status_code == 422
    api.STATE.clear()
