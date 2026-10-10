"""FastAPI inference service.

Web page    :  GET /  (src/static/index.html)   |   API docs: /docs
Run locally :  uvicorn src.api:app --reload
Render start:  uvicorn src.api:app --host 0.0.0.0 --port $PORT

Environment variables
    CHECKPOINT_PATH  path to the checkpoint (default checkpoints/best.pt)
    CHECKPOINT_URL   optional: if the checkpoint file is missing it is downloaded from this URL at startup
                     (e.g. a GitHub release asset or a direct Google-Drive download link)
"""
from __future__ import annotations

import os
import threading
import urllib.request
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.generate import generate_text
from src.utils import load_checkpoint, resolve

STATE: dict = {}
MAX_NEW_TOKENS_CAP = 500
_lock = threading.Lock()          # one generation at a time keeps memory/CPU use predictable on small hosts


def _load():
    path = resolve(os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt"))
    url = os.environ.get("CHECKPOINT_URL")
    if not path.exists() and url:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading checkpoint from {url} ...")
        urllib.request.urlretrieve(url, path)
    if not path.exists():
        STATE["error"] = (f"Checkpoint not found at {path}. Train a model (python -m src.train) or set CHECKPOINT_URL.")
        return
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    model, tok, ckpt = load_checkpoint(path, "cpu")
    STATE.update(model=model, tok=tok, ckpt=ckpt, error=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="MiniGPT API", description="Character-level GPT trained from scratch on Tiny Shakespeare", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", min_length=1, max_length=1000)
    max_new_tokens: int = Field(100, ge=1, le=MAX_NEW_TOKENS_CAP)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=1000)
    seed: int | None = None


def _require_model():
    if STATE.get("model") is None:
        raise HTTPException(status_code=503, detail=STATE.get("error", "model not loaded"))
    return STATE["model"], STATE["tok"]


@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("model") is not None else "model_not_loaded", "detail": STATE.get("error")}


@app.get("/", include_in_schema=False)
def home():
    """The web page (src/static/index.html) that calls POST /generate."""
    return FileResponse(resolve("src/static/index.html"))


@app.get("/info")
def info():
    model, _ = _require_model()
    return {"name": "MiniGPT", "parameters": model.count_parameters(), "context_length": model.context_length,
            "vocab_size": model.vocab_size, "n_layers": model.n_layers, "n_heads": model.n_heads, "d_model": model.d_model,
            "usage": "POST /generate  {prompt, max_new_tokens, temperature, top_k}   |   docs at /docs"}


@app.post("/generate")
def generate(req: GenerateRequest):
    model, tok = _require_model()
    unknown = tok.unknown_chars(req.prompt)
    with _lock:
        if req.seed is not None:
            torch.manual_seed(req.seed)
        text = generate_text(model, tok, req.prompt, req.max_new_tokens, req.temperature, top_k=req.top_k)
    return {"prompt": req.prompt, "generated": text, "completion": text[len(req.prompt):],
            "model_parameters": model.count_parameters(), "context_length": model.context_length,
            "temperature": req.temperature, "max_new_tokens": req.max_new_tokens,
            "note": (f"characters not in the vocabulary were replaced by a space: {unknown}" if unknown else None)}
