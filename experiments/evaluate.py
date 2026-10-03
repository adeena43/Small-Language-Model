"""Evaluate a trained checkpoint: full validation loss/perplexity + fixed-prompt generations.

Run:  python experiments/evaluate.py --checkpoint checkpoints/best.pt
"""
from _common import LOG_DIR, SAMPLE_DIR

import argparse
import json
import math

import torch

from src.dataset import build_splits, load_text, sequential_batches
from src.generate import generate_text
from src.utils import load_checkpoint, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/best.pt")
ap.add_argument("--device", default="cpu")
ap.add_argument("--max-new-tokens", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.8)
args = ap.parse_args()

model, tok, ckpt = load_checkpoint(args.checkpoint, args.device)
text = load_text(ckpt.get("config", load_config())["data"]["path"])
train, val = build_splits(text, tok, 0.1)

total, count = 0.0, 0
with torch.no_grad():
    for x, y in sequential_batches(val, 64, model.context_length, args.device):
        loss = model(x, y)[1].item()
        total += loss * x.numel(); count += x.numel()
val_loss = total / count
res = {"checkpoint": args.checkpoint, "parameters": model.count_parameters(), "context_length": model.context_length,
       "full_val_loss": val_loss, "full_val_perplexity": math.exp(val_loss), "val_tokens_evaluated": count,
       "bits_per_char": val_loss / math.log(2)}
print(json.dumps(res, indent=2))

torch.manual_seed(1337)
prompts = load_config()["generate"]["fixed_prompts"]
md = [f"# Fixed-prompt samples (temperature {args.temperature}, {model.count_parameters()/1e6:.2f}M params)\n"]
for i, p in enumerate(prompts, 1):
    g = generate_text(model, tok, p, args.max_new_tokens, args.temperature, device=args.device)
    md.append(f"## Sample {i}: prompt {p!r}\n```\n{g}\n```\n")
(SAMPLE_DIR / "evaluation_samples.md").write_text("\n".join(md), encoding="utf-8")
(LOG_DIR / "evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\n".join(md))
