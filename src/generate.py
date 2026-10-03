"""Text generation CLI.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
    python -m src.generate --fixed-prompts            # runs the 5 fixed prompts from the config
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.utils import get_device, load_checkpoint, load_config, resolve


def generate_text(model, tok, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k=None, top_p=None, greedy: bool = False, device="cpu") -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained MiniGPT")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fixed-prompts", action="store_true", help="run the fixed prompts from the config")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    model, tok, ckpt = load_checkpoint(args.checkpoint, device)
    n = model.count_parameters()
    prompts = load_config()["generate"]["fixed_prompts"] if args.fixed_prompts else [args.prompt]
    blocks = []
    for p in prompts:
        text = generate_text(model, tok, p, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.greedy, device)
        block = (f"Prompt: {p!r}\nGenerated: {text}\n"
                 f"model parameters: {n / 1e6:.2f}M\ncontext length: {model.context_length}\n"
                 f"temperature: {'greedy' if args.greedy else args.temperature}\n" + "-" * 60)
        print(block)
        blocks.append(block)
    if args.fixed_prompts:
        out = resolve("outputs/samples/fixed_prompt_samples.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(blocks), encoding="utf-8")
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
