"""Training pipeline for MiniGPT.

Usage
-----
    python -m src.train --config configs/tiny_gpt.yaml
    python -m src.train --set model.n_layers=2 train.max_steps=500
    python -m src.train --overfit_tiny          # debugging gate: loss must collapse on one tiny batch
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from src.dataset import build_splits, dataset_stats, get_batch, load_text
from src.gpt import GPT
from src.tokenizer import CharTokenizer
from src.utils import apply_overrides, get_device, load_config, resolve, set_seed


def lr_at(step: int, cfg_t: dict) -> float:
    """Linear warm-up then cosine decay from lr to min_lr."""
    lr, min_lr, warm, total = cfg_t["lr"], cfg_t["min_lr"], cfg_t["warmup_steps"], cfg_t["max_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    if step >= total:
        return min_lr
    progress = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, n_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, cfg["train"]["batch_size"], cfg["model"]["context_length"], device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def make_model(cfg: dict, vocab_size: int) -> GPT:
    return GPT(vocab_size=vocab_size, **cfg["model"])


def save_loss_plot(rows: list[dict], path: Path, title: str = "Training vs validation loss") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([r["step"] for r in rows if r["train_loss"] is not None],
            [r["train_loss"] for r in rows if r["train_loss"] is not None], label="train loss", alpha=0.8)
    v = [r for r in rows if r["val_loss"] is not None]
    ax.plot([r["step"] for r in v], [r["val_loss"] for r in v], label="validation loss", marker="o", ms=3)
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss (nats)"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


RESUME_KEYS = ["lr", "min_lr", "warmup_steps", "weight_decay", "grad_clip", "batch_size", "max_steps"]


def fingerprint(cfg: dict) -> str:
    """Identifies a training setup; a resume file is only reused if the setup is identical."""
    return json.dumps({"model": cfg["model"], "train": {k: cfg["train"][k] for k in RESUME_KEYS}, "seed": cfg["seed"]}, sort_keys=True)


def train(cfg: dict, run_name: str = "main", quiet: bool = False, text: str | None = None,
          resume: bool = False, stop_after_step: int | None = None) -> dict:
    """Full training run. Returns a summary dict (also written to <out_dir>/logs/<run_name>_summary.json).

    resume=True : every `save_interval` steps the complete training state is written to <checkpoint_dir>/<run>_resume.pt;
                  if that file exists (and the config is identical) training continues from there instead of restarting.
    stop_after_step : test hook that simulates an interruption (raises InterruptedError).
    """
    t = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device(t["device"])
    text = text if text is not None else load_text(cfg["data"]["path"])
    tok = CharTokenizer.from_text(text)
    train_data, val_data = build_splits(text, tok, cfg["data"]["val_fraction"])
    stats = dataset_stats(text, tok, train_data, val_data)

    model = make_model(cfg, tok.vocab_size).to(device)
    n_params = model.count_parameters()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"], betas=(0.9, 0.95))

    out_dir = resolve(t["out_dir"]); log_dir = out_dir / "logs"; fig_dir = out_dir / "figures"; sample_dir = out_dir / "samples"
    ckpt_dir = resolve(t["checkpoint_dir"])
    for d in (log_dir, fig_dir, sample_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    main_run = run_name == "main"
    best_path = ckpt_dir / ("best.pt" if main_run else f"{run_name}_best.pt")

    if not quiet:
        print(f"device={device} | dataset={stats}")
        print(model.model_summary())

    T, B = cfg["model"]["context_length"], t["batch_size"]
    rows, best_val, t0 = [], float("inf"), time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokens_seen, last_tick, tick_tokens = 0, time.time(), 0
    sample_file = sample_dir / f"{run_name}_training_samples.txt"
    prompt_ids = torch.tensor([tok.encode(t["sample_prompt"])], device=device)
    tps = 0.0
    train_loss_ema = None
    start_step = 0
    resume_path = ckpt_dir / f"{run_name}_resume.pt"
    save_every = t.get("save_interval") or t["eval_interval"]

    def save_ckpt(path, step, val_loss):
        torch.save({"model_state": model.state_dict(), "model_config": model.config, "chars": tok.chars,
                    "step": step, "val_loss": val_loss, "config": cfg}, path)

    def save_resume(next_step):
        """Atomic write (tmp file then rename) so a timeout in the middle of saving cannot corrupt the resume file."""
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "next_step": next_step,
                    "rows": rows, "best_val": best_val, "tokens_seen": tokens_seen, "elapsed": time.time() - t0,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "fingerprint": fingerprint(cfg)}, tmp)
        os.replace(tmp, resume_path)

    resumed = False
    if resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st["fingerprint"] == fingerprint(cfg):
            model.load_state_dict(st["model_state"]); opt.load_state_dict(st["opt_state"])
            start_step, rows, best_val = st["next_step"], st["rows"], st["best_val"]
            tokens_seen, t0 = st["tokens_seen"], time.time() - st["elapsed"]
            torch.set_rng_state(st["rng_cpu"])
            if st["rng_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["rng_cuda"])
            resumed = True
            print(f"[resume] {run_name}: continuing from step {start_step}/{t['max_steps']}")
        else:
            print(f"[resume] {run_name}: found a resume file for a DIFFERENT config - ignoring it and starting fresh")
    if not resumed:
        sample_file.write_text("", encoding="utf-8")

    model.train()
    for step in range(start_step, t["max_steps"] + 1):
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % t["eval_interval"] == 0 or step == t["max_steps"]:
            tr = estimate_loss(model, train_data, cfg, device, t["eval_batches"])
            va = estimate_loss(model, val_data, cfg, device, t["eval_batches"])
            rows.append({"step": step, "epoch": round(tokens_seen / len(train_data), 3), "train_loss": tr, "val_loss": va,
                         "lr": lr, "tokens_per_sec": tps, "elapsed_sec": time.time() - t0})
            if not quiet:
                print(f"step {step:>5d} | train {tr:.4f} | val {va:.4f} (ppl {math.exp(va):.2f}) | lr {lr:.2e} | {tps:,.0f} tok/s")
            if va < best_val:
                best_val = va
                save_ckpt(best_path, step, va)
        if step % t["sample_interval"] == 0 or step == t["max_steps"]:
            out = model.generate(prompt_ids, t["sample_tokens"], temperature=cfg["generate"]["temperature"])
            with open(sample_file, "a", encoding="utf-8") as f:
                f.write(f"=== step {step} ===\n{tok.decode(out[0].tolist())}\n\n")
        if step == t["max_steps"]:
            break

        x, y = get_batch(train_data, B, T, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if t["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        tokens_seen += B * T; tick_tokens += B * T
        train_loss_ema = loss.item() if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * loss.item()
        if step % t["log_interval"] == 0:
            now = time.time()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tps = tick_tokens / max(1e-9, now - last_tick); last_tick, tick_tokens = now, 0
            rows.append({"step": step, "epoch": round(tokens_seen / len(train_data), 3), "train_loss": loss.item(),
                         "val_loss": None, "lr": lr, "tokens_per_sec": tps, "elapsed_sec": now - t0})
        if resume and (step + 1) % save_every == 0 and step + 1 < t["max_steps"]:
            save_resume(step + 1)
        if stop_after_step is not None and step + 1 >= stop_after_step:
            raise InterruptedError(f"simulated interruption after step {step + 1}")

    elapsed = time.time() - t0
    if resume_path.exists():
        resume_path.unlink()          # run finished: the resume file is no longer needed
    save_ckpt(ckpt_dir / ("last.pt" if main_run else f"{run_name}_last.pt"), t["max_steps"], rows[-1]["val_loss"])
    with open(log_dir / f"{run_name}_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    save_loss_plot(rows, fig_dir / f"{run_name}_loss_curve.png")

    final = [r for r in rows if r["val_loss"] is not None][-1]
    tps_list = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"]]
    summary = {
        "run_name": run_name, "parameters": n_params, "model_config": model.config,
        "train_config": t, "dataset": stats, "device": str(device),
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "wall_clock_sec": round(elapsed, 1), "final_train_loss": final["train_loss"], "final_val_loss": final["val_loss"],
        "best_val_loss": best_val, "final_val_perplexity": math.exp(final["val_loss"]),
        "mean_tokens_per_sec": sum(tps_list) / len(tps_list) if tps_list else 0.0,
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "best_checkpoint": str(best_path),
    }
    (log_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not quiet:
        print(f"\nDone in {elapsed:.0f}s | params {n_params:,} | final val loss {final['val_loss']:.4f} | best {best_val:.4f}")
        print(f"best checkpoint: {best_path}")
    return summary


def run_overfit(cfg: dict, steps: int = 300, lr: float = 3e-3, batch_size: int = 4, quiet: bool = False) -> list[float]:
    """Debugging gate: train on ONE tiny repeated batch. If gradients / target shifting / masking are right,
    the loss collapses towards zero. Returns the list of per-step losses."""
    set_seed(cfg["seed"])
    text = load_text(cfg["data"]["path"])
    tok = CharTokenizer.from_text(text)
    data, _ = build_splits(text, tok, cfg["data"]["val_fraction"])
    mc = dict(cfg["model"]); mc.update(dropout=0.0)               # no dropout so memorisation is possible
    model = GPT(vocab_size=tok.vocab_size, **mc)
    x, y = get_batch(data, batch_size, mc["context_length"])
    assert torch.equal(x[:, 1:], y[:, :-1]), "targets must be inputs shifted by one"
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for s in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if not quiet and (s % 25 == 0 or s == steps - 1):
            print(f"overfit step {s:>4d} | loss {loss.item():.4f}")
    return losses


def main():
    ap = argparse.ArgumentParser(description="Train MiniGPT")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="section.key=value", help="override config values")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/<run>_resume.pt if it exists")
    ap.add_argument("--overfit_tiny", action="store_true", help="overfit one tiny batch (debug gate) and exit")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.overfit_tiny:
        losses = run_overfit(cfg)
        print(f"\ninitial loss {losses[0]:.4f} -> final loss {losses[-1]:.4f}")
        return
    train(cfg, run_name=args.run_name, resume=args.resume)


if __name__ == "__main__":
    main()
