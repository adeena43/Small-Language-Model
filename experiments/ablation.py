"""Controlled ablation experiments (one factor changed at a time, same seed / data / steps / optimiser).

Groups
    depth     : n_layers 2 / 4 / 6
    heads     : n_heads 2 / 4 / 8
    context   : context_length 64 / 128 / 256   (batch size is kept constant, so longer context = more tokens per step)
    position  : sinusoidal vs learned            (optional)
    generation: greedy vs temperature / top-k / top-p sampling on a trained checkpoint (optional, no training)

Examples
    python experiments/ablation.py --group depth heads context --steps 3000      # the 3 required experiments
    python experiments/ablation.py --group depth --steps 200 --quick             # fast smoke test on CPU
    python experiments/ablation.py --group generation --checkpoint checkpoints/best.pt

Finished runs are cached in outputs/ablation/logs/<run>_summary.json and skipped on re-run (use --force to redo).
Results: outputs/ablation/ablation_results.csv / .md and outputs/figures/fig9_ablation_*.png
"""
from _common import FIG_DIR, ROOT

import argparse
import json
import re

import pandas as pd
import torch

from src.dataset import load_text
from src.generate import generate_text
from src.train import train
from src.utils import apply_overrides, load_checkpoint, load_config, resolve
from _common import mpl  # noqa: E402

GROUPS = {
    "depth": ("model.n_layers", [2, 4, 6]),
    "heads": ("model.n_heads", [2, 4, 8]),
    "context": ("model.context_length", [64, 128, 256]),
    "position": ("model.pos_encoding", ["sinusoidal", "learned"]),
}
ABL_DIR = ROOT / "outputs" / "ablation"
PROMPT = "ROMEO:"


def word_stats(text: str, corpus_words: set) -> dict:
    """Cheap quantitative proxy for generation quality (character-level models invent non-words)."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return {"valid_word_rate": 0.0, "repeat_3gram_rate": 0.0}
    valid = sum(w in corpus_words for w in words) / len(words)
    tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
    return {"valid_word_rate": valid, "repeat_3gram_rate": rep}


def run_group(group: str, args, base_cfg, corpus_words) -> list[dict]:
    key, values = GROUPS[group]
    rows = []
    for v in values:
        name = f"{group}_{str(v).replace('.', 'p')}"
        summ_path = ABL_DIR / "logs" / f"{name}_summary.json"
        want_d = 64 if args.quick else base_cfg["model"]["d_model"]
        cached = json.loads(summ_path.read_text()) if summ_path.exists() else None
        if cached and not args.force and cached["train_config"]["max_steps"] == args.steps and cached["model_config"]["d_model"] == want_d:
            print(f"[skip] {name} (cached, same steps/size)"); summary = cached
        else:
            ov = [f"{key}={v}", f"train.max_steps={args.steps}", f"train.eval_interval={max(1, args.steps // 10)}", f"train.save_interval={max(1, args.steps // 20)}",
                  f"train.sample_interval={args.steps}", "train.sample_tokens=100", f"train.eval_batches={args.eval_batches}",
                  f"train.out_dir={ABL_DIR}", f"train.checkpoint_dir={ABL_DIR / 'checkpoints'}", "train.log_interval=50"]
            if args.quick:
                ov += ["model.d_model=64", "model.ffn_dim=256", "train.batch_size=16"]
            cfg = apply_overrides(base_cfg, ov)
            print(f"\n=== {name}: {key}={v} | steps={args.steps} ===")
            summary = train(cfg, run_name=name, quiet=True, resume=True)   # resumes automatically after a timeout
            print(f"    params {summary['parameters']:,} | val loss {summary['final_val_loss']:.4f} | {summary['wall_clock_sec']}s")
        # generation-quality proxy from the best checkpoint of this run
        ckpt = ABL_DIR / "checkpoints" / f"{name}_best.pt"
        ws = {"valid_word_rate": None, "repeat_3gram_rate": None}; sample = ""
        if ckpt.exists():
            model, tok, _ = load_checkpoint(ckpt, "cpu")
            torch.manual_seed(1337)
            sample = generate_text(model, tok, PROMPT, 400, 0.8)
            ws = word_stats(sample, corpus_words)
        rows.append({"group": group, "factor": key.split(".")[1], "value": v, "run": name, "params": summary["parameters"],
                     "final_train_loss": summary["final_train_loss"], "final_val_loss": summary["final_val_loss"],
                     "val_perplexity": summary["final_val_perplexity"], "train_time_sec": summary["wall_clock_sec"],
                     "tokens_per_sec": summary["mean_tokens_per_sec"], "peak_cuda_mem_MB": summary["peak_cuda_memory_mb"],
                     "hardware": summary["hardware"], "valid_word_rate": ws["valid_word_rate"],
                     "repeat_3gram_rate": ws["repeat_3gram_rate"], "sample": sample[:300].replace("\n", " / ")})
    return rows


def generation_group(args, corpus_words) -> pd.DataFrame:
    model, tok, _ = load_checkpoint(args.checkpoint, "cpu")
    settings = {"greedy": dict(greedy=True), "temp 0.5": dict(temperature=0.5), "temp 0.8": dict(temperature=0.8),
                "temp 1.2": dict(temperature=1.2), "top-k 10 (t=0.8)": dict(temperature=0.8, top_k=10),
                "top-p 0.9 (t=0.8)": dict(temperature=0.8, top_p=0.9)}
    rows = []
    for name, kw in settings.items():
        torch.manual_seed(1337)
        txt = generate_text(model, tok, PROMPT, 600, **kw)
        rows.append({"strategy": name, **word_stats(txt, corpus_words), "sample": txt[:300].replace("\n", " / ")})
    df = pd.DataFrame(rows)
    df.to_csv(ABL_DIR / "generation_strategy_results.csv", index=False)
    (ABL_DIR / "generation_strategy_results.md").write_text(df.drop(columns="sample").to_markdown(index=False, floatfmt=".3f"))
    print(df.drop(columns="sample").to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--group", nargs="+", default=["depth", "heads", "context"],
                    choices=list(GROUPS) + ["generation", "all"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides applied to every run")
    ap.add_argument("--quick", action="store_true", help="tiny model for a fast CPU smoke test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = ap.parse_args()
    ABL_DIR.mkdir(parents=True, exist_ok=True); (ABL_DIR / "logs").mkdir(exist_ok=True)

    base_cfg = apply_overrides(load_config(args.config), args.set)
    corpus_words = set(re.findall(r"[A-Za-z']+", load_text(base_cfg["data"]["path"]).lower()))
    groups = ["depth", "heads", "context"] if "all" in args.group else args.group
    plt = mpl()
    all_rows = []
    for g in [g for g in groups if g != "generation"]:
        rows = run_group(g, args, base_cfg, corpus_words)
        all_rows += rows
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        x = [str(v) for v in df["value"]]
        axes[0].bar(x, df["final_val_loss"], color="tab:blue"); axes[0].set_title("validation loss (lower = better)")
        axes[1].bar(x, df["train_time_sec"], color="tab:orange"); axes[1].set_title("training time (s)")
        axes[2].bar(x, df["params"] / 1e6, color="tab:green"); axes[2].set_title("parameters (M)")
        for a in axes:
            a.set_xlabel(df["factor"].iloc[0]); a.grid(alpha=0.3, axis="y")
        lo, hi = df["final_val_loss"].min(), df["final_val_loss"].max()
        axes[0].set_ylim(max(0, lo - 0.5 * (hi - lo + 0.05)), hi + 0.5 * (hi - lo + 0.05))
        fig.suptitle(f"Figure 9 ({g} ablation): {args.steps} steps per run, same seed and data")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"fig9_ablation_{g}.png", dpi=150); plt.close(fig)
    if all_rows:
        res = pd.DataFrame(all_rows)
        if (ABL_DIR / "ablation_results.csv").exists() and not args.force:   # merge with earlier groups
            old = pd.read_csv(ABL_DIR / "ablation_results.csv")
            res = pd.concat([old[~old["run"].isin(res["run"])], res], ignore_index=True)
        res.to_csv(ABL_DIR / "ablation_results.csv", index=False)
        md = res.drop(columns=["sample", "hardware"]).to_markdown(index=False, floatfmt=".4f")
        (ABL_DIR / "ablation_results.md").write_text(md)
        print("\n" + md)
    if "generation" in groups:
        generation_group(args, corpus_words)


if __name__ == "__main__":
    main()
