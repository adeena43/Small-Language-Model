# Small-Language-Model — MiniGPT from scratch (PyTorch)

A small GPT-style **causal language model** whose Transformer is implemented by hand (scaled dot-product attention,
multi-head attention, sinusoidal positional encoding, pre-LN decoder blocks, LM head, next-token loss, autoregressive
sampling) and trained **from scratch** on **Tiny Shakespeare** (character level). No pretrained weights and no
`transformers` model classes are used. The trained model is exposed through a CLI and a FastAPI service.

> Course assignment: *AI Engineering — Transformer & GPT From Scratch* (handout in `docs/A1_AIE.pdf`).

## Links

| Deliverable | Link |
|---|---|
| Live demo (web page + API) | [adeena43-slm.hf.space](https://adeena43-slm.hf.space) |
| Technical report (Medium article) | [Building a Small GPT from Scratch](https://medium.com/@hf.rashid01/building-a-small-gpt-from-scratch-afb5f7720148) |
| Technical report (PDF) | [`report/MiniGPT_Report.pdf`](report/MiniGPT_Report.pdf) |
| Colab notebook | [Open in Colab](https://colab.research.google.com/github/adeena43/Small-Language-Model/blob/main/notebooks/colab_demo.ipynb) · [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) |
| Demo video | [Watch on Google Drive](https://drive.google.com/file/d/19bNzfXgW0sel1BA62IG9oB-tnQTws90m/view?usp=drive_link) |
| Trained checkpoint | [`checkpoints/best.pt`](checkpoints/best.pt) |

The live demo runs on a free Hugging Face Space: the first request after it has been idle can take a little longer while the Space restarts.

## Repository map

| What | Where |
|---|---|
| Tokenizer (char-level, vocab policy documented) | `src/tokenizer.py` |
| Data pipeline (download, split, shifted targets, batches) | `src/dataset.py` |
| Attention: scaled dot-product, single head, **multi-head (explicit split/merge)**, causal mask | `src/attention.py` |
| Sinusoidal (and optional learned) positional encoding | `src/positional_encoding.py` |
| Transformer block (LN → MHA → residual → LN → FFN(GELU) → residual) | `src/transformer_block.py` |
| Full model (`GPT`), loss, `generate()`, parameter summary | `src/gpt.py` |
| Training loop (AdamW, warmup+cosine, logging, checkpoints, overfit gate) | `src/train.py` |
| Generation CLI | `src/generate.py` |
| REST API (FastAPI) | `src/api.py` |
| Web page (served at `/`, calls `POST /generate`) | `src/static/index.html` |
| Config | `configs/tiny_gpt.yaml` |
| Unit tests | `tests/` |
| Experiments / evidence figures | `experiments/` → results in `outputs/` |
| Colab runner | `notebooks/colab_demo.ipynb` |
| Viva-ready explanation of every component | `docs/EXPLAINER.md` |
| Technical report (PDF), handwritten calculation and screenshots | `report/` |

## 1. Setup

Python 3.10+.

```bash
git clone https://github.com/adeena43/Small-Language-Model.git
cd Small-Language-Model
pip install -r requirements-train.txt      # torch, numpy, fastapi, matplotlib, pytest, ...
```

The dataset (`data/input.txt`, 1.1 MB) is downloaded automatically on first use (see `data/README.md`).
**Training is fast on a GPU** (Colab T4: minutes). On a CPU use the small settings shown below.

## 2. Verify the implementation (tests + architecture checks)

```bash
python -m pytest -q                              # 31 tests: attention, causal mask, shapes, overfit, checkpoint reload, resume, API
python experiments/hand_calculation.py           # Fig 1 + Fig 2: hand-calculated attention and causal mask
python experiments/shape_trace.py                # Fig 3 (multi-head shape trace) + Fig 5 (model summary / parameter count)
python experiments/positional_encoding_viz.py    # Fig 4
python experiments/architecture_diagram.py       # architecture diagram
python experiments/mask_comparison.py            # BERT-style (bidirectional) vs GPT-style (causal) comparison
python experiments/why_transformers.py           # sequential vs parallel depth: measured timings (run on a GPU for the fair comparison)
python -m src.train --overfit_tiny               # debugging gate: loss must collapse on one tiny batch
```

Key correctness test: `tests/test_mask.py::test_full_model_never_uses_future_tokens` changes tokens *after* position *t* and asserts the
logits at positions ≤ *t* are bit-for-bit unchanged (**a token at position t cannot use information from positions > t**).

## 3. Train

```bash
python -m src.train --config configs/tiny_gpt.yaml           # default: d_model 256, 4 layers, 4 heads, context 128 (~3.2M params)
python -m src.train --set train.max_steps=3000 train.batch_size=32     # any config value can be overridden
# CPU-friendly small model:
python -m src.train --set model.d_model=128 model.ffn_dim=512 model.n_layers=4 train.max_steps=2000 train.batch_size=32
```

**Timeouts (Colab):** add `--resume` and re-run the same command after a disconnect; the run continues from the last saved step
(`checkpoints/main_resume.pt`, written every `train.save_interval` steps). Ablation runs always resume automatically.

Outputs: `checkpoints/best.pt` (best validation loss), `checkpoints/last.pt`, `outputs/logs/main_log.csv`,
`outputs/logs/main_summary.json` (params, hardware, wall-clock time, final train/val loss), `outputs/figures/main_loss_curve.png`
(**Fig 6**), `outputs/samples/main_training_samples.txt`. Logged: step, epoch, train loss, validation loss, learning rate, tokens/sec.

## 4. Inference (CLI)

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
python -m src.generate --prompt "To be, or not to be" --top-k 40 --temperature 0.9
python -m src.generate --prompt "ROMEO:" --greedy
python -m src.generate --fixed-prompts            # 5 fixed prompts -> outputs/samples/fixed_prompt_samples.txt (Fig 8)
```

Expected output format (numbers/text will differ for your run):

```
Prompt: 'ROMEO:'
Generated: ROMEO: ...
model parameters: 3.19M
context length: 128
temperature: 0.8
```

Checkpoints are self-contained (weights + model config + vocabulary), so a fresh process can load and generate with no other file.

## 5. Evaluation, attention inspection, ablations

```bash
python experiments/evaluate.py --checkpoint checkpoints/best.pt                     # full validation loss, perplexity, 5 fixed-prompt samples
python experiments/attention_visualization.py --checkpoint checkpoints/best.pt      # Fig 7
python experiments/ablation.py --group depth heads context --steps 3000             # 3 required experiments -> Fig 9 + table
python experiments/ablation.py --group position --steps 3000                        # optional: sinusoidal vs learned
python experiments/ablation.py --group generation --checkpoint checkpoints/best.pt  # optional: greedy vs sampling
```

Each ablation changes **one** factor (same seed, data, optimiser, number of steps). Results: `outputs/ablation/ablation_results.md|csv`.
Note for the *context length* experiment: batch size is held constant, so longer context means more tokens per step.

## 6. API (FastAPI)

```bash
uvicorn src.api:app --reload                      # web page: http://127.0.0.1:8000   API docs: /docs
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/info                   # model info (parameters, context length, ...)
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 120, "temperature": 0.8}'
```

Endpoints: `GET /` (web page), `GET /info`, `GET /health`, `POST /generate` (`prompt`, `max_new_tokens`, `temperature`, optional `top_k`, `seed`).

Environment variables: `CHECKPOINT_PATH` (default `checkpoints/best.pt`), `CHECKPOINT_URL` (optional download URL if the checkpoint is not in the repo).

### Deploy on Hugging Face Spaces
1. Create a new Space on huggingface.co → **New Space** → SDK: **Gradio**, hardware: **ZeroGPU** (free).
2. Upload the repo files to the Space: `app.py`, `src/`, `configs/`, `checkpoints/best.pt` and `requirements.txt`.
3. `app.py` is the entry point. It loads the checkpoint once and serves the web page at `/` plus the `/generate`, `/health` and `/info` endpoints. The model runs on CPU.
4. The Space builds and starts by itself. Open `https://<username>-<space-name>.hf.space` for the web page.
5. Free Spaces sleep when idle, so open the URL a minute before a demo.

Live URL: see the [Links](#links) table at the top.

## 7. Configuration (`configs/tiny_gpt.yaml`, all overridable with `--set section.key=value`)

| parameter | default | note |
|---|---|---|
| vocab_size | 65 | from tokenizer |
| context_length | 128 | |
| d_model | 256 | |
| n_heads | 4 | `d_model % n_heads == 0` |
| n_layers | 4 | |
| ffn_dim | 1024 | 4 × d_model |
| dropout | 0.1 | |
| optimizer / lr | AdamW / 3e-4 | 100 warm-up steps, cosine decay to 3e-5, grad-clip 1.0 |
| batch_size | 64 | reduce on small hardware |
| temperature | 0.8 | generation baseline |
| pos_encoding | sinusoidal | or `learned` |
| weight_tying | false | optional experiment |

## 8. Evidence map (assignment Section 12)

| Fig | Evidence | File |
|---|---|---|
| 1 | Hand-calculated attention | handwritten: report PDF (Figure 1, pages 1 to 5); reproduced by code: `outputs/figures/fig1_hand_attention.png`, `outputs/logs/hand_calculation.txt` |
| 2 | Causal mask | `outputs/figures/fig2_causal_mask.png` |
| 3 | Multi-head shape trace | `report/figures/fig3_shape_trace.png`, `outputs/logs/shape_trace_and_summary.txt` |
| 4 | Positional encoding | `outputs/figures/fig4_positional_encoding.png` |
| 5 | Model summary / parameter count | `report/figures/fig5_model_summary.png` (3,192,832 parameters) |
| 6 | Train vs validation loss | `outputs/figures/main_loss_curve.png` |
| 7 | Attention visualisation (trained model) | `outputs/figures/fig7_attention_visualization.png` |
| 8 | Generation samples | `outputs/samples/fixed_prompt_samples.txt`, `outputs/samples/evaluation_samples.md` |
| 9 | Ablation table / plots | `outputs/ablation/ablation_results.md`, `outputs/figures/fig9_ablation_*.png` |
| 10 | CLI / API call | `report/figures/fig10a_cli.png` (CLI); deployed site: report PDF (Figure 10b) and the live demo link |
| extra | Architecture diagram, MLM-vs-CLM, sequential-vs-parallel | `outputs/figures/architecture_diagram.png`, `fig6b_*`, `fig6c_*`, `why_transformers.png` |

## 9. Known limitations
* Character-level model on a 1 MB corpus: it learns Shakespeare-like *form* (speaker tags, line breaks, archaic spellings) but produces many invented words and no long-range coherence.
* Context window is only 128 characters; the model cannot condition on anything older.
* Generation recomputes the full context at every step (no KV cache) — fine for a demo, slow for long outputs.
* Single run per ablation setting (one seed): small loss differences may be within seed noise.
* Not a factual system — generated text is an experimental artifact.

## 10. Sources and attribution
* Vaswani et al., *Attention Is All You Need* (2017) — attention, multi-head attention, sinusoidal encoding.
* Devlin et al., *BERT* (2018) and Radford et al., *GPT-2* (2019) — MLM vs causal LM objectives; GPT-2-style scaled residual initialisation.
* Karpathy, *nanoGPT* and *Let's build GPT* — conceptual reference for the training setup (AdamW betas, warm-up + cosine schedule, char-level Tiny Shakespeare).
* Hendrycks & Gimpel (GELU); Ba et al. (LayerNorm).
* Course material: `rohitg00/ai-engineering-from-scratch`, `phases/07-transformers-deep-dive` (lessons 02–07), used for concepts only.
* Dataset: Tiny Shakespeare, <https://huggingface.co/datasets/karpathy/tiny_shakespeare>.
