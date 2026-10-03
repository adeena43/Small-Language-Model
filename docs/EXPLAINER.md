# EXPLAINER — every component in plain language (for the viva)

Read this with the code open. Each section names the file/function to point at during the demo.

## 1. Big picture
A language model assigns probabilities to the next token. We turn text into token ids, feed a window of ids to a stack of Transformer blocks,
and at **every position** predict the token that comes next. Training minimises cross-entropy between that prediction and the real next token.
Generation repeats: predict → sample a token → append → predict again (autoregression).

## 2. Why Transformers (`experiments/why_transformers.py`)
An RNN computes h_t from h_{t-1}, so T tokens need **T sequential steps** (sequential depth O(T)) and information between far tokens travels
through O(T) steps. Self-attention connects all pairs in one matrix multiply: **sequential depth O(1)**, path length O(1) — much better for
GPUs. The price: a T×T score matrix per head → **O(T²) memory/compute**. (On a 1-core CPU there is no parallel hardware, so the measured
advantage only shows on a GPU — that is an honest limitation of the comparison, not of the idea.)

## 3. Self-attention (`src/attention.py::scaled_dot_product_attention`)
Each token vector x is projected to a **query** (what I'm looking for), a **key** (what I offer) and a **value** (what I hand over).
`scores = Q Kᵀ / √d_k` → similarity of every query with every key; `softmax` turns each row into weights summing to 1;
`output = weights @ V` is a weighted average of values. **Why divide by √d_k?** Dot products of d_k-dimensional vectors have variance ∝ d_k;
without scaling the softmax saturates (almost one-hot) and gradients vanish. Hand example: `experiments/hand_calculation.py`.

## 4. Masking
Causal mask = lower-triangular boolean matrix. We set blocked scores to **−inf before the softmax** so exp(−inf)=0 and the row still sums to 1
over the allowed positions. (Masking *after* softmax would leave rows not summing to 1.) Token 1 sees only itself, token t sees 1..t.
Proof test: `tests/test_mask.py` (editing future tokens does not change earlier logits).

## 5. Multi-head attention (`MultiHeadAttention`)
One head can only express one "pattern" of looking. We split d_model into H heads of size d_h = d_model/H, run attention in each in parallel,
concatenate and mix with an output projection W_o. Shapes: `(B,T,D) → Linear → (B,T,D) → view/transpose → (B,H,T,d_h)` → attention
`(B,H,T,T)` → `(B,H,T,d_h)` → transpose/view → `(B,T,D)` → W_o → `(B,T,D)`. Trace: `experiments/shape_trace.py` / `trace=True`.

## 6. Positional encoding (`src/positional_encoding.py`)
Attention is permutation-invariant: shuffle the tokens and each token's output is just shuffled too. So we **add** position information:
PE[pos,2i]=sin(pos/10000^{2i/d}), PE[pos,2i+1]=cos(...). Different dimensions oscillate at different frequencies (like digits of a clock), so each
position gets a unique pattern and relative offsets are a linear function of the encodings. It's a buffer, not a learned parameter. Optional: learned embeddings.

## 7. Transformer block (`src/transformer_block.py`)
`x = x + MHA(LN(x))` then `x = x + FFN(LN(x))`.
* **Residual connections**: give gradients a direct path and let each layer learn a *correction* to x.
* **LayerNorm**: normalises each token vector (mean 0, var 1, then learned scale/shift) → stable training. **Pre-LN** (norm before the sub-layer) trains more stably than the original post-LN.
* **FFN**: Linear(d→4d) → GELU → Linear(4d→d), applied to each position independently; attention *mixes information across positions*, the FFN *processes it per position* (and stores much of the model's "knowledge").
* **GELU**: smooth ReLU-like nonlinearity.

## 8. The full model (`src/gpt.py`)
token ids → embedding (×√d) → + positional encoding → dropout → N blocks → final LayerNorm → LM head Linear(d→V) = **logits** (B,T,V).
Parameter count ≈ 12·d²·N + embeddings (3.19M for d=256, N=4). Initial loss should be ≈ ln(V) = 4.17 for V=65 (uniform guess) — a quick sanity check (tested).

## 9. Causal LM objective, shifted targets, loss (`src/dataset.py::get_batch`, `GPT.forward`)
For a window t[i..i+T]: input x = t[i:i+T], target y = t[i+1:i+T+1]. At position k the model sees x[0..k] and must predict y[k]=x[k+1].
Loss = mean over B·T positions of cross-entropy(logits[k], y[k]). Perplexity = exp(loss) = "effective number of equally likely choices".
One forward pass yields T training signals per sequence. Verified by `tests/test_overfit.py`.

## 10. MLM (BERT) vs CLM (GPT) (`experiments/mask_comparison.py`)
BERT: hide ~15 % of tokens with [MASK], attend **bidirectionally**, predict only the hidden tokens → great representations, cannot generate left-to-right.
GPT: predict the next token with a **causal mask** → can generate. Experiment: the same model on the next-token task *without* the causal mask reaches
near-zero loss because it simply reads the next token from the right (cheating) — which is exactly why GPT needs the mask.

## 11. Training details (`src/train.py`)
AdamW (Adam + decoupled weight decay; no decay on biases/LayerNorm), learning-rate warm-up (avoids early instability) then cosine decay,
gradient clipping 1.0 (prevents exploding updates), dropout 0.1 (regularisation), validation loss on held-out text to detect over-fitting.
**Overfit gate**: a correct model/loss/shift/mask must be able to memorise one tiny batch — if it can't, there is a bug.

## 12. Generation (`GPT.generate`)
Take logits of the **last** position, divide by temperature (<1 sharper/safer, >1 flatter/more random), optional top-k/top-p truncation, softmax,
sample, append, crop to the last `context_length` tokens, repeat. Greedy (argmax) is deterministic and tends to repeat.

## 13. Likely viva questions (short answers)
1. *Why scale by √d_k?* Keeps score variance ~1 so softmax doesn't saturate.
2. *Why −inf and not 0 for masked scores?* exp(0)=1 would still give weight; exp(−inf)=0 gives exactly zero.
3. *Why do we need Q, K and V separately instead of x·xᵀ?* Separate learned projections let "what I look for", "what I advertise" and "what I pass on" differ.
4. *What is the shape of the attention matrix and why is it O(T²)?* (B,H,T,T): every token scores every token.
5. *Why multiple heads if total size is the same?* Different heads can specialise in different relations (e.g. previous char, same speaker, line start).
6. *What happens without positional encoding?* The model is a bag-of-tokens; "dog bites man" = "man bites dog".
7. *Why sinusoids?* Unique per position, bounded, no parameters, relative offsets are linear transforms, can extrapolate somewhat.
8. *Pre-LN vs post-LN?* Pre-LN keeps a clean residual path → more stable gradients, no warm-up fragility.
9. *Role of the FFN?* Per-token nonlinear processing; most of the block's parameters.
10. *How do targets relate to inputs?* Targets are inputs shifted left by one.
11. *Why is initial loss ≈ ln(65)?* Near-uniform predictions over 65 characters.
12. *How do you prove no future leakage?* `test_full_model_never_uses_future_tokens` (and the control `test_without_mask_the_future_leaks`).
13. *Why does validation loss go above training loss / when to stop?* Over-fitting; we keep the checkpoint with the best validation loss.
14. *What does temperature do?* Rescales logits before softmax; lower = more deterministic.
15. *Why can BERT not generate text like GPT?* It is trained with bidirectional context to fill masks, not to continue a prefix.
16. *Parameter count of one block?* ≈ 4d² (attention) + 8d² (FFN) = 12d² (+ biases/LN).
17. *Weight tying?* Share the embedding and LM-head matrices: fewer parameters, often a small gain.
18. *Limitations of your model?* Char-level, tiny data and context, no KV cache, single seed ablations.
