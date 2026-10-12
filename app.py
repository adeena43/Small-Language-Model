"""Hugging Face Space entry point (Gradio SDK / ZeroGPU).

Everything is served by Gradio's own server on port 7860:
    /          custom web page (src/static/index.html) -> calls POST /generate
    /generate  REST API    /health    /info
    Gradio UI  is the fallback demo (see below)
The model itself runs on CPU; the @spaces.GPU decorator only satisfies ZeroGPU's startup check.
"""
import gradio as gr

try:
    import spaces
    gpu = spaces.GPU(duration=10)
except ImportError:                     # local run / CPU Space: no `spaces` package
    def gpu(fn):
        return fn

from src.api import app as api_app, STATE, _load, _require_model
from src.generate import generate_text

if STATE.get("model") is None:          # load the checkpoint once at import time
    _load()


@gpu
def gradio_generate(prompt, max_new_tokens, temperature, top_k):
    model, tok = _require_model()
    return generate_text(model, tok, prompt or "ROMEO:", int(max_new_tokens), float(temperature),
                         top_k=int(top_k) if top_k else None)


demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(value="ROMEO:", label="Prompt", lines=2),
        gr.Slider(10, 500, value=150, step=10, label="Max new tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(0, 65, value=0, step=1, label="Top-k (0 = off)"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=12),
    title="MiniGPT - Shakespeare (from scratch)",
    description="Character-level GPT (3.2M params) trained from scratch on Tiny Shakespeare.",
)

server_app, _, _ = demo.launch(prevent_thread_lock=True, ssr_mode=False)

# Put our FastAPI routes in front of Gradio's so "/" shows the custom page and /generate, /health, /info work.
ours = [r for r in api_app.router.routes if getattr(r, "path", None) in ("/", "/health", "/info", "/generate")]
server_app.router.routes[0:0] = ours

demo.block_thread()