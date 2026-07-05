"""Gradio demo for Tiny Latent Control Lab.

Polished, shareable demo surface. Live mode requires the distilgpt2 + probe artifacts.

This file demonstrates the same zero-generation invariant:
text -> frozen forward pass + hook -> vector -> probe -> enum action -> kernel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from neural_native.llm.extractor import extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm
from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_grid_html
from neural_native.vectorbot.router import VectorBotLinearProbeRouter, thresholds_from_json
from neural_native.vectorbot.vector_port import VectorBotVectorPort

MODEL_ID = "distilgpt2"
PROBE_PATH = Path("artifacts/vectorbot_probe_distilgpt2_full.joblib")
if not PROBE_PATH.exists():
    PROBE_PATH = Path("artifacts/vectorbot_probe_distilgpt2.joblib")
THRESHOLDS_PATH = Path("artifacts/vectorbot_thresholds_full.json")
if not THRESHOLDS_PATH.exists():
    THRESHOLDS_PATH = Path("artifacts/vectorbot_thresholds.json")

DEMO_PROMPTS = [
    "go north",
    "take one step south",
    "slide left",
    "move the bot right",
    "toggle the lamp",
    "reset the board",
    "what is the weather tomorrow?",
    "delete all files on this laptop",
    "move up and then right",
]


def _try_load_runtime():
    """Lazy + guarded load so the app can be imported even without torch/llm extras."""
    if not (PROBE_PATH.exists() and THRESHOLDS_PATH.exists()):
        return None, None, None, None
    try:
        thresholds = thresholds_from_json(THRESHOLDS_PATH)
        router = VectorBotLinearProbeRouter(PROBE_PATH, thresholds=thresholds)
        tokenizer, model = load_causal_lm(MODEL_ID, use_4bit=False)
        tap = PreLMHeadActivationTap(model)
        return tokenizer, model, tap, router
    except Exception:
        return None, None, None, None


# Do not load at import time (torch may be missing in minimal envs)
TOKENIZER = MODEL = TAP = ROUTER = None
LIVE_OK = False

def _ensure_runtime():
    global TOKENIZER, MODEL, TAP, ROUTER, LIVE_OK
    if ROUTER is not None:
        return
    TOKENIZER, MODEL, TAP, ROUTER = _try_load_runtime()
    LIVE_OK = ROUTER is not None


def _simulated_naive(text: str) -> dict[str, Any]:
    t = (text or "").lower()
    if any(x in t for x in ["delete", "erase", "rm", "shutdown"]):
        return {
            "action": "SHELL: rm -rf /",
            "would_execute": True,
            "why": "Naive generation often emits dangerous commands when asked to 'help'.",
        }
    if "and" in t and ("up" in t or "left" in t):
        return {
            "action": "MOVE (partial compound)",
            "would_execute": True,
            "why": "Text parsers frequently mis-handle conjunctions.",
        }
    return {
        "action": "MOVE_UP (heuristic)",
        "would_execute": True,
        "why": "Best-guess from text without native OOD gate.",
    }


def route(text: str, kernel_state: dict | None = None) -> tuple[str, str, str, str, dict]:
    """Core live route. Returns (html_grid, action, details, comparison, new_state_dict)."""
    kernel = VectorBotKernel()
    if kernel_state:
        for k, v in kernel_state.items():
            if hasattr(kernel.state, k):
                setattr(kernel.state, k, v)

    _ensure_runtime()

    if not LIVE_OK or not text:
        grid = render_grid_html(kernel)
        naive = _simulated_naive(text or "go north")
        return (
            grid,
            "ABSTAIN (demo)",
            "Live model unavailable — using replay mode.",
            json.dumps(naive, indent=2),
            kernel.snapshot(),
        )

    port = VectorBotVectorPort(router=ROUTER, app=kernel)
    vec = extract_vectors([text], TOKENIZER, MODEL, TAP, batch_size=1, max_length=96)[0]
    result = port.ingest(vec, raw_text=text)

    route_info = result.get("route", {})
    grid_html = render_grid_html(kernel)
    action = route_info.get("label", "—")
    details = (
        f"accepted={route_info.get('accepted')}  "
        f"conf={route_info.get('confidence', 0):.3f}  "
        f"margin={route_info.get('margin', 0):.3f}"
    )
    comp = json.dumps(_simulated_naive(text), indent=2)
    return grid_html, action, details, comp, kernel.snapshot()


def make_demo():
    with gr.Blocks(title="Tiny Latent Control Lab") as demo:
        gr.Markdown(
            "# Tiny Latent Control Lab\n"
            "**Zero-generation latent action routing** — frozen `distilgpt2` activations control a sandboxed VectorBot via a probe. "
            "No `model.generate()`, no tool-call parsing."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Command", value="go north", lines=1)
                run = gr.Button("Route (latent vector)", variant="primary")
                gr.Examples(examples=DEMO_PROMPTS, inputs=prompt)
                reset = gr.Button("Reset Grid")

            with gr.Column(scale=1):
                grid_out = gr.HTML(label="VectorBot (live state + trail)")

        with gr.Row():
            with gr.Column():
                action_out = gr.Textbox(label="Latent-routed action", interactive=False)
                detail_out = gr.Textbox(label="Probe details", interactive=False)
            with gr.Column():
                gr.Markdown("**What a naive generation router might do (illustrative)**")
                naive_out = gr.Code(language="json")

        state = gr.State()

        def _on_route(text):
            g, a, d, c, s = route(text)
            return g, a, d, c, s

        run.click(_on_route, inputs=[prompt], outputs=[grid_out, action_out, detail_out, naive_out, state])
        prompt.submit(_on_route, inputs=[prompt], outputs=[grid_out, action_out, detail_out, naive_out, state])

        def _on_reset():
            k = VectorBotKernel()
            return render_grid_html(k), "—", "", "{}", k.snapshot()

        reset.click(_on_reset, outputs=[grid_out, action_out, detail_out, naive_out, state])

        gr.Markdown(
            "The routing decision is made exclusively from the pre-lm-head activation vector. "
            "The kernel receives only a typed enum. This is the core of the 'neural-native' idea."
        )

    return demo


if __name__ == "__main__":
    demo = make_demo()
    demo.launch()
