from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from neural_native.llm.extractor import extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm
from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_ascii, render_grid_html
from neural_native.vectorbot.router import VectorBotLinearProbeRouter, thresholds_from_json
from neural_native.vectorbot.vector_port import VectorBotVectorPort

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dashboard dependency
    px = None

import streamlit.components.v1 as components


MODEL_ID = "distilgpt2"
PROBE_PATH = Path("artifacts/vectorbot_probe_distilgpt2_full.joblib")
if not PROBE_PATH.exists():
    PROBE_PATH = Path("artifacts/vectorbot_probe_distilgpt2.joblib")
THRESHOLDS_PATH = Path("artifacts/vectorbot_thresholds_full.json")
if not THRESHOLDS_PATH.exists():
    THRESHOLDS_PATH = Path("artifacts/vectorbot_thresholds.json")
ROUTES_PATH = Path("artifacts/vectorbot_routes_full.jsonl")
if not ROUTES_PATH.exists():
    ROUTES_PATH = Path("artifacts/vectorbot_routes.jsonl")
PROJECTION_PATH = Path("artifacts/vectorbot_projection_full.csv")
if not PROJECTION_PATH.exists():
    PROJECTION_PATH = Path("artifacts/vectorbot_projection.csv")


def _can_load_llm() -> bool:
    """Safely check if torch + transformers are importable without triggering full load."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource(show_spinner=False)
def load_runtime():
    thresholds = thresholds_from_json(THRESHOLDS_PATH) if THRESHOLDS_PATH.exists() else None
    router = VectorBotLinearProbeRouter(PROBE_PATH, thresholds=thresholds)
    tokenizer, model = load_causal_lm(MODEL_ID, use_4bit=False)
    tap = PreLMHeadActivationTap(model)
    return tokenizer, model, tap, router


def load_routes() -> list[dict]:
    if not ROUTES_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in ROUTES_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def route_live(text: str) -> dict:
    tokenizer, model, tap, router = load_runtime()
    if "kernel" not in st.session_state:
        st.session_state.kernel = VectorBotKernel()
    port = VectorBotVectorPort(router=router, app=st.session_state.kernel)
    vector = extract_vectors([text], tokenizer, model, tap, batch_size=1, max_length=96)[0]
    return port.ingest(vector, raw_text=text)


DEMO_COMMANDS = [
    ("go north", "MOVE_UP"),
    ("take one step south", "MOVE_DOWN"),
    ("slide left", "MOVE_LEFT"),
    ("move the bot right", "MOVE_RIGHT"),
    ("toggle the lamp", "TOGGLE_LIGHT"),
    ("reset the board", "RESET"),
    ("what is the weather tomorrow?", "ABSTAIN"),
    ("delete all files on this laptop", "ABSTAIN"),
    ("move up and then right", "ABSTAIN"),
    ("paint the robot purple", "ABSTAIN"),
]


def _simulated_generation_router(text: str) -> dict:
    """Simulated 'naive LLM generate + parse' path for comparison demo.

    This is intentionally scripted to illustrate common failure modes.
    It does NOT call any model.generate() and is only for the UI narrative.
    """
    t = text.lower()
    if any(kw in t for kw in ["delete", "rm -rf", "erase", "format", "shutdown"]):
        return {
            "label": "SHELL:rm -rf /",
            "accepted": True,  # would have been accepted by naive system
            "confidence": 0.92,
            "note": "Generated unsafe shell command (would execute in naive agent)",
            "danger": True,
        }
    if "weather" in t or "time" in t or "paint" in t:
        return {
            "label": "UNKNOWN",
            "accepted": False,
            "confidence": 0.41,
            "note": "Vague or out-of-scope → often still tries to call a tool or hallucinates",
            "danger": False,
        }
    if "and" in t and ("up" in t or "left" in t or "right" in t):
        return {
            "label": "COMPOUND_MOVE",
            "accepted": True,
            "confidence": 0.78,
            "note": "Compound parsed incorrectly or partially executed",
            "danger": False,
        }
    # Default: pretend it mapped somewhat
    return {
        "label": "MOVE_UP (guessed)",
        "accepted": True,
        "confidence": 0.65,
        "note": "Text-based parser / tool call guess (brittle)",
        "danger": False,
    }


def render_3b1b_under_the_bonnet(full_probs: list[dict[str, float]] | None = None) -> str:
    """3Blue1Brown-inspired real-time probability distribution visual.

    Shows animated bars for the model's predicted likelihood (after softmax)
    of each possible action. The bars animate from uniform/zero to the actual
    output of the linear probe.
    """
    if not full_probs:
        # Fallback demo values if somehow not provided
        full_probs = [
            {"label": "MOVE_UP", "probability": 0.15},
            {"label": "MOVE_DOWN", "probability": 0.12},
            {"label": "MOVE_LEFT", "probability": 0.08},
            {"label": "MOVE_RIGHT", "probability": 0.55},
            {"label": "TOGGLE_LIGHT", "probability": 0.04},
            {"label": "RESET", "probability": 0.03},
            {"label": "ABSTAIN", "probability": 0.03},
        ]

    # Sort for consistent order (optional, or keep model order)
    label_order = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "TOGGLE_LIGHT", "RESET", "ABSTAIN"]
    prob_map = {p["label"]: p["probability"] for p in full_probs}
    ordered = [{"label": lab, "probability": prob_map.get(lab, 0.0)} for lab in label_order]

    # Build bar HTML
    bars_html = ""
    max_p = max(p["probability"] for p in ordered) or 1.0
    for i, p in enumerate(ordered):
        pct = p["probability"] * 100
        # Use CSS var for the final width so we can animate it
        bars_html += f'''
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="width:110px; font-size:12px; color:#bae6fd; text-align:right; font-family:monospace;">{p["label"]}</div>
            <div style="flex:1; background:#1e2937; height:22px; border-radius:3px; position:relative; overflow:hidden; border:1px solid #334155;">
                <div class="prob-bar" data-pct="{pct}" style="
                    --final-width: {pct}%;
                    width: 0%;
                    height:100%;
                    background: linear-gradient(90deg, #22d3ee, #67e8f9);
                    transition: width 1200ms cubic-bezier(0.23, 1.0, 0.32, 1);
                    box-shadow: 0 0 8px #67e8f9;
                "></div>
            </div>
            <div style="width:60px; font-size:12px; color:#e0f2fe; font-family:monospace; text-align:left;">{pct:.2f}%</div>
        </div>
        '''

    return f'''
    <div style="
        background: #0a0f1e;
        border: 2px solid #1e40af;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0 16px;
        font-family: ui-monospace, monospace;
        color: #bae6fd;
    ">
        <div style="font-size:15px; font-weight:600; color:#67e8f9; margin-bottom:6px;">
            Probe output — predicted probability distribution
        </div>
        <div style="font-size:10px; color:#64748b; margin-bottom:8px;">
            The linear probe turns the 768-dimensional activation vector into a distribution over the 7 possible actions (sums to ~100%).
        </div>

        {bars_html}

        <div style="margin-top:6px; font-size:9px; color:#475569;">
            These are the model's "beliefs" right after the forward pass. The gate then decides whether to act or ABSTAIN.
        </div>
    </div>

    <script>
    // Trigger the bar animations shortly after render
    setTimeout(() => {{
        document.querySelectorAll('.prob-bar').forEach(bar => {{
            const final = bar.getAttribute('data-pct') || '0';
            bar.style.width = final + '%';
        }});
    }}, 80);
    </script>
    '''

def _apply_replay_to_kernel(replay_row: dict) -> None:
    if "kernel" not in st.session_state:
        st.session_state.kernel = VectorBotKernel()
    k = st.session_state.kernel
    sa = replay_row.get("state_after", {})
    k.state.x = int(sa.get("x", k.state.x))
    k.state.y = int(sa.get("y", k.state.y))
    k.state.light_on = bool(sa.get("light_on", k.state.light_on))
    k.state.mode = sa.get("mode", k.state.mode)
    k.state.step_count = int(sa.get("step_count", k.state.step_count))
    k.state.action_history = sa.get("action_history", k.state.action_history)
    # trail support (graceful for old replays)
    trail = sa.get("trail")
    if trail:
        k.state.trail = [tuple(p) for p in trail]


def main() -> None:
    st.set_page_config(
        page_title="Tiny Latent Control Lab • VectorBot",
        page_icon="🧭",
        layout="wide",
    )
    if "kernel" not in st.session_state:
        st.session_state.kernel = VectorBotKernel()

    if "cmd_input" not in st.session_state:
        st.session_state.cmd_input = "go north"

    routes = load_routes()
    live_available = (
        PROBE_PATH.exists()
        and THRESHOLDS_PATH.exists()
        and _can_load_llm()
    )

    # Header
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <div style="font-size:28px;font-weight:700;">Tiny Latent Control Lab</div>
            <div style="font-size:13px;color:#64748b;padding-top:6px;">zero-generation • frozen activations → typed actions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if live_available:
        # Pre-warm the model on first load so the *first* user command is fast
        if "model_warmed" not in st.session_state:
            with st.spinner("Warming up distilgpt2 (one-time on first app load)..."):
                _ = load_runtime()
                # Do a quick dummy forward to warm caches
                try:
                    from neural_native.llm.extractor import extract_vectors
                    tok, mod, tap, _ = load_runtime()  # cached
                    _ = extract_vectors(["warmup"], tok, mod, tap, batch_size=1, max_length=16)
                except Exception:
                    pass
            st.session_state.model_warmed = True
        st.caption("🟢 Live mode (distilgpt2 + probe) — forward-pass only, no generation on the action path  |  ⏱️ Real CPU inference (usually <1s after warmup)")
    else:
        st.info(
            "Live mode unavailable (missing torch or artifacts). "
            "Using replay + simulated comparison. "
            "Install with: pip install -e '.[llm]' and ensure artifacts are present."
        )

    # Quick command chips — clicking only fills the input bar (user then clicks Route)
    st.subheader("Try these commands")
    chip_cols = st.columns(min(5, len(DEMO_COMMANDS)))
    for i, (cmd, expected) in enumerate(DEMO_COMMANDS[:10]):
        with chip_cols[i % len(chip_cols)]:
            if st.button(cmd, key=f"chip_{i}", width="stretch"):
                st.session_state.cmd_input = cmd

    # Main single-page live view
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("**Natural language command**")
        command = st.text_input("Input", label_visibility="collapsed", key="cmd_input")
        col_run, col_reset = st.columns(2)
        with col_run:
            run_clicked = st.button("🚀 Route with Latent Vector", type="primary", width="stretch")
        with col_reset:
            if st.button("↺ Reset grid", width="stretch"):
                st.session_state.kernel = VectorBotKernel()
                st.rerun()

        result = None
        if run_clicked and live_available:
            import time
            t0 = time.time()
            try:
                result = route_live(command)
                dt = time.time() - t0
                st.caption(f"⏱️ Live inference took {dt:.2f}s (CPU forward + hook + probe)")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Live routing unavailable. {exc}")

            # The main explanatory visual: full probability distribution over actions
            if result and "route" in result:
                full_probs = result["route"].get("full_probabilities") or result["route"].get("top_probabilities", [])
                st.markdown("### 🧠 What's happening under the bonnet")
                html = render_3b1b_under_the_bonnet(full_probs)
                components.html(html, height=320, scrolling=False)
                st.caption("The linear probe's softmax output — the model's current belief over every possible action.")
        elif run_clicked and not live_available:
            st.info("Live model not loaded (torch or artifacts missing). Using replay/simulated data.")

    with c2:
        st.markdown("**Current state (latent-routed)**")
        grid_html = render_grid_html(st.session_state.kernel.state)
        st.markdown(grid_html, unsafe_allow_html=True)

    # Optional replay log (collapsed to keep it one focused page)
    with st.expander("📼 Replay log (optional)", expanded=False):
        if not routes:
            st.caption("No route log available. Generate with scripts/run_vectorbot_demo.py or quickstart.")
        else:
            idx = st.selectbox(
                "Select replay entry",
                range(len(routes)),
                format_func=lambda i: f"{i}: {routes[i].get('input_text','?')[:60]}",
            )
            row = routes[idx]
            if st.button("Load this replay into grid"):
                _apply_replay_to_kernel(row)
                st.rerun()
            st.json({
                "input": row.get("input_text"),
                "predicted": row.get("predicted_label"),
                "accepted": row.get("accepted"),
                "confidence": row.get("confidence"),
            })
            st.markdown("**State after (replayed)**", unsafe_allow_html=True)
            st.code(render_ascii(st.session_state.kernel.snapshot()), language="text")

    # Footer note
    st.caption(
        "Core invariant: action is chosen exclusively from the activation vector + probe. "
        "Raw text is logged only for audit. The kernel never sees or parses natural language."
    )


if __name__ == "__main__":
    main()
