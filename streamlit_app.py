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

    routes = load_routes()
    live_available = PROBE_PATH.exists() and THRESHOLDS_PATH.exists()

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
        st.caption("🟢 Live mode (distilgpt2 + probe) — forward-pass only, no generation on the action path")
    else:
        st.info("Live model artifacts not found — using replay + simulated comparison. Run the quickstart to enable live routing.")

    # Quick command chips
    st.subheader("Try these commands")
    chip_cols = st.columns(min(5, len(DEMO_COMMANDS)))
    clicked_cmd = None
    for i, (cmd, expected) in enumerate(DEMO_COMMANDS[:10]):
        with chip_cols[i % len(chip_cols)]:
            if st.button(cmd, key=f"chip_{i}", use_container_width=True):
                clicked_cmd = cmd

    # Main area
    cmd_tab, replay_tab = st.tabs(["💬 Live / Custom", "📼 Replay log"])

    with cmd_tab:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("**Natural language command**")
            default_cmd = clicked_cmd or "go north"
            command = st.text_input("Input", value=default_cmd, label_visibility="collapsed", key="cmd_input")
            col_run, col_reset = st.columns(2)
            with col_run:
                run_clicked = st.button("🚀 Route with Latent Vector", type="primary", use_container_width=True)
            with col_reset:
                if st.button("↺ Reset grid", use_container_width=True):
                    st.session_state.kernel = VectorBotKernel()
                    st.rerun()

            result = None
            if run_clicked and live_available:
                try:
                    result = route_live(command)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Live routing unavailable. {exc}")
            elif run_clicked and not live_available:
                st.info("Live model not loaded. Using simulated latent result for UI demo.")

        with c2:
            st.markdown("**Current state (latent-routed)**")
            grid_html = render_grid_html(st.session_state.kernel)
            st.markdown(grid_html, unsafe_allow_html=True)

    with replay_tab:
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

    # Route decision panel + comparison
    st.divider()
    st.subheader("Routing decision")

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**Latent router (ours — zero generation)**")
        route = None
        if result is not None:
            route = result.get("route", {})
            # Also sync kernel state if result carried after
            if "after" in result:
                st.session_state.kernel.state.x = result["after"].get("x", st.session_state.kernel.state.x)
                # (full sync is done inside port; we rely on it)
        if route is None and routes:
            # fallback to last replayed if any
            pass

        if route:
            accepted = route.get("accepted", False)
            st.metric("Predicted action", route.get("label", "—"))
            st.metric("Confidence", f"{route.get('confidence', 0):.3f}")
            st.metric("Accepted → kernel", "✅ YES" if accepted else "🚫 ABSTAIN")
            top = route.get("top_probabilities", [])
            if top:
                dfp = pd.DataFrame(top)
                st.bar_chart(dfp.set_index("label")["probability"], height=180)
            if "ood_score" in route:
                st.caption(f"margin={route.get('margin', 0):.3f} | ood={route.get('ood_score', 0):.3f}")
        else:
            st.caption("Enter a command and click Route (or use chips) to see live latent routing.")
            st.code(render_ascii(st.session_state.kernel.snapshot()), language="text")

    with r2:
        st.markdown("**Naive generation-based router (simulated)**")
        sim_text = command if "command" in locals() else (clicked_cmd or "go north")
        sim = _simulated_generation_router(sim_text)
        color = "#ef4444" if sim.get("danger") else "#f59e0b"
        st.markdown(
            f'<div style="padding:8px 12px;border-radius:6px;background:{color}22;border:1px solid {color};">'
            f'<strong>{sim["label"]}</strong><br/>'
            f'<span style="font-size:12px;">{sim["note"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("Typical real-world path: LLM emits text/JSON/tool call → parser → execute. No native ABSTAIN for unsafe intent.")

    # Latent space + log (bottom)
    st.divider()
    bl, br = st.columns([1.35, 1])

    with bl:
        st.subheader("Latent space (PCA of pre-lm-head activations)")
        if PROJECTION_PATH.exists() and px is not None:
            proj = pd.read_csv(PROJECTION_PATH)
            fig = px.scatter(
                proj,
                x="x",
                y="y",
                color="label",
                hover_data=["split", "text", "accepted"] if "text" in proj.columns else None,
                height=380,
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True, key="latent")
        elif PROJECTION_PATH.exists():
            st.dataframe(pd.read_csv(PROJECTION_PATH).head(200))
        else:
            st.caption("Projection CSV not found. Run build_vectorbot_visuals.py to generate.")

    with br:
        st.subheader("Recent routes (from artifact)")
        if routes:
            df = pd.DataFrame(routes)
            cols = [c for c in ["input_text", "predicted_label", "accepted", "confidence", "vector_norm"] if c in df.columns]
            st.dataframe(df[cols].tail(12), use_container_width=True, height=320)
        else:
            st.caption("No routes log found.")

    # Footer note
    st.caption(
        "Core invariant: action is chosen exclusively from the activation vector + probe. "
        "Raw text is logged only for audit. The kernel never sees or parses natural language."
    )


if __name__ == "__main__":
    main()
