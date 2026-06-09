from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from neural_native.llm.extractor import extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm
from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_ascii
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


def main() -> None:
    st.set_page_config(page_title="Tiny Latent Control Lab - VectorBot", layout="wide")
    if "kernel" not in st.session_state:
        st.session_state.kernel = VectorBotKernel()

    routes = load_routes()
    live_available = PROBE_PATH.exists() and THRESHOLDS_PATH.exists()
    st.title("Tiny Latent Control Lab")
    if live_available:
        st.caption("Live mode uses local distilgpt2/probe artifacts; replay is always available.")
    else:
        st.info("Live model/probe artifacts are missing. Showing replay mode from route logs.")

    left, center, right = st.columns([0.95, 1.2, 0.95])
    with left:
        st.subheader("Command")
        command = st.text_input("VectorBot input", value="go north", label_visibility="collapsed")
        run_clicked = st.button("Route", type="primary", use_container_width=True)
        replay_index = 0
        if routes:
            replay_index = st.selectbox(
                "Replay",
                range(len(routes)),
                format_func=lambda i: routes[i]["input_text"],
            )

    result = None
    replay_row = routes[replay_index] if routes else None
    if run_clicked and live_available:
        try:
            result = route_live(command)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Live model unavailable; showing replay data. {exc}")
    elif replay_row is not None:
        st.session_state.kernel.state.x = replay_row["state_after"]["x"]
        st.session_state.kernel.state.y = replay_row["state_after"]["y"]
        st.session_state.kernel.state.light_on = replay_row["state_after"]["light_on"]
        st.session_state.kernel.state.mode = replay_row["state_after"]["mode"]
        st.session_state.kernel.state.step_count = replay_row["state_after"]["step_count"]
        st.session_state.kernel.state.action_history = replay_row["state_after"]["action_history"]

    with center:
        st.subheader("Grid")
        st.code(render_ascii(st.session_state.kernel.snapshot()), language="text")

    with right:
        st.subheader("Route")
        route = result["route"] if result is not None else None
        if route is None and replay_row is not None:
            route = {
                "label": replay_row["predicted_label"],
                "accepted": replay_row["accepted"],
                "confidence": replay_row["confidence"],
                "top_probabilities": replay_row.get("top_probabilities", []),
            }
        if route is not None:
            st.metric("Action", route["label"])
            st.metric("Confidence", f"{route['confidence']:.3f}")
            st.metric("Accepted", str(route["accepted"]))
            probs = pd.DataFrame(route.get("top_probabilities", []))
            if not probs.empty:
                st.bar_chart(probs.set_index("label")["probability"])

    bottom_left, bottom_right = st.columns([1.3, 1.0])
    with bottom_left:
        st.subheader("Latent Space")
        if PROJECTION_PATH.exists():
            projection = pd.read_csv(PROJECTION_PATH)
            if px is not None:
                fig = px.scatter(
                    projection,
                    x="x",
                    y="y",
                    color="label",
                    hover_data=["split", "text", "accepted"],
                    height=430,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(projection[["x", "y", "label", "split", "accepted"]])

    with bottom_right:
        st.subheader("Route Log")
        if routes:
            st.dataframe(
                pd.DataFrame(routes)[
                    ["input_text", "predicted_label", "accepted", "confidence", "vector_norm"]
                ],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
