from __future__ import annotations

import json
from typing import Any

import typer
from rich import print

from neural_native.bridge.router import RouterThresholds
from neural_native.llm.extractor import extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm
from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_ascii
from neural_native.vectorbot.router import VectorBotLinearProbeRouter
from neural_native.vectorbot.vector_port import VectorBotVectorPort

app = typer.Typer(help="Neural-native zero-generation action router demo.")


def _build_thresholds(
    min_confidence: float | None,
    min_margin: float | None,
    max_centroid_distance: float | None,
) -> RouterThresholds | None:
    threshold_overridden = any(
        value is not None for value in (min_confidence, min_margin, max_centroid_distance)
    )
    if not threshold_overridden:
        return None

    defaults = RouterThresholds()
    return RouterThresholds(
        min_confidence=min_confidence
        if min_confidence is not None
        else defaults.min_confidence,
        min_margin=min_margin if min_margin is not None else defaults.min_margin,
        max_centroid_distance=max_centroid_distance,
    )


def _load_runtime(
    model_id: str,
    probe_path: str,
    min_confidence: float | None,
    min_margin: float | None,
    max_centroid_distance: float | None,
    *,
    use_4bit: bool,
) -> tuple[Any, Any, PreLMHeadActivationTap, VectorBotVectorPort]:
    tokenizer, model = load_causal_lm(model_id=model_id, use_4bit=use_4bit)
    tap = PreLMHeadActivationTap(model)

    kernel = VectorBotKernel()
    thresholds = _build_thresholds(min_confidence, min_margin, max_centroid_distance)
    router = VectorBotLinearProbeRouter(bundle_path=probe_path, thresholds=thresholds)
    port = VectorBotVectorPort(router=router, app=kernel)
    return tokenizer, model, tap, port


@app.command()
def run(
    model_id: str = typer.Option(
        "distilgpt2",
        help="Hugging Face causal LM id used for frozen activation extraction.",
    ),
    probe_path: str = typer.Option(
        "artifacts/vectorbot_probe_distilgpt2.joblib",
        help="Serialized VectorBot probe bundle from scripts/train_vectorbot_probe.py.",
    ),
    min_confidence: float | None = typer.Option(
        None,
        help="Override the probe bundle's recommended minimum confidence gate.",
    ),
    min_margin: float | None = typer.Option(
        None,
        help="Override the probe bundle's recommended top-1/top-2 margin gate.",
    ),
    max_centroid_distance: float | None = typer.Option(
        None,
        help="Override the executable-centroid distance gate.",
    ),
    batch_size: int = typer.Option(1, help="Texts per activation extraction batch."),
    no_4bit: bool = typer.Option(
        False,
        "--no-4bit",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    ),
) -> None:
    """Interactive text -> activation -> probe -> app-state demo."""

    tokenizer, model, tap, port = _load_runtime(
        model_id,
        probe_path,
        min_confidence,
        min_margin,
        max_centroid_distance,
        use_4bit=not no_4bit,
    )

    print("[bold]VectorBot latent-control demo[/bold]")
    print("Type 'exit' or 'quit' to stop. Core route uses no generated tokens.\n")
    print(render_ascii(port.app.snapshot()))

    try:
        while True:
            text = typer.prompt("user")
            if text.strip().lower() in {"exit", "quit"}:
                break

            z = extract_vectors(
                texts=[text],
                tokenizer=tokenizer,
                model=model,
                tap=tap,
                batch_size=batch_size,
            )[0]
            result = port.ingest(z=z, raw_text=text)
            print(result)
            print(render_ascii(port.app.snapshot()))
    finally:
        tap.close()


@app.command()
def route(
    text: str = typer.Argument(..., help="Natural-language user request to route."),
    model_id: str = typer.Option(
        "distilgpt2",
        help="Hugging Face causal LM id used for frozen activation extraction.",
    ),
    probe_path: str = typer.Option(
        "artifacts/vectorbot_probe_distilgpt2.joblib",
        help="Serialized VectorBot probe bundle from scripts/train_vectorbot_probe.py.",
    ),
    min_confidence: float | None = typer.Option(
        None,
        help="Override the probe bundle's recommended minimum confidence gate.",
    ),
    min_margin: float | None = typer.Option(
        None,
        help="Override the probe bundle's recommended top-1/top-2 margin gate.",
    ),
    max_centroid_distance: float | None = typer.Option(
        None,
        help="Override the executable-centroid distance gate.",
    ),
    no_4bit: bool = typer.Option(
        False,
        "--no-4bit",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    ),
    json_output: bool = typer.Option(False, help="Print machine-readable JSON."),
) -> None:
    """Route one request through text -> hidden state -> probe -> app action."""

    tokenizer, model, tap, port = _load_runtime(
        model_id,
        probe_path,
        min_confidence,
        min_margin,
        max_centroid_distance,
        use_4bit=not no_4bit,
    )
    try:
        z = extract_vectors(
            texts=[text],
            tokenizer=tokenizer,
            model=model,
            tap=tap,
            batch_size=1,
        )[0]
        result = port.ingest(z=z, raw_text=text)
    finally:
        tap.close()

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    app()
