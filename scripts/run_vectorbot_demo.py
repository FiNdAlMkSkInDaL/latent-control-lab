from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neural_native.bridge.router import RouterThresholds
from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm
from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_ascii
from neural_native.vectorbot.router import VectorBotLinearProbeRouter, thresholds_from_json
from neural_native.vectorbot.state import LABEL_TO_ACTION, VectorBotAction
from neural_native.vectorbot.vector_port import VectorBotRouteDecision, VectorBotVectorPort

try:
    from scripts.generate_vectorbot_dataset import LABELS
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_vectorbot_dataset import LABELS

SCRIPTED_EXAMPLES = [
    ("MOVE_UP", "go north"),
    ("MOVE_DOWN", "take one step south"),
    ("MOVE_LEFT", "slide left"),
    ("MOVE_RIGHT", "move the bot right"),
    ("TOGGLE_LIGHT", "toggle the lamp"),
    ("RESET", "reset the board"),
    ("ABSTAIN", "what is the weather tomorrow?"),
    ("ABSTAIN", "delete all files on this laptop"),
    ("ABSTAIN", "move up and then right"),
    ("ABSTAIN", "paint the robot purple"),
]


class FakeVectorBotRouter:
    """Vector-only smoke router for fallback demos; it never reads raw text."""

    def __init__(self) -> None:
        self.classes = list(LABELS)
        self.metadata = {
            "model_id": "fake-vectorbot-plumbing",
            "feature_space": "fake_scripted_one_hot",
        }

    def predict(self, z: np.ndarray) -> VectorBotRouteDecision:
        probs = np.asarray(z, dtype=np.float32)
        probs = probs / probs.sum()
        order = np.argsort(probs)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        label = self.classes[top1]
        action = LABEL_TO_ACTION.get(label, VectorBotAction.ABSTAIN)
        accepted = label != "ABSTAIN"
        return VectorBotRouteDecision(
            action=action,
            label=label,
            confidence=float(probs[top1]),
            margin=float(probs[top1] - probs[top2]),
            ood_score=0.0,
            accepted=accepted,
            top_probabilities=[
                {"label": self.classes[index], "probability": float(probs[index])}
                for index in order[:3]
            ],
        )


def _fake_vectors(examples: list[tuple[str, str]]) -> np.ndarray:
    vectors = []
    for label, _text in examples:
        values = np.full(len(LABELS), 0.01, dtype=np.float32)
        values[LABELS.index(label)] = 0.94
        vectors.append(values)
    return np.stack(vectors)


def build_thresholds(args: argparse.Namespace) -> RouterThresholds | None:
    if args.thresholds_json:
        return thresholds_from_json(args.thresholds_json)
    overridden = any(
        value is not None
        for value in (args.min_confidence, args.min_margin, args.max_centroid_distance)
    )
    if not overridden:
        return None
    defaults = RouterThresholds()
    return RouterThresholds(
        min_confidence=args.min_confidence
        if args.min_confidence is not None
        else defaults.min_confidence,
        min_margin=args.min_margin if args.min_margin is not None else defaults.min_margin,
        max_centroid_distance=args.max_centroid_distance,
    )


def _route_one(
    *,
    port: VectorBotVectorPort,
    kernel: VectorBotKernel,
    vector: np.ndarray,
    text: str,
    expected_label: str | None,
    model_id: str,
    feature_space: str,
    fake_run: bool,
) -> dict[str, Any]:
    before = kernel.snapshot()
    result = port.ingest(vector, raw_text=text)
    after = kernel.snapshot()
    route = result["route"]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_text": text,
        "expected_label": expected_label,
        "predicted_label": route["label"],
        "accepted": route["accepted"],
        "confidence": route["confidence"],
        "margin": route["margin"],
        "ood_score": route["ood_score"],
        "top_probabilities": route["top_probabilities"],
        "vector_norm": float(np.linalg.norm(vector)),
        "app_status": result["status"],
        "app_action": result["action"],
        "state_before": before,
        "state_after": after,
        "diff": result["diff"],
        "model_id": model_id,
        "feature_space": feature_space,
        "fake_run": fake_run,
    }
    print()
    print("before:")
    print(render_ascii(before))
    print("after:")
    print(render_ascii(after))
    print(f"input: {text}")
    print(
        "predicted: "
        f"{row['predicted_label']} | accepted={row['accepted']} | "
        f"confidence={row['confidence']:.3f} | margin={row['margin']:.3f} | "
        f"norm={row['vector_norm']:.3f}"
    )
    top = ", ".join(
        f"{item['label']}={item['probability']:.3f}" for item in row["top_probabilities"]
    )
    print(f"top-3: {top}")
    print(f"diff: {json.dumps(row['diff'], sort_keys=True)}")
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_transcript(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VectorBot Demo Transcript",
        "",
        "This transcript was generated from the zero-generation route log.",
        "",
        "| Command | Predicted | Accepted | Confidence | Vector norm | Diff |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        diff = json.dumps(row["diff"], sort_keys=True)
        lines.append(
            f"| `{row['input_text']}` | `{row['predicted_label']}` | "
            f"{row['accepted']} | {row['confidence']:.3f} | "
            f"{row['vector_norm']:.3f} | `{diff}` |"
        )
    lines.extend(
        [
            "",
            "The route is text -> tokenizer -> frozen LM forward pass -> hidden vector "
            "-> probe -> gate -> VectorBot enum action.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_fake(args: argparse.Namespace) -> None:
    kernel = VectorBotKernel()
    router = FakeVectorBotRouter()
    port = VectorBotVectorPort(router=router, app=kernel)
    examples = SCRIPTED_EXAMPLES if args.scripted else [(None, args.text or "go north")]
    normalized_examples = [
        (label if label is not None else "MOVE_UP", text) for label, text in examples
    ]
    vectors = _fake_vectors(normalized_examples)
    rows = [
        _route_one(
            port=port,
            kernel=kernel,
            vector=vector,
            text=text,
            expected_label=label,
            model_id="fake-vectorbot-plumbing",
            feature_space="fake_scripted_one_hot",
            fake_run=True,
        )
        for (label, text), vector in zip(normalized_examples, vectors, strict=True)
    ]
    _write_rows(Path(args.output), rows)
    if args.transcript_output:
        _write_transcript(Path(args.transcript_output), rows)
    print(f"\nwrote {args.output}")


def _run_replay(args: argparse.Namespace) -> None:
    path = Path(args.replay_path or args.output)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        print()
        print(render_ascii(row["state_after"]))
        print(
            f"input: {row['input_text']} -> {row['predicted_label']} "
            f"accepted={row['accepted']} confidence={row['confidence']:.3f}"
        )
    print(f"\nreplayed {len(rows)} routes from {path}")


def _run_live(args: argparse.Namespace) -> None:
    thresholds = build_thresholds(args)
    router = VectorBotLinearProbeRouter(args.probe, thresholds=thresholds)
    kernel = VectorBotKernel()
    port = VectorBotVectorPort(router=router, app=kernel)
    tokenizer, model = load_causal_lm(args.model_id, use_4bit=False)
    tap = PreLMHeadActivationTap(model)
    feature_space = str(router.metadata.get("feature_space", "pre_lm_head_last_token"))

    rows: list[dict[str, Any]] = []
    try:
        if args.scripted:
            texts = [text for _label, text in SCRIPTED_EXAMPLES]
            vectors = extract_vectors(
                texts,
                tokenizer,
                model,
                tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
                prompt_template=args.prompt_template,
            )
            for (expected_label, text), vector in zip(SCRIPTED_EXAMPLES, vectors, strict=True):
                rows.append(
                    _route_one(
                        port=port,
                        kernel=kernel,
                        vector=vector,
                        text=text,
                        expected_label=expected_label,
                        model_id=args.model_id,
                        feature_space=feature_space,
                        fake_run=False,
                    )
                )
        else:
            print(render_ascii(kernel.snapshot()))
            print("Type exit or quit to stop. The route uses no generated tokens.")
            while True:
                text = input("user> ")
                if text.strip() in {"exit", "quit"}:
                    break
                vector = extract_vectors(
                    [text],
                    tokenizer,
                    model,
                    tap,
                    batch_size=1,
                    max_length=args.max_length,
                    prompt_template=args.prompt_template,
                )[0]
                rows.append(
                    _route_one(
                        port=port,
                        kernel=kernel,
                        vector=vector,
                        text=text,
                        expected_label=None,
                        model_id=args.model_id,
                        feature_space=feature_space,
                        fake_run=False,
                    )
                )
    finally:
        tap.close()

    _write_rows(Path(args.output), rows)
    if args.transcript_output:
        _write_transcript(Path(args.transcript_output), rows)
    print(f"\nwrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the VectorBot terminal demo through vector-routed actions."
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Frozen causal LM id.")
    parser.add_argument(
        "--probe",
        default="artifacts/vectorbot_probe_distilgpt2.joblib",
        help="VectorBot probe bundle.",
    )
    parser.add_argument(
        "--output",
        "--routes-output",
        dest="output",
        default="artifacts/vectorbot_routes.jsonl",
        help="JSONL route log path.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per batch.")
    parser.add_argument("--max-length", type=int, default=96, help="Tokenizer max length.")
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Prompt template containing {text}.",
    )
    parser.add_argument("--scripted", action="store_true", help="Run scripted examples.")
    parser.add_argument("--text", default=None, help="Single fake-mode text when not scripted.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use deterministic fake vectors for plumbing-only scripted demos.",
    )
    parser.add_argument("--replay", action="store_true", help="Replay an existing route log.")
    parser.add_argument("--replay-path", default=None, help="JSONL route log to replay.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Gate override.")
    parser.add_argument("--min-margin", type=float, default=None, help="Gate override.")
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=None,
        help="Centroid gate override.",
    )
    parser.add_argument("--thresholds-json", default=None, help="Router thresholds JSON.")
    parser.add_argument(
        "--transcript-output",
        default=None,
        help="Optional Markdown transcript path for scripted or fake runs.",
    )
    args = parser.parse_args()

    if args.replay:
        _run_replay(args)
    elif args.fake:
        _run_fake(args)
    else:
        _run_live(args)


if __name__ == "__main__":
    main()
