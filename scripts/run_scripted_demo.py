from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_native.app.kernel import TaskFlowKernel
from neural_native.app.vector_port import VectorActionPort
from neural_native.bridge.router import LinearProbeRouter, RouterThresholds, thresholds_from_json
from neural_native.llm.extractor import extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

SCRIPTED_EXAMPLES = [
    ("CREATE_TASK", "please create a new task"),
    ("PROMOTE_TASK", "please start the next task"),
    ("COMPLETE_ACTIVE", "please finish the current task"),
    ("ARCHIVE_COMPLETED", "please archive completed tasks"),
    ("TOGGLE_FOCUS_MODE", "please toggle focus mode"),
    ("ABSTAIN", "what is the weather tomorrow"),
]

V2_SCRIPTED_EXAMPLES = [
    ("CREATE_TASK", "please add budget review to my task list"),
    ("PROMOTE_TASK", "please make the next backlog item active for budget review"),
    ("COMPLETE_ACTIVE", "please mark the active item complete for budget review"),
    ("ARCHIVE_COMPLETED", "please archive completed tasks for budget review"),
    ("TOGGLE_FOCUS_MODE", "please toggle focus mode for budget review"),
    ("ABSTAIN", "do not finish the active task while handling budget review"),
]


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


def state_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    active = snapshot["active"]
    return {
        "backlog_count": len(snapshot["backlog"]),
        "active_task_id": active["id"] if active else None,
        "completed_count": len(snapshot["completed"]),
        "archive_count": len(snapshot["archive"]),
        "focus_mode": snapshot["focus_mode"],
        "next_id": snapshot["next_id"],
    }


def write_demo_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    model_id: str,
    feature_space: str,
    output_path: Path,
    command: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo Results",
        "",
        f"- Timestamp: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Model id: `{model_id}`",
        f"- Feature space: `{feature_space}`",
        f"- Route artifact: `{output_path.as_posix()}`",
        "",
        "This scripted run used the zero-generation path:",
        "",
        "```text",
        (
            "text -> tokenizer -> frozen LM forward pass -> pre-lm_head hook "
            "-> vector -> probe -> gate -> TaskFlowKernel"
        ),
        "```",
        "",
        "No generated text, JSON/tool-call parsing, regex, or keyword route selection is used.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Routes",
        "",
        "| Expected | Predicted | Accepted | Confidence | Margin | App status | State after |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    for row in rows:
        after = row["state_summary_after"]
        state_after = (
            f"backlog={after['backlog_count']}, active={after['active_task_id']}, "
            f"completed={after['completed_count']}, archive={after['archive_count']}, "
            f"focus={after['focus_mode']}"
        )
        lines.append(
            "| "
            f"`{row['expected_label']}` | "
            f"`{row['predicted_label']}` | "
            f"{row['accepted']} | "
            f"{row['confidence']:.4f} | "
            f"{row['margin']:.4f} | "
            f"`{row['app_result']['status']}` | "
            f"{state_after} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real-model zero-generation route example per action and write "
            "JSONL plus a Markdown summary."
        )
    )
    parser.add_argument(
        "--model-id",
        default="distilgpt2",
        help="Hugging Face causal LM id used for frozen activation extraction.",
    )
    parser.add_argument(
        "--probe",
        default="artifacts/probe.joblib",
        help="Joblib probe bundle produced by scripts/train_probe.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/example_routes.jsonl",
        help="JSONL path for example route results.",
    )
    parser.add_argument(
        "--summary-output",
        default="docs/DEMO_RESULTS.md",
        help="Markdown summary path for readable demo evidence.",
    )
    parser.add_argument("--batch-size", type=int, default=6, help="Texts per forward-pass batch.")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer truncation length.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded reproducibility seed. The demo order remains state-machine safe.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use shorter sequence length for quick sandbox verification; "
            "still routes every action."
        ),
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Override the probe bundle's recommended minimum confidence gate.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Override the probe bundle's recommended top-1/top-2 margin gate.",
    )
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=None,
        help="Override the executable-centroid distance gate.",
    )
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="Optional calibrated router thresholds JSON.",
    )
    parser.add_argument(
        "--example-set",
        choices=("classic", "v2"),
        default="classic",
        help="Scripted example set to route.",
    )
    args = parser.parse_args()

    max_length = min(args.max_length, 96) if args.fast else args.max_length

    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    tap = PreLMHeadActivationTap(model)
    kernel = TaskFlowKernel()
    router = LinearProbeRouter(args.probe, thresholds=build_thresholds(args))
    port = VectorActionPort(router=router, app=kernel)
    feature_space = str(router.metadata.get("feature_space", "unknown"))

    try:
        examples = V2_SCRIPTED_EXAMPLES if args.example_set == "v2" else SCRIPTED_EXAMPLES
        vectors = extract_vectors(
            [text for _label, text in examples],
            tokenizer,
            model,
            tap,
            batch_size=args.batch_size,
            max_length=max_length,
        )
    finally:
        tap.close()

    rows: list[dict[str, Any]] = []
    for (expected_label, text), vector in zip(examples, vectors, strict=True):
        before = state_summary(kernel.snapshot())
        result = port.ingest(vector, raw_text=text)
        after = state_summary(kernel.snapshot())
        route = result["route"]
        rows.append(
            {
                "input_text": text,
                "expected_label": expected_label,
                "predicted_label": route["label"],
                "accepted": route["accepted"],
                "confidence": route["confidence"],
                "margin": route["margin"],
                "ood_score": route["ood_score"],
                "app_result": {key: value for key, value in result.items() if key != "route"},
                "state_summary_before": before,
                "state_summary_after": after,
                "model_id": args.model_id,
                "feature_space": feature_space,
                "seed": args.seed,
                "max_length": max_length,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary_path = Path(args.summary_output)
    write_demo_summary(
        summary_path,
        rows=rows,
        model_id=args.model_id,
        feature_space=feature_space,
        output_path=output_path,
        command=(
            "python scripts/run_scripted_demo.py "
            f"--model-id {args.model_id} "
            f"--probe {args.probe} "
            f"--output {args.output} "
            f"--summary-output {args.summary_output} "
            f"--batch-size {args.batch_size} "
            f"--max-length {args.max_length} "
            f"--seed {args.seed}"
            + (" --no-4bit" if args.no_4bit else "")
            + (" --fast" if args.fast else "")
            + (f" --thresholds-json {args.thresholds_json}" if args.thresholds_json else "")
            + (f" --example-set {args.example_set}" if args.example_set != "classic" else "")
            + (
                f" --min-confidence {args.min_confidence}"
                if args.min_confidence is not None
                else ""
            )
            + (f" --min-margin {args.min_margin}" if args.min_margin is not None else "")
            + (
                f" --max-centroid-distance {args.max_centroid_distance}"
                if args.max_centroid_distance is not None
                else ""
            )
        ),
    )

    accepted = sum(1 for row in rows if row["accepted"])
    abstained = sum(1 for row in rows if not row["accepted"])
    print(
        f"routed {len(rows)} examples with {args.model_id}: "
        f"{accepted} accepted, {abstained} abstained; wrote {output_path} and {summary_path}"
    )


if __name__ == "__main__":
    main()
