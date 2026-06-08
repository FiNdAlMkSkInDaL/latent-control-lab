from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from neural_native.bridge.router import LinearProbeRouter
from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

try:
    from scripts.evaluate_hard_eval import (
        classification_metrics,
        effective_route_labels,
        sample_frame,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from evaluate_hard_eval import classification_metrics, effective_route_labels, sample_frame

PROMPT_TEMPLATES = {
    "router_prompt": PROMPT_TEMPLATE,
    "minimal": "{text}",
    "instruction": "Classify the controller intent represented by this request: {text}",
    "noisy": "Request from user: {text}\nInternal controller representation:",
}


def _normal_eval_frame(path: str, sample_size: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" in df:
        test = df[df["split"] == "test"].copy()
        if not test.empty:
            df = test
    df = df.copy()
    df["subset"] = "synthetic_test"
    df["expected_behavior"] = df["label"].map(
        lambda label: "reject" if label == "ABSTAIN" else "execute"
    )
    df["notes"] = "Synthetic held-out test row."
    return sample_frame(df, sample_size, seed)


def _hard_eval_frame(path: str, sample_size: int | None, seed: int) -> pd.DataFrame:
    return sample_frame(pd.read_csv(path), sample_size, seed)


def evaluate_prompt(
    df: pd.DataFrame,
    *,
    prompt_name: str,
    prompt_template: str,
    dataset_name: str,
    tokenizer: Any,
    model: Any,
    tap: Any,
    router: LinearProbeRouter,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    X = extract_vectors(
        df["text"].astype(str).tolist(),
        tokenizer,
        model,
        tap,
        batch_size=batch_size,
        max_length=max_length,
        prompt_template=prompt_template,
    )
    routes = effective_route_labels(router, X)
    labels = sorted(set(router.classes) | set(df["label"].astype(str)) | {"ABSTAIN"})
    metrics = classification_metrics(
        df["label"].astype(str).to_numpy(),
        routes["effective_label"].astype(str).to_numpy(),
        df["subset"].astype(str).to_numpy(),
        labels=labels,
    )
    return {
        "prompt_name": prompt_name,
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "accuracy": metrics["overall_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "abstain_precision": metrics["abstain_precision"],
        "abstain_recall": metrics["abstain_recall"],
        "per_subset_accuracy": metrics["per_subset_accuracy"],
    }


def add_prompt_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default_accuracy = {
        row["dataset"]: row["accuracy"]
        for row in rows
        if row["prompt_name"] == "router_prompt"
    }
    default_macro_f1 = {
        row["dataset"]: row["macro_f1"]
        for row in rows
        if row["prompt_name"] == "router_prompt"
    }
    for row in rows:
        baseline_accuracy = default_accuracy.get(row["dataset"])
        baseline_macro_f1 = default_macro_f1.get(row["dataset"])
        row["accuracy_drop_from_router_prompt"] = (
            float(baseline_accuracy - row["accuracy"]) if baseline_accuracy is not None else None
        )
        row["macro_f1_drop_from_router_prompt"] = (
            float(baseline_macro_f1 - row["macro_f1"]) if baseline_macro_f1 is not None else None
        )
        row["probe_generalizes_from_router_prompt"] = (
            row["accuracy_drop_from_router_prompt"] is not None
            and row["accuracy_drop_from_router_prompt"] <= 0.10
            and row["macro_f1_drop_from_router_prompt"] is not None
            and row["macro_f1_drop_from_router_prompt"] <= 0.10
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt-template robustness for the trained latent probe."
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Hugging Face causal LM id.")
    parser.add_argument("--probe", default="artifacts/probe.joblib", help="Probe bundle path.")
    parser.add_argument(
        "--normal-dataset",
        default="data/intent_dataset.csv",
        help="Synthetic dataset with split metadata.",
    )
    parser.add_argument("--hard-dataset", default="data/hard_eval.csv", help="Hard eval CSV.")
    parser.add_argument(
        "--output-json",
        default="artifacts/prompt_robustness.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/prompt_robustness.csv",
        help="CSV output path.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per forward batch.")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer max length.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional per-dataset sample size for fast verification.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    args = parser.parse_args()

    normal_df = _normal_eval_frame(args.normal_dataset, args.sample_size, args.seed)
    hard_df = _hard_eval_frame(args.hard_dataset, args.sample_size, args.seed)
    router = LinearProbeRouter(args.probe)
    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    tap = PreLMHeadActivationTap(model)

    rows: list[dict[str, Any]] = []
    try:
        for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
            rows.append(
                evaluate_prompt(
                    normal_df,
                    prompt_name=prompt_name,
                    prompt_template=prompt_template,
                    dataset_name="synthetic_test",
                    tokenizer=tokenizer,
                    model=model,
                    tap=tap,
                    router=router,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            )
            rows.append(
                evaluate_prompt(
                    hard_df,
                    prompt_name=prompt_name,
                    prompt_template=prompt_template,
                    dataset_name="hard_eval",
                    tokenizer=tokenizer,
                    model=model,
                    tap=tap,
                    router=router,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            )
    finally:
        tap.close()

    rows = add_prompt_deltas(rows)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "probe": args.probe,
        "mode": "pretrained_router_prompt_probe_evaluated_on_prompt_variants",
        "sample_size": args.sample_size,
        "prompt_templates": PROMPT_TEMPLATES,
        "runs": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(pd.DataFrame(rows)[["prompt_name", "dataset", "accuracy", "macro_f1"]])
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
