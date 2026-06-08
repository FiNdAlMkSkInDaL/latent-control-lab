from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_native.bridge.train_sklearn import compute_centroids
from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

PROMPT_TEMPLATES = {
    "default": PROMPT_TEMPLATE,
    "minimal": "{text}",
    "instruction": "Classify the controller intent represented by this request: {text}",
    "controller": "Request from user: {text}\nInternal controller representation:",
    "terse": "Intent: {text}",
}


def sample_dataset(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def train_probe(X: np.ndarray, y: np.ndarray) -> Any:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    probe.fit(X, y)
    return probe


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    abstain = report.get("ABSTAIN", {})
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "abstain_precision": abstain.get("precision"),
        "abstain_recall": abstain.get("recall"),
        "classification_report": report,
    }


def extract_for_template(
    df: pd.DataFrame,
    *,
    prompt_template: str,
    tokenizer: Any,
    model: Any,
    tap: Any,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    return extract_vectors(
        df["text"].astype(str).tolist(),
        tokenizer,
        model,
        tap,
        batch_size=batch_size,
        max_length=max_length,
        prompt_template=prompt_template,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a prompt-augmented latent probe over multiple prompt templates."
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Hugging Face causal LM id.")
    parser.add_argument("--dataset", default="data/intent_dataset_v2.csv", help="V2 dataset CSV.")
    parser.add_argument("--hard-dataset", default="data/hard_eval.csv", help="Hard eval CSV.")
    parser.add_argument(
        "--output-probe",
        default="artifacts/probe_prompt_augmented.joblib",
        help="Ignored joblib probe bundle path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/prompt_augmented_metrics.json",
        help="JSON metrics path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/prompt_augmented_metrics.csv",
        help="CSV metrics path.",
    )
    parser.add_argument(
        "--output-hard-eval",
        default="artifacts/prompt_augmented_hard_eval.csv",
        help="Hard-eval prediction CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per forward batch.")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer max length.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional v2 dataset sample.")
    parser.add_argument("--hard-sample-size", type=int, default=None, help="Optional hard sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller deterministic sample.")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    args = parser.parse_args()

    sample_size = args.sample_size
    hard_sample_size = args.hard_sample_size
    if args.fast:
        sample_size = sample_size or 180
        hard_sample_size = hard_sample_size or 120

    df = sample_dataset(pd.read_csv(args.dataset), sample_size, args.seed)
    hard_df = sample_dataset(pd.read_csv(args.hard_dataset), hard_sample_size, args.seed)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Dataset sample must include train and test rows")

    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    tap = PreLMHeadActivationTap(model)
    train_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    try:
        for prompt_template in PROMPT_TEMPLATES.values():
            X_train = extract_for_template(
                train_df,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                model=model,
                tap=tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            train_blocks.append(X_train)
            y_blocks.append(train_df["label"].astype(str).to_numpy())

        X_aug = np.concatenate(train_blocks, axis=0).astype(np.float32)
        y_aug = np.concatenate(y_blocks, axis=0).astype(str)
        probe = train_probe(X_aug, y_aug)
        labels = list(probe.classes_)

        rows: list[dict[str, Any]] = []
        hard_prediction_frames: list[pd.DataFrame] = []
        for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
            X_test = extract_for_template(
                test_df,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                model=model,
                tap=tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            y_test = test_df["label"].astype(str).to_numpy()
            pred_test = probe.predict(X_test).astype(str)
            test_metrics = evaluate_predictions(y_test, pred_test, labels)
            rows.append(
                {
                    "prompt_name": prompt_name,
                    "dataset": "v2_test",
                    "n_rows": int(len(test_df)),
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "abstain_precision": test_metrics["abstain_precision"],
                    "abstain_recall": test_metrics["abstain_recall"],
                }
            )

            X_hard = extract_for_template(
                hard_df,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                model=model,
                tap=tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            y_hard = hard_df["label"].astype(str).to_numpy()
            pred_hard = probe.predict(X_hard).astype(str)
            hard_metrics = evaluate_predictions(y_hard, pred_hard, labels)
            rows.append(
                {
                    "prompt_name": prompt_name,
                    "dataset": "hard_eval",
                    "n_rows": int(len(hard_df)),
                    "accuracy": hard_metrics["accuracy"],
                    "macro_f1": hard_metrics["macro_f1"],
                    "abstain_precision": hard_metrics["abstain_precision"],
                    "abstain_recall": hard_metrics["abstain_recall"],
                }
            )
            hard_predictions = hard_df.copy()
            hard_predictions["prompt_name"] = prompt_name
            hard_predictions["predicted_label"] = pred_hard
            hard_prediction_frames.append(hard_predictions)
    finally:
        tap.close()

    bundle = {
        "probe": probe,
        "classes": probe.classes_,
        "centroids": compute_centroids(X_aug, y_aug),
        "model_id": args.model_id,
        "prompt_template": "prompt_augmented_union",
        "prompt_templates": PROMPT_TEMPLATES,
        "feature_space": "prompt_augmented_pre_lm_head_last_token",
        "metrics": {
            "labels": labels,
            "recommended_thresholds": {
                "min_confidence": 0.0,
                "min_margin": 0.0,
                "max_centroid_distance": None,
            },
            "prompt_augmented_rows": rows,
            "train_rows": int(len(train_df)),
            "augmented_train_rows": int(len(y_aug)),
        },
    }
    output_probe = Path(args.output_probe)
    output_probe.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_probe)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "dataset": args.dataset,
        "hard_dataset": args.hard_dataset,
        "sample_size": sample_size,
        "hard_sample_size": hard_sample_size,
        "prompt_templates": PROMPT_TEMPLATES,
        "probe": args.output_probe,
        "runs": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    output_hard = Path(args.output_hard_eval)
    output_hard.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(hard_prediction_frames, ignore_index=True).to_csv(output_hard, index=False)

    print(pd.DataFrame(rows))
    print(f"wrote {output_probe}")
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_hard}")


if __name__ == "__main__":
    main()
