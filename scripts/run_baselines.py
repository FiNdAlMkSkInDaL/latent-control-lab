from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
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
    }


def make_row(
    baseline: str,
    dataset: str,
    metrics: dict[str, Any],
    *,
    notes: str,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "dataset": dataset,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "abstain_precision": metrics.get("abstain_precision"),
        "abstain_recall": metrics.get("abstain_recall"),
        "notes": notes,
    }


def text_baseline_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    *,
    labels: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(train_df["text"].astype(str), train_df["label"].astype(str))
    for dataset, df in (("synthetic_test", test_df), ("hard_eval", hard_df)):
        pred = model.predict(df["text"].astype(str))
        metrics = evaluate_predictions(df["label"].astype(str).to_numpy(), pred, labels=labels)
        rows.append(
            make_row(
                "tfidf_logreg_text",
                dataset,
                metrics,
                notes="Surface-text baseline; not used by the zero-generation app route.",
            )
        )

    for strategy in ("stratified", "most_frequent"):
        dummy = DummyClassifier(strategy=strategy, random_state=seed)
        dummy.fit(train_df[["text"]], train_df["label"].astype(str))
        baseline_name = "random_stratified" if strategy == "stratified" else "majority_class"
        for dataset, df in (("synthetic_test", test_df), ("hard_eval", hard_df)):
            pred = dummy.predict(df[["text"]])
            metrics = evaluate_predictions(df["label"].astype(str).to_numpy(), pred, labels=labels)
            rows.append(
                make_row(
                    baseline_name,
                    dataset,
                    metrics,
                    notes="Non-semantic reference baseline.",
                )
            )
    return rows


def latent_rows(metrics_path: str, hard_metrics_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if Path(metrics_path).exists():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        rows.append(
            make_row(
                "latent_probe_router",
                "synthetic_test",
                {
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "abstain_precision": metrics.get("abstain_precision"),
                    "abstain_recall": metrics.get("abstain_recall"),
                },
                notes="Frozen-LM hidden state plus linear probe.",
            )
        )
    if Path(hard_metrics_path).exists():
        metrics = json.loads(Path(hard_metrics_path).read_text(encoding="utf-8"))
        rows.append(
            make_row(
                "latent_probe_router",
                "hard_eval",
                {
                    "accuracy": metrics.get("overall_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "abstain_precision": metrics.get("abstain_precision"),
                    "abstain_recall": metrics.get("abstain_recall"),
                },
                notes="Frozen-LM hidden state plus linear probe and gates.",
            )
        )
    return rows


def extra_metric_rows(
    *,
    latent_v2_metrics: str | None,
    hard_v2_metrics: str | None,
    prompt_augmented_metrics: str | None,
    calibrated_metrics: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if latent_v2_metrics and Path(latent_v2_metrics).exists():
        metrics = json.loads(Path(latent_v2_metrics).read_text(encoding="utf-8"))
        rows.append(
            make_row(
                "latent_probe_router_v2",
                "v2_test",
                {
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "abstain_precision": metrics.get("abstain_precision"),
                    "abstain_recall": metrics.get("abstain_recall"),
                },
                notes="V2 frozen-LM hidden state plus linear probe.",
            )
        )
    if hard_v2_metrics and Path(hard_v2_metrics).exists():
        metrics = json.loads(Path(hard_v2_metrics).read_text(encoding="utf-8"))
        rows.append(
            make_row(
                "latent_probe_router_v2",
                "hard_eval",
                {
                    "accuracy": metrics.get("overall_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "abstain_precision": metrics.get("abstain_precision"),
                    "abstain_recall": metrics.get("abstain_recall"),
                },
                notes="V2 latent probe with default router thresholds.",
            )
        )
    if prompt_augmented_metrics and Path(prompt_augmented_metrics).exists():
        metrics = json.loads(Path(prompt_augmented_metrics).read_text(encoding="utf-8"))
        for run in metrics.get("runs", []):
            if run.get("prompt_name") == "default":
                rows.append(
                    make_row(
                        "prompt_augmented_latent_v2",
                        str(run.get("dataset")),
                        {
                            "accuracy": run.get("accuracy"),
                            "macro_f1": run.get("macro_f1"),
                            "abstain_precision": run.get("abstain_precision"),
                            "abstain_recall": run.get("abstain_recall"),
                        },
                        notes="Single latent probe trained across multiple prompt templates.",
                    )
                )
    if calibrated_metrics and Path(calibrated_metrics).exists():
        metrics = json.loads(Path(calibrated_metrics).read_text(encoding="utf-8"))
        rows.append(
            make_row(
                "calibrated_latent_v2",
                "hard_eval",
                {
                    "accuracy": metrics.get("overall_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "abstain_precision": metrics.get("abstain_precision"),
                    "abstain_recall": metrics.get("abstain_recall"),
                },
                notes="V2 latent probe with validation-selected calibrated OOD threshold.",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run text, random, and majority baselines for the intent datasets."
    )
    parser.add_argument("--dataset", default="data/intent_dataset.csv", help="Synthetic CSV.")
    parser.add_argument("--hard-dataset", default="data/hard_eval.csv", help="Hard eval CSV.")
    parser.add_argument(
        "--latent-metrics",
        default="artifacts/metrics.json",
        help="Latent synthetic metrics JSON.",
    )
    parser.add_argument(
        "--hard-metrics",
        default="artifacts/hard_eval_metrics.json",
        help="Latent hard-eval metrics JSON.",
    )
    parser.add_argument("--latent-v2-metrics", default=None, help="Optional v2 test metrics JSON.")
    parser.add_argument("--hard-v2-metrics", default=None, help="Optional v2 hard metrics JSON.")
    parser.add_argument(
        "--prompt-augmented-metrics",
        default=None,
        help="Optional prompt-augmented metrics JSON.",
    )
    parser.add_argument(
        "--calibrated-metrics",
        default=None,
        help="Optional calibrated hard-eval metrics JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/baseline_metrics.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/baseline_metrics.csv",
        help="CSV output path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed.")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    hard_df = pd.read_csv(args.hard_dataset)
    labels = sorted(set(df["label"].astype(str)) | set(hard_df["label"].astype(str)))

    rows = latent_rows(args.latent_metrics, args.hard_metrics)
    rows.extend(
        extra_metric_rows(
            latent_v2_metrics=args.latent_v2_metrics,
            hard_v2_metrics=args.hard_v2_metrics,
            prompt_augmented_metrics=args.prompt_augmented_metrics,
            calibrated_metrics=args.calibrated_metrics,
        )
    )
    rows.extend(text_baseline_rows(train_df, test_df, hard_df, labels=labels, seed=args.seed))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "hard_dataset": args.hard_dataset,
        "rows": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(pd.DataFrame(rows))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
