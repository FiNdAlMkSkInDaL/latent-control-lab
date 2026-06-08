from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from neural_native.bridge.router import LinearProbeRouter, thresholds_from_json
from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

EXECUTABLE_LABELS = {
    "ARCHIVE_COMPLETED",
    "COMPLETE_ACTIVE",
    "CREATE_TASK",
    "PROMOTE_TASK",
    "TOGGLE_FOCUS_MODE",
}


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def effective_route_labels(router: LinearProbeRouter, X: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for z in X:
        route = router.predict(z)
        rows.append(
            {
                "predicted_label": route.label,
                "effective_label": route.label if route.accepted else "ABSTAIN",
                "accepted": bool(route.accepted),
                "confidence": float(route.confidence),
                "margin": float(route.margin),
                "ood_score": float(route.ood_score),
                "route_action": route.action.name,
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subsets: np.ndarray,
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
    executable_mask = np.asarray([label in EXECUTABLE_LABELS for label in y_true])
    subset_accuracy = {}
    for subset in sorted(set(map(str, subsets))):
        mask = subsets == subset
        subset_accuracy[str(subset)] = float(accuracy_score(y_true[mask], y_pred[mask]))

    def rejection_rate(subset: str) -> float | None:
        mask = subsets == subset
        if not mask.any():
            return None
        return float(np.mean(y_pred[mask] == "ABSTAIN"))

    abstain = report.get("ABSTAIN", {})
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "executable_only_accuracy": float(
            accuracy_score(y_true[executable_mask], y_pred[executable_mask])
        )
        if executable_mask.any()
        else None,
        "abstain_precision": abstain.get("precision"),
        "abstain_recall": abstain.get("recall"),
        "per_class_precision_recall_f1": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
            if label in report
        },
        "per_subset_accuracy": subset_accuracy,
        "negation_rejection_rate": rejection_rate("negation"),
        "compound_rejection_rate": rejection_rate("compound"),
        "unsafe_rejection_rate": rejection_rate("unsafe"),
        "gibberish_rejection_rate": rejection_rate("gibberish"),
        "near_miss_rejection_rate": rejection_rate("near_miss"),
        "classification_report": report,
    }


def write_confusion_matrix(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output)


def save_feature_npz(
    path: str | Path,
    X: np.ndarray,
    df: pd.DataFrame,
    *,
    model_id: str,
    prompt_template: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=X.astype(np.float32),
        y=df["label"].astype(str).to_numpy(),
        text=df["text"].astype(str).to_numpy(),
        subset=df["subset"].astype(str).to_numpy(),
        expected_behavior=df["expected_behavior"].astype(str).to_numpy(),
        notes=df["notes"].astype(str).to_numpy(),
        split=np.array(["hard_eval"] * len(df)),
        model_id=np.array([model_id]),
        prompt_template=np.array([prompt_template]),
        feature_space=np.array(["pre_lm_head_last_token"]),
    )


def threshold_summary(router: LinearProbeRouter) -> dict[str, Any]:
    payload = asdict(router.thresholds)
    for key in ("mahalanobis_mean", "mahalanobis_precision"):
        value = payload.get(key)
        if isinstance(value, np.ndarray):
            payload[key] = list(value.shape)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the latent router on the curated hard evaluation set."
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Hugging Face causal LM id.")
    parser.add_argument("--probe", default="artifacts/probe.joblib", help="Probe bundle path.")
    parser.add_argument("--dataset", default="data/hard_eval.csv", help="Hard eval CSV path.")
    parser.add_argument(
        "--output-metrics",
        default="artifacts/hard_eval_metrics.json",
        help="JSON metrics output path.",
    )
    parser.add_argument(
        "--output-predictions",
        default="artifacts/hard_eval_predictions.csv",
        help="CSV prediction output path.",
    )
    parser.add_argument(
        "--output-confusion",
        default="artifacts/hard_eval_confusion_matrix.csv",
        help="CSV confusion matrix output path.",
    )
    parser.add_argument(
        "--features-output",
        default=None,
        help="Optional ignored NPZ feature artifact for follow-on evaluation scripts.",
    )
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="Optional calibrated router thresholds JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per forward batch.")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer max length.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional fast sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    args = parser.parse_args()

    df = sample_frame(pd.read_csv(args.dataset), args.sample_size, args.seed)
    thresholds = thresholds_from_json(args.thresholds_json) if args.thresholds_json else None
    router = LinearProbeRouter(args.probe, thresholds=thresholds)

    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    tap = PreLMHeadActivationTap(model)
    try:
        X = extract_vectors(
            df["text"].astype(str).tolist(),
            tokenizer,
            model,
            tap,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    finally:
        tap.close()

    routes = effective_route_labels(router, X)
    predictions = pd.concat([df.reset_index(drop=True), routes], axis=1)
    y_true = predictions["label"].astype(str).to_numpy()
    y_pred = predictions["effective_label"].astype(str).to_numpy()
    labels = sorted(set(router.classes) | set(y_true) | {"ABSTAIN"})
    metrics = classification_metrics(
        y_true,
        y_pred,
        predictions["subset"].astype(str).to_numpy(),
        labels=labels,
    )
    metrics.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "probe_model_id": router.metadata.get("model_id"),
            "feature_space": router.metadata.get("feature_space"),
            "prompt_template": PROMPT_TEMPLATE,
            "dataset": args.dataset,
            "n_rows": int(len(df)),
            "sample_size": args.sample_size,
            "thresholds": threshold_summary(router),
        }
    )

    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    output_predictions = Path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_predictions, index=False)
    write_confusion_matrix(args.output_confusion, y_true, y_pred, labels)

    if args.features_output:
        save_feature_npz(
            args.features_output,
            X,
            df,
            model_id=args.model_id,
            prompt_template=PROMPT_TEMPLATE,
        )

    print(json.dumps({k: metrics[k] for k in ("overall_accuracy", "macro_f1")}, indent=2))
    print(f"wrote {output_metrics}")
    print(f"wrote {output_predictions}")
    print(f"wrote {args.output_confusion}")


if __name__ == "__main__":
    main()
