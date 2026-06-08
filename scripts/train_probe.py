from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neural_native.bridge.train_sklearn import train_probe


def build_metrics_payload(bundle: dict, *, fallback_run: bool) -> dict:
    metrics = bundle["metrics"]
    test_metrics = metrics["splits"]["test"]
    report = test_metrics["classification_report"]
    split_counts = metrics["split_counts"]
    labels = metrics["labels"]
    abstain = report.get("ABSTAIN", {})

    return {
        "model_id": bundle["model_id"],
        "prompt_template": bundle["prompt_template"],
        "feature_space": bundle["feature_space"],
        "dataset_size": int(sum(metrics["class_counts"].values())),
        "train_size": int(split_counts.get("train", 0)),
        "validation_size": int(split_counts.get("validation", 0)),
        "test_size": int(split_counts.get("test", test_metrics["n"])),
        "labels": labels,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
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
        "abstain_precision": abstain.get("precision"),
        "abstain_recall": abstain.get("recall"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "real_hf_model": bundle["model_id"] not in {"fake", "unknown"} and not fallback_run,
        "fallback_run": fallback_run,
        "split_source": metrics["split_source"],
        "recommended_thresholds": metrics["recommended_thresholds"],
        "class_counts": metrics["class_counts"],
        "detailed_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and serialize a scikit-learn linear probe over frozen LLM activations."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="NPZ activation artifact produced by scripts/extract_features.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/probe.joblib",
        help="Joblib path for the trained probe bundle.",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/metrics.json",
        help="JSON path for split metrics and class metadata.",
    )
    parser.add_argument(
        "--confusion-matrix",
        default="artifacts/confusion_matrix.csv",
        help="CSV path for the held-out test confusion matrix.",
    )
    parser.add_argument(
        "--thresholds",
        default="artifacts/thresholds.json",
        help="JSON path for recommended confidence/margin/centroid gates.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fallback held-out test fraction when feature metadata has no split column.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for fallback train/test splitting.",
    )
    parser.add_argument(
        "--fallback-run",
        action="store_true",
        help="Mark metrics as fallback-only rather than real HF semantic evidence.",
    )
    args = parser.parse_args()

    bundle = train_probe(
        args.features,
        args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = build_metrics_payload(bundle, fallback_run=args.fallback_run)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    labels = bundle["metrics"]["labels"]
    confusion_matrix_path = Path(args.confusion_matrix)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        bundle["metrics"]["confusion_matrix"],
        index=labels,
        columns=labels,
    ).to_csv(confusion_matrix_path)

    thresholds_path = Path(args.thresholds)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_payload = {
        **bundle["metrics"]["recommended_thresholds"],
        "selection_method": (
            "5th percentile confidence/margin and 95th percentile class-centroid "
            "distance on validation split when present, otherwise training split"
        ),
        "threshold_source": "validation"
        if bundle["metrics"]["splits"]["validation"]["n"] > 0
        else "train",
    }
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)

    print(f"wrote {args.output}")
    print(f"wrote {metrics_path}")
    print(f"wrote {confusion_matrix_path}")
    print(f"wrote {thresholds_path}")
    print(json.dumps(bundle["metrics"]["classification_report"], indent=2)[:2000])


if __name__ == "__main__":
    main()
