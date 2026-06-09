from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_native.bridge.train_sklearn import train_probe


def _feature_shape(feature_path: str | Path) -> list[int]:
    data = np.load(feature_path, allow_pickle=True)
    return [int(value) for value in data["X"].shape]


def build_metrics_payload(bundle: dict[str, Any], *, feature_shape: list[int]) -> dict[str, Any]:
    metrics = bundle["metrics"]
    test_metrics = metrics["splits"]["test"]
    report = test_metrics["classification_report"]
    split_counts = metrics["split_counts"]
    labels = metrics["labels"]
    abstain = report.get("ABSTAIN", {})
    model_id = bundle["model_id"]
    fallback_run = str(model_id).startswith("fake-vectorbot-plumbing")

    return {
        "model_id": model_id,
        "prompt_template": bundle["prompt_template"],
        "feature_space": bundle["feature_space"],
        "feature_shape": feature_shape,
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
        "real_hf_model": not fallback_run,
        "fallback_run": fallback_run,
        "split_source": metrics["split_source"],
        "recommended_thresholds": metrics["recommended_thresholds"],
        "class_counts": metrics["class_counts"],
        "detailed_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a scikit-learn linear probe for VectorBot hidden-state routes."
    )
    parser.add_argument(
        "--features",
        default="artifacts/vectorbot_features_distilgpt2.npz",
        help="NPZ activation artifact produced by scripts/extract_vectorbot_features.py.",
    )
    parser.add_argument(
        "--output",
        "--probe-output",
        dest="output",
        default="artifacts/vectorbot_probe_distilgpt2.joblib",
        help="Joblib path for the trained VectorBot probe bundle.",
    )
    parser.add_argument(
        "--metrics",
        "--metrics-output",
        dest="metrics",
        default="artifacts/vectorbot_metrics.json",
        help="JSON path for VectorBot metrics.",
    )
    parser.add_argument(
        "--confusion-matrix",
        "--confusion-output",
        dest="confusion_matrix",
        default="artifacts/vectorbot_confusion_matrix.csv",
        help="CSV path for held-out test confusion matrix.",
    )
    parser.add_argument(
        "--thresholds",
        "--thresholds-output",
        dest="thresholds",
        default="artifacts/vectorbot_thresholds.json",
        help="JSON path for recommended confidence/margin/centroid gates.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fallback held-out test fraction when feature metadata has no split column.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Fallback split seed.")
    args = parser.parse_args()

    bundle = train_probe(
        args.features,
        args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    feature_shape = _feature_shape(args.features)
    metrics_payload = build_metrics_payload(bundle, feature_shape=feature_shape)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

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
    recommended_thresholds = dict(bundle["metrics"]["recommended_thresholds"])
    thresholds_payload = {
        **recommended_thresholds,
        "max_centroid_distance": None,
        "centroid_distance_reference": recommended_thresholds.get("max_centroid_distance"),
        "selection_method": (
            "5th percentile confidence/margin on validation split when present, "
            "otherwise training split. The centroid reference is reported but not "
            "activated by default for the small VectorBot demo."
        ),
        "threshold_source": "validation"
        if bundle["metrics"]["splits"]["validation"]["n"] > 0
        else "train",
    }
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"wrote {metrics_path}")
    print(f"wrote {confusion_matrix_path}")
    print(f"wrote {thresholds_path}")
    print(json.dumps(metrics_payload, indent=2)[:2000])


if __name__ == "__main__":
    main()
