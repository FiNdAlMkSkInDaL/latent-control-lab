from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_examples_per_class(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_features(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    payload: dict[str, Any] = {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(str),
        "split": data["split"].astype(str)
        if "split" in data.files
        else np.array(["unknown"] * len(data["y"])),
    }
    if "subset" in data.files:
        payload["subset"] = data["subset"].astype(str)
    if "model_id" in data.files:
        payload["model_id"] = str(data["model_id"][0])
    if "feature_space" in data.files:
        payload["feature_space"] = str(data["feature_space"][0])
    return payload


def balanced_train_indices(y: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in sorted(set(map(str, y))):
        label_indices = np.flatnonzero(y == label)
        n_take = min(n_per_class, len(label_indices))
        chosen = rng.choice(label_indices, size=n_take, replace=False)
        indices.extend(int(i) for i in chosen)
    return np.asarray(sorted(indices), dtype=int)


def train_linear_probe(X: np.ndarray, y: np.ndarray) -> Any:
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


def evaluate(probe: Any, X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict[str, Any]:
    y_pred = probe.predict(X)
    report = classification_report(
        y,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    abstain = report.get("ABSTAIN", {})
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
        "abstain_precision": abstain.get("precision"),
        "abstain_recall": abstain.get("recall"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train probes with different examples-per-class budgets."
    )
    parser.add_argument(
        "--features",
        default="artifacts/features_distilgpt2_pre_lm_head.npz",
        help="Normal synthetic feature NPZ.",
    )
    parser.add_argument(
        "--hard-features",
        default="artifacts/hard_eval_features_distilgpt2_pre_lm_head.npz",
        help="Hard-eval feature NPZ produced by scripts/evaluate_hard_eval.py.",
    )
    parser.add_argument(
        "--examples-per-class",
        default="5,10,20,30,50",
        help="Comma-separated training examples per class.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/data_efficiency.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/data_efficiency.csv",
        help="CSV output path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    normal = _load_features(args.features)
    hard = _load_features(args.hard_features)
    train_mask = normal["split"] == "train"
    test_mask = normal["split"] == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Normal features must include train and test split metadata")

    X_train_all = normal["X"][train_mask]
    y_train_all = normal["y"][train_mask]
    X_test = normal["X"][test_mask]
    y_test = normal["y"][test_mask]
    X_hard = hard["X"]
    y_hard = hard["y"]
    labels = sorted(set(map(str, normal["y"])) | set(map(str, hard["y"])))

    rows = []
    for n_per_class in parse_examples_per_class(args.examples_per_class):
        train_idx = balanced_train_indices(y_train_all, n_per_class, args.seed + n_per_class)
        probe = train_linear_probe(X_train_all[train_idx], y_train_all[train_idx])
        test_metrics = evaluate(probe, X_test, y_test, labels)
        hard_metrics = evaluate(probe, X_hard, y_hard, labels)
        rows.append(
            {
                "examples_per_class": n_per_class,
                "total_train_examples": int(len(train_idx)),
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "hard_eval_accuracy": hard_metrics["accuracy"],
                "hard_eval_macro_f1": hard_metrics["macro_f1"],
                "abstain_precision": hard_metrics["abstain_precision"],
                "abstain_recall": hard_metrics["abstain_recall"],
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": args.features,
        "hard_features": args.hard_features,
        "model_id": normal.get("model_id"),
        "feature_space": normal.get("feature_space"),
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
