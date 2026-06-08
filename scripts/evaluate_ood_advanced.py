from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from neural_native.eval.ood import (
    classwise_centroid_threshold_scores,
    classwise_centroid_thresholds,
    entropy,
    mahalanobis_scores,
    max_probability,
    nearest_centroid_distances,
    top2_margin,
)

ScoreFn = Callable[[np.ndarray], np.ndarray]


def _load_features(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    y = data["y"].astype(str)
    payload: dict[str, Any] = {
        "X": data["X"].astype(np.float32),
        "y": y,
        "split": data["split"].astype(str)
        if "split" in data.files
        else np.array(["unknown"] * len(y)),
        "subset": data["subset"].astype(str)
        if "subset" in data.files
        else np.array(["unknown"] * len(y)),
    }
    return payload


def _best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    finite = np.isfinite(scores)
    scores = scores[finite]
    y_true = y_true[finite]
    if len(set(y_true.tolist())) < 2:
        return float(np.nanmedian(scores))
    candidates = np.unique(scores)
    best_threshold = float(candidates[0])
    best_youden = -float("inf")
    for threshold in candidates:
        accepted = scores >= threshold
        true_positive = np.mean(accepted[y_true == 1]) if (y_true == 1).any() else 0.0
        false_positive = np.mean(accepted[y_true == 0]) if (y_true == 0).any() else 0.0
        youden = true_positive - false_positive
        if youden > best_youden:
            best_youden = float(youden)
            best_threshold = float(threshold)
    return best_threshold


def _metric_value(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    finite = np.isfinite(scores)
    if finite.sum() == 0 or len(set(y_true[finite].tolist())) < 2:
        return None
    return float(metric_fn(y_true[finite], scores[finite]))


def _accept(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(scores) & (scores >= threshold)


def rejection_by_subset(
    scores: np.ndarray,
    threshold: float,
    subsets: np.ndarray,
) -> dict[str, float]:
    accepted = _accept(scores, threshold)
    rates = {}
    for subset in sorted(set(map(str, subsets))):
        mask = subsets == subset
        if mask.any():
            rates[str(subset)] = float(np.mean(~accepted[mask]))
    return rates


def build_score_functions(bundle: dict[str, Any], normal: dict[str, Any]) -> dict[str, ScoreFn]:
    probe = bundle["probe"]
    classes = list(bundle["classes"])
    centroids = bundle.get("centroids", {})
    train_exec_mask = (normal["split"] == "train") & (normal["y"] != "ABSTAIN")
    X_train_exec = normal["X"][train_exec_mask]
    y_train_exec = normal["y"][train_exec_mask]
    centroid_thresholds = classwise_centroid_thresholds(
        X_train_exec,
        y_train_exec,
        centroids,
        excluded_labels={"ABSTAIN"},
    )

    lof = None
    if X_train_exec.shape[0] > 3:
        n_neighbors = min(20, X_train_exec.shape[0] - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train_exec)

    one_class_svm = None
    if X_train_exec.shape[0] > 3:
        one_class_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        one_class_svm.fit(X_train_exec)

    def probs(X: np.ndarray) -> np.ndarray:
        return probe.predict_proba(X)

    return {
        "max_softmax_probability": lambda X: max_probability(probs(X)),
        "top2_margin": lambda X: top2_margin(probs(X)),
        "negative_entropy": lambda X: -entropy(probs(X)),
        "negative_nearest_executable_centroid_distance": lambda X: -nearest_centroid_distances(
            X,
            centroids,
            excluded_labels={"ABSTAIN"},
        ),
        "classwise_centroid_threshold": lambda X: classwise_centroid_threshold_scores(
            X,
            probs(X),
            classes,
            centroids,
            centroid_thresholds,
            excluded_labels={"ABSTAIN"},
        ),
        "negative_mahalanobis_distance": lambda X: mahalanobis_scores(X_train_exec, X),
        "lof_novelty_score": lambda X: lof.score_samples(X).astype(np.float32)
        if lof is not None
        else np.full(X.shape[0], np.nan, dtype=np.float32),
        "one_class_svm_score": lambda X: one_class_svm.score_samples(X).astype(np.float32)
        if one_class_svm is not None
        else np.full(X.shape[0], np.nan, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare stronger OOD scores for latent router rejection."
    )
    parser.add_argument("--probe", default="artifacts/probe.joblib", help="Probe bundle path.")
    parser.add_argument(
        "--features",
        default="artifacts/features_distilgpt2_pre_lm_head.npz",
        help="Normal synthetic feature NPZ.",
    )
    parser.add_argument(
        "--hard-features",
        default="artifacts/hard_eval_features_distilgpt2_pre_lm_head.npz",
        help="Hard-eval feature NPZ.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/ood_advanced_metrics.json",
        help="JSON metrics output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/ood_advanced_metrics.csv",
        help="CSV metrics output path.",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.probe)
    normal = _load_features(args.features)
    hard = _load_features(args.hard_features)

    id_mask = (normal["split"] == "test") & (normal["y"] != "ABSTAIN")
    normal_ood_mask = (normal["split"] == "test") & (normal["y"] == "ABSTAIN")
    hard_ood_mask = hard["y"] == "ABSTAIN"
    validation_mask = normal["split"] == "validation"

    X_id = normal["X"][id_mask]
    X_ood = np.concatenate([normal["X"][normal_ood_mask], hard["X"][hard_ood_mask]], axis=0)
    ood_subsets = np.concatenate(
        [
            np.array(["normal_test_abstain"] * int(normal_ood_mask.sum())),
            hard["subset"][hard_ood_mask],
        ]
    )
    X_eval = np.concatenate([X_id, X_ood], axis=0)
    y_eval = np.concatenate(
        [np.ones(X_id.shape[0], dtype=int), np.zeros(X_ood.shape[0], dtype=int)]
    )

    X_validation = normal["X"][validation_mask]
    y_validation = (normal["y"][validation_mask] != "ABSTAIN").astype(int)
    score_functions = build_score_functions(bundle, normal)
    rows: list[dict[str, Any]] = []
    for name, score_fn in score_functions.items():
        eval_scores = score_fn(X_eval)
        validation_scores = score_fn(X_validation)
        threshold = _best_threshold(validation_scores, y_validation)
        accepted = _accept(eval_scores, threshold)
        id_accepted = accepted[: X_id.shape[0]]
        ood_accepted = accepted[X_id.shape[0] :]
        rows.append(
            {
                "score": name,
                "auroc": _metric_value(roc_auc_score, y_eval, eval_scores),
                "auprc": _metric_value(average_precision_score, y_eval, eval_scores),
                "best_threshold_from_validation": threshold,
                "false_accept_rate": float(np.mean(ood_accepted)) if len(ood_accepted) else None,
                "false_reject_rate": float(np.mean(~id_accepted)) if len(id_accepted) else None,
                "rejection_rate_by_ood_subset": rejection_by_subset(
                    eval_scores[X_id.shape[0] :],
                    threshold,
                    ood_subsets,
                ),
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe": args.probe,
        "features": args.features,
        "hard_features": args.hard_features,
        "id_definition": "normal held-out test executable examples",
        "ood_definition": "normal ABSTAIN test plus hard-eval ABSTAIN subsets",
        "n_id": int(X_id.shape[0]),
        "n_ood": int(X_ood.shape[0]),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {key: value for key, value in row.items() if key != "rejection_rate_by_ood_subset"}
        for row in rows
    ]
    pd.DataFrame(flat_rows).to_csv(output_csv, index=False)
    print(pd.DataFrame(flat_rows))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
