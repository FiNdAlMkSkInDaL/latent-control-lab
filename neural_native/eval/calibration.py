from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from neural_native.eval.ood import (
    classwise_centroid_threshold_scores,
    classwise_centroid_thresholds,
    entropy,
    mahalanobis_scores,
    max_probability,
    nearest_centroid_distances,
    top2_margin,
)


@dataclass(slots=True)
class CalibrationResult:
    method: str
    threshold: float
    objective: float
    executable_accept_rate: float
    hard_negative_reject_rate: float
    macro_f1: float


def calibration_objective(
    *,
    executable_accept_rate: float,
    hard_negative_reject_rate: float,
    macro_f1: float,
) -> float:
    return (
        0.4 * executable_accept_rate
        + 0.4 * hard_negative_reject_rate
        + 0.2 * macro_f1
    )


def effective_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.isfinite(scores) & (scores >= threshold) & (labels != "ABSTAIN")
    effective = np.where(accepted, labels, "ABSTAIN")
    return effective.astype(str), accepted


def evaluate_threshold(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: list[str],
) -> dict[str, float]:
    effective, accepted = effective_predictions(predicted_labels, scores, threshold)
    executable_mask = y_true != "ABSTAIN"
    hard_negative_mask = y_true == "ABSTAIN"
    executable_accept_rate = (
        float(np.mean(accepted[executable_mask])) if executable_mask.any() else 0.0
    )
    hard_negative_reject_rate = (
        float(np.mean(~accepted[hard_negative_mask])) if hard_negative_mask.any() else 0.0
    )
    macro_f1 = float(f1_score(y_true, effective, labels=labels, average="macro", zero_division=0))
    objective = calibration_objective(
        executable_accept_rate=executable_accept_rate,
        hard_negative_reject_rate=hard_negative_reject_rate,
        macro_f1=macro_f1,
    )
    return {
        "objective": objective,
        "executable_accept_rate": executable_accept_rate,
        "hard_negative_reject_rate": hard_negative_reject_rate,
        "macro_f1": macro_f1,
    }


def select_threshold(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
) -> CalibrationResult:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("Cannot calibrate threshold: all scores are NaN")
    candidates = np.unique(finite_scores)
    best: CalibrationResult | None = None
    for threshold in candidates:
        metrics = evaluate_threshold(
            y_true,
            predicted_labels,
            scores,
            float(threshold),
            labels,
        )
        result = CalibrationResult(
            method="",
            threshold=float(threshold),
            objective=metrics["objective"],
            executable_accept_rate=metrics["executable_accept_rate"],
            hard_negative_reject_rate=metrics["hard_negative_reject_rate"],
            macro_f1=metrics["macro_f1"],
        )
        if best is None or result.objective > best.objective:
            best = result
    if best is None:
        raise RuntimeError("Threshold selection produced no candidate")
    return best


def score_methods(
    X: np.ndarray,
    *,
    probs: np.ndarray,
    classes: list[str],
    centroids: dict[str, np.ndarray],
    X_train_executable: np.ndarray,
    y_train_executable: np.ndarray,
) -> dict[str, np.ndarray]:
    thresholds = classwise_centroid_thresholds(
        X_train_executable,
        y_train_executable,
        centroids,
        excluded_labels={"ABSTAIN"},
    )
    return {
        "max_softmax_probability": max_probability(probs),
        "top2_margin": top2_margin(probs),
        "negative_entropy": -entropy(probs),
        "negative_nearest_executable_centroid_distance": -nearest_centroid_distances(
            X,
            centroids,
            excluded_labels={"ABSTAIN"},
        ),
        "classwise_centroid_threshold": classwise_centroid_threshold_scores(
            X,
            probs,
            classes,
            centroids,
            thresholds,
            excluded_labels={"ABSTAIN"},
        ),
        "negative_mahalanobis_distance": mahalanobis_scores(X_train_executable, X),
    }


def mahalanobis_parameters(X_train: np.ndarray, *, regularization: float = 1e-3) -> dict[str, Any]:
    mean = X_train.mean(axis=0)
    covariance = np.cov(X_train, rowvar=False)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * regularization
    precision = np.linalg.pinv(covariance)
    return {
        "mean": mean.astype(float).tolist(),
        "precision": precision.astype(float).tolist(),
        "regularization": regularization,
    }
