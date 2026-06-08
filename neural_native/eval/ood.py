from __future__ import annotations

from typing import Any

import numpy as np


def max_probability(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def entropy(probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def centroid_distances(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    distances = []
    for z, label in zip(X, labels, strict=True):
        distances.append(float(np.linalg.norm(z - centroids[str(label)])))
    return np.asarray(distances, dtype=np.float32)


def predicted_centroid_distances(
    X: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    predicted_labels = np.asarray(classes, dtype=object)[np.argmax(probs, axis=1)]
    distances: list[float] = []
    for z, label in zip(X, predicted_labels, strict=True):
        centroid = centroids.get(str(label))
        if centroid is None:
            distances.append(float("nan"))
        else:
            distances.append(float(np.linalg.norm(z - centroid)))
    return np.asarray(distances, dtype=np.float32)


def nearest_centroid_distances(
    X: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    excluded_labels: set[str] | None = None,
) -> np.ndarray:
    excluded_labels = excluded_labels or set()
    executable_centroids = [
        centroid
        for label, centroid in centroids.items()
        if label not in excluded_labels
    ]
    if not executable_centroids:
        return np.full(X.shape[0], np.nan, dtype=np.float32)

    centroid_matrix = np.stack(executable_centroids).astype(np.float32)
    distances = np.linalg.norm(X[:, None, :] - centroid_matrix[None, :, :], axis=2)
    return distances.min(axis=1).astype(np.float32)


def classwise_centroid_threshold_scores(
    X: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    centroids: dict[str, np.ndarray],
    thresholds: dict[str, float],
    *,
    excluded_labels: set[str] | None = None,
) -> np.ndarray:
    """
    Score examples by the predicted executable class's centroid threshold.

    Higher scores mean more in-domain: positive values are inside the class
    radius, negative values are outside it. Predicted excluded labels get NaN.
    """

    excluded_labels = excluded_labels or set()
    predicted_labels = np.asarray(classes, dtype=object)[np.argmax(probs, axis=1)]
    scores: list[float] = []
    for z, label in zip(X, predicted_labels, strict=True):
        label = str(label)
        centroid = centroids.get(label)
        threshold = thresholds.get(label)
        if label in excluded_labels or centroid is None or threshold is None:
            scores.append(float("nan"))
            continue
        distance = float(np.linalg.norm(z - centroid))
        scores.append(float(threshold - distance))
    return np.asarray(scores, dtype=np.float32)


def classwise_centroid_thresholds(
    X: np.ndarray,
    y: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    quantile: float = 0.95,
    excluded_labels: set[str] | None = None,
) -> dict[str, float]:
    excluded_labels = excluded_labels or set()
    thresholds: dict[str, float] = {}
    for label in sorted(set(map(str, y))):
        if label in excluded_labels or label not in centroids:
            continue
        mask = y.astype(str) == label
        if not mask.any():
            continue
        distances = np.linalg.norm(X[mask] - centroids[label], axis=1)
        thresholds[label] = float(np.quantile(distances, quantile))
    return thresholds


def mahalanobis_scores(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    regularization: float = 1e-3,
) -> np.ndarray:
    """
    Return negative Mahalanobis distance to executable training activations.

    The covariance is diagonal-regularized for small feature sets. Higher scores
    mean more in-domain.
    """

    if X_train.shape[0] < 2:
        return np.full(X_eval.shape[0], np.nan, dtype=np.float32)
    mean = X_train.mean(axis=0)
    covariance = np.cov(X_train, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * regularization
    try:
        precision = np.linalg.pinv(covariance)
    except np.linalg.LinAlgError:
        return np.full(X_eval.shape[0], np.nan, dtype=np.float32)
    delta = X_eval - mean
    distances = np.sqrt(np.einsum("ij,jk,ik->i", delta, precision, delta))
    return (-distances).astype(np.float32)


def score_id_vs_abstain(
    probe: Any,
    X: np.ndarray,
    y: np.ndarray,
    centroids: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return OOD gate signals where higher scores mean more likely in-domain."""

    del y
    probs = probe.predict_proba(X)
    scores = {
        "max_probability": max_probability(probs),
        "top2_margin": top2_margin(probs),
        "negative_entropy": -entropy(probs),
    }
    if centroids:
        distances = nearest_centroid_distances(X, centroids, excluded_labels={"ABSTAIN"})
        scores["negative_executable_centroid_distance"] = -distances
    return scores
