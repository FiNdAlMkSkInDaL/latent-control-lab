from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_centroids(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    centroids = {}
    for label in sorted(set(map(str, y))):
        centroids[label] = X[y == label].mean(axis=0).astype(np.float32)
    return centroids


def _array_from_npz(data: Any, key: str, length: int, default: str) -> np.ndarray:
    if key in data.files:
        return data[key].astype(str)
    return np.array([default] * length, dtype=str)


def _metadata_splits(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"
    if train_mask.any() and test_mask.any():
        return train_mask, validation_mask, test_mask
    return None


def _evaluate_probe(
    probe: Any,
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    if X.shape[0] == 0:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "classification_report": {},
            "confusion_matrix": [],
        }

    y_pred = probe.predict(X)
    return {
        "n": int(X.shape[0]),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
    }


def _centroid_distances(
    X: np.ndarray,
    y: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    distances = []
    for z, label in zip(X, y, strict=True):
        centroid = centroids.get(str(label))
        if centroid is not None:
            distances.append(float(np.linalg.norm(z - centroid)))
    return np.asarray(distances, dtype=np.float32)


def _recommended_thresholds(
    probe: Any,
    X: np.ndarray,
    y: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> dict[str, float | None]:
    if X.shape[0] == 0:
        return {
            "min_confidence": 0.60,
            "min_margin": 0.15,
            "max_centroid_distance": None,
        }

    probs = probe.predict_proba(X)
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    id_mask = y != "ABSTAIN"

    threshold_mask = id_mask if id_mask.any() else np.ones_like(id_mask, dtype=bool)
    distances = _centroid_distances(X[threshold_mask], y[threshold_mask], centroids)

    return {
        "min_confidence": float(np.quantile(confidence[threshold_mask], 0.05)),
        "min_margin": float(np.quantile(margin[threshold_mask], 0.05)),
        "max_centroid_distance": float(np.quantile(distances, 0.95))
        if distances.size
        else None,
    }


def train_probe(
    feature_path: str | Path,
    output_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    data = np.load(feature_path, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype(str)
    split = _array_from_npz(data, "split", len(y), "unknown")

    metadata_splits = _metadata_splits(split)
    if metadata_splits is not None:
        train_mask, validation_mask, test_mask = metadata_splits
        X_train, y_train = X[train_mask], y[train_mask]
        X_validation, y_validation = X[validation_mask], y[validation_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        split_source = "metadata"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        X_validation = np.empty((0, X.shape[1]), dtype=np.float32)
        y_validation = np.array([], dtype=str)
        split_source = "generated_train_test_split"

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    probe.fit(X_train, y_train)

    labels = list(probe.classes_)
    centroids = compute_centroids(X_train, y_train)
    split_metrics = {
        "train": _evaluate_probe(probe, X_train, y_train, labels),
        "validation": _evaluate_probe(probe, X_validation, y_validation, labels),
        "test": _evaluate_probe(probe, X_test, y_test, labels),
    }
    threshold_source_X = X_validation if X_validation.shape[0] else X_train
    threshold_source_y = y_validation if y_validation.shape[0] else y_train
    recommended_thresholds = _recommended_thresholds(
        probe,
        threshold_source_X,
        threshold_source_y,
        centroids,
    )

    class_counts = {
        label: int(count)
        for label, count in zip(*np.unique(y, return_counts=True), strict=True)
    }
    split_counts = {
        name: int(count)
        for name, count in zip(*np.unique(split, return_counts=True), strict=True)
    }

    bundle: dict[str, Any] = {
        "probe": probe,
        "classes": probe.classes_,
        "centroids": centroids,
        "model_id": str(data["model_id"][0]) if "model_id" in data else "unknown",
        "prompt_template": str(data["prompt_template"][0])
        if "prompt_template" in data
        else "unknown",
        "feature_space": str(data["feature_space"][0]) if "feature_space" in data else "unknown",
        "metrics": {
            "classification_report": split_metrics["test"]["classification_report"],
            "confusion_matrix": split_metrics["test"]["confusion_matrix"],
            "labels": labels,
            "split_source": split_source,
            "splits": split_metrics,
            "recommended_thresholds": recommended_thresholds,
            "class_counts": class_counts,
            "split_counts": split_counts,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle
