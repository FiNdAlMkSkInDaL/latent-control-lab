from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from neural_native.app.state import LABEL_TO_ACTION, Action
from neural_native.app.vector_port import RouteDecision
from neural_native.eval.ood import entropy, nearest_centroid_distances


@dataclass(slots=True)
class RouterThresholds:
    min_confidence: float = 0.60
    min_margin: float = 0.15
    max_centroid_distance: float | None = None
    ood_method: str | None = None
    ood_threshold: float | None = None
    classwise_centroid_thresholds: dict[str, float] | None = None
    mahalanobis_mean: np.ndarray | None = None
    mahalanobis_precision: np.ndarray | None = None


def _thresholds_from_bundle(bundle: dict[str, Any]) -> RouterThresholds:
    defaults = RouterThresholds()
    recommendations = bundle.get("metrics", {}).get("recommended_thresholds", {})
    if not recommendations:
        return defaults

    max_centroid_distance = recommendations.get("max_centroid_distance")
    return RouterThresholds(
        min_confidence=float(recommendations.get("min_confidence", defaults.min_confidence)),
        min_margin=float(recommendations.get("min_margin", defaults.min_margin)),
        max_centroid_distance=float(max_centroid_distance)
        if max_centroid_distance is not None
        else defaults.max_centroid_distance,
    )


def thresholds_from_json(path: str | Path) -> RouterThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = payload.get("parameters", {})
    mahalanobis = params.get("mahalanobis", {})
    mean = mahalanobis.get("mean")
    precision = mahalanobis.get("precision")
    return RouterThresholds(
        min_confidence=float(payload.get("min_confidence", 0.0)),
        min_margin=float(payload.get("min_margin", 0.0)),
        max_centroid_distance=payload.get("max_centroid_distance"),
        ood_method=payload.get("ood_method"),
        ood_threshold=payload.get("ood_threshold"),
        classwise_centroid_thresholds=params.get("classwise_centroid_thresholds"),
        mahalanobis_mean=np.asarray(mean, dtype=np.float32) if mean is not None else None,
        mahalanobis_precision=np.asarray(precision, dtype=np.float32)
        if precision is not None
        else None,
    )


class LinearProbeRouter:
    """Routes activation vectors using a serialized scikit-learn probe bundle."""

    def __init__(self, bundle_path: str, thresholds: RouterThresholds | None = None) -> None:
        bundle = joblib.load(bundle_path)
        self.probe = bundle["probe"]
        self.classes = list(bundle["classes"])
        self.thresholds = thresholds or _thresholds_from_bundle(bundle)
        self.centroids: dict[str, np.ndarray] | None = bundle.get("centroids")
        self.metadata: dict[str, Any] = {
            key: value for key, value in bundle.items() if key not in {"probe", "centroids"}
        }

    def predict(self, z: np.ndarray) -> RouteDecision:
        z2 = np.asarray(z, dtype=np.float32).reshape(1, -1)
        probs = self.probe.predict_proba(z2)[0]
        order = np.argsort(probs)[::-1]

        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else top1_idx
        label = str(self.classes[top1_idx])

        confidence = float(probs[top1_idx])
        margin = float(probs[top1_idx] - probs[top2_idx])
        centroid_distance = self._centroid_distance(label, z2[0])
        ood_score = self._ood_score(z2[0], probs, label, centroid_distance)

        action = LABEL_TO_ACTION.get(label, Action.ABSTAIN)
        accepted = self._accepted(label, confidence, margin, centroid_distance, ood_score)

        return RouteDecision(
            action=action,
            label=label,
            confidence=confidence,
            margin=margin,
            ood_score=ood_score if ood_score is not None else 0.0,
            accepted=accepted,
        )

    def _centroid_distance(self, label: str, z: np.ndarray) -> float | None:
        if not self.centroids or label not in self.centroids:
            return None
        return float(np.linalg.norm(z - self.centroids[label]))

    def _ood_score(
        self,
        z: np.ndarray,
        probs: np.ndarray,
        label: str,
        centroid_distance: float | None,
    ) -> float | None:
        method = self.thresholds.ood_method
        if method is None:
            return centroid_distance
        if method == "max_softmax_probability":
            return float(np.max(probs))
        if method == "top2_margin":
            sorted_probs = np.sort(probs)
            return float(sorted_probs[-1] - sorted_probs[-2])
        if method == "negative_entropy":
            return float(-entropy(probs.reshape(1, -1))[0])
        if method == "negative_nearest_executable_centroid_distance" and self.centroids:
            return float(
                -nearest_centroid_distances(
                    z.reshape(1, -1),
                    self.centroids,
                    excluded_labels={"ABSTAIN"},
                )[0]
            )
        if method == "classwise_centroid_threshold":
            thresholds = self.thresholds.classwise_centroid_thresholds or {}
            if centroid_distance is None or label not in thresholds:
                return None
            return float(thresholds[label] - centroid_distance)
        if method == "negative_mahalanobis_distance":
            mean = self.thresholds.mahalanobis_mean
            precision = self.thresholds.mahalanobis_precision
            if mean is None or precision is None:
                return None
            delta = z - mean
            distance = float(np.sqrt(delta @ precision @ delta.T))
            return -distance
        return centroid_distance

    def _accepted(
        self,
        label: str,
        confidence: float,
        margin: float,
        centroid_distance: float | None,
        ood_score: float | None,
    ) -> bool:
        if label == "ABSTAIN":
            return False
        if confidence < self.thresholds.min_confidence:
            return False
        if margin < self.thresholds.min_margin:
            return False
        if (
            self.thresholds.max_centroid_distance is not None
            and centroid_distance is not None
            and centroid_distance > self.thresholds.max_centroid_distance
        ):
            return False
        if (
            self.thresholds.ood_method is not None
            and self.thresholds.ood_threshold is not None
            and (ood_score is None or ood_score < self.thresholds.ood_threshold)
        ):
            return False
        return True
