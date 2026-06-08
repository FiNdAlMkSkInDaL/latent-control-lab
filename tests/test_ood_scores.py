from __future__ import annotations

import numpy as np

from neural_native.eval.ood import (
    classwise_centroid_threshold_scores,
    classwise_centroid_thresholds,
    entropy,
    mahalanobis_scores,
    max_probability,
    nearest_centroid_distances,
    top2_margin,
)


def test_probability_scores_have_expected_shape() -> None:
    probs = np.array([[0.8, 0.2], [0.45, 0.55]], dtype=np.float32)

    assert np.allclose(max_probability(probs), [0.8, 0.55])
    assert np.allclose(top2_margin(probs), [0.6, 0.1], atol=1e-6)
    assert entropy(probs).shape == (2,)


def test_centroid_and_threshold_scores() -> None:
    X = np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32)
    y = np.array(["CREATE_TASK", "PROMOTE_TASK"])
    centroids = {
        "CREATE_TASK": np.array([0.0, 0.0], dtype=np.float32),
        "PROMOTE_TASK": np.array([3.0, 3.0], dtype=np.float32),
        "ABSTAIN": np.array([10.0, 10.0], dtype=np.float32),
    }
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], dtype=np.float32)
    classes = ["CREATE_TASK", "PROMOTE_TASK", "ABSTAIN"]

    distances = nearest_centroid_distances(X, centroids, excluded_labels={"ABSTAIN"})
    thresholds = classwise_centroid_thresholds(X, y, centroids)
    scores = classwise_centroid_threshold_scores(
        X,
        probs,
        classes,
        centroids,
        thresholds,
        excluded_labels={"ABSTAIN"},
    )

    assert np.allclose(distances, [0.0, 0.0])
    assert set(thresholds) == {"CREATE_TASK", "PROMOTE_TASK"}
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_mahalanobis_scores_are_finite_for_regularized_covariance() -> None:
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X_eval = np.array([[0.1, 0.1], [5.0, 5.0]], dtype=np.float32)

    scores = mahalanobis_scores(X_train, X_eval)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1]
