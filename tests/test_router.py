import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_native.app.state import Action
from neural_native.bridge.router import LinearProbeRouter, RouterThresholds


def make_probe_bundle(path) -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 1.0],
            [0.0, 1.1],
            [-1.0, -1.0],
            [-1.1, -1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([
        "CREATE_TASK",
        "CREATE_TASK",
        "COMPLETE_ACTIVE",
        "COMPLETE_ACTIVE",
        "ABSTAIN",
        "ABSTAIN",
    ])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    joblib.dump(
        {
            "probe": probe,
            "classes": probe.classes_,
            "centroids": {
                "CREATE_TASK": np.array([1.05, 0.0], dtype=np.float32),
                "COMPLETE_ACTIVE": np.array([0.0, 1.05], dtype=np.float32),
                "ABSTAIN": np.array([-1.05, -1.0], dtype=np.float32),
            },
        },
        path,
    )


def test_router_maps_probe_label_to_action(tmp_path) -> None:
    bundle_path = tmp_path / "probe.joblib"
    make_probe_bundle(bundle_path)
    router = LinearProbeRouter(
        str(bundle_path),
        thresholds=RouterThresholds(min_confidence=0.0, min_margin=0.0),
    )
    decision = router.predict(np.array([1.2, 0.0], dtype=np.float32))
    assert decision.action == Action.CREATE_TASK
    assert decision.accepted is True


def test_router_rejects_abstain(tmp_path) -> None:
    bundle_path = tmp_path / "probe.joblib"
    make_probe_bundle(bundle_path)
    router = LinearProbeRouter(
        str(bundle_path),
        thresholds=RouterThresholds(min_confidence=0.0, min_margin=0.0),
    )
    decision = router.predict(np.array([-1.2, -1.0], dtype=np.float32))
    assert decision.action == Action.ABSTAIN
    assert decision.accepted is False
