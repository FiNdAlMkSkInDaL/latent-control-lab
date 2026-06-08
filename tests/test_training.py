import numpy as np

from neural_native.bridge.train_sklearn import train_probe


def test_train_probe_serializes_bundle(tmp_path) -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.0],
            [0.0, 1.0],
            [0.0, 1.1],
            [0.0, 1.2],
            [-1.0, -1.0],
            [-1.1, -1.0],
            [-1.2, -1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([
        "CREATE_TASK",
        "CREATE_TASK",
        "CREATE_TASK",
        "COMPLETE_ACTIVE",
        "COMPLETE_ACTIVE",
        "COMPLETE_ACTIVE",
        "ABSTAIN",
        "ABSTAIN",
        "ABSTAIN",
    ])
    features = tmp_path / "features.npz"
    np.savez_compressed(
        features,
        X=X,
        y=y,
        model_id=np.array(["fake"]),
        prompt_template=np.array(["template"]),
        feature_space=np.array(["test_space"]),
    )
    output = tmp_path / "probe.joblib"
    bundle = train_probe(features, output, test_size=0.34)
    assert output.exists()
    assert "probe" in bundle
    assert "metrics" in bundle
