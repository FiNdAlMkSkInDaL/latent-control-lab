from __future__ import annotations

import numpy as np

from neural_native.eval.calibration import (
    calibration_objective,
    effective_predictions,
    select_threshold,
)


def test_calibration_objective_weights_terms() -> None:
    score = calibration_objective(
        executable_accept_rate=0.5,
        hard_negative_reject_rate=1.0,
        macro_f1=0.25,
    )
    assert abs(score - 0.65) < 1e-9


def test_effective_predictions_rejects_below_threshold_and_abstain_label() -> None:
    labels = np.array(["CREATE_TASK", "ABSTAIN", "PROMOTE_TASK"])
    scores = np.array([0.8, 0.9, 0.1], dtype=np.float32)

    effective, accepted = effective_predictions(labels, scores, threshold=0.5)

    assert effective.tolist() == ["CREATE_TASK", "ABSTAIN", "ABSTAIN"]
    assert accepted.tolist() == [True, False, False]


def test_select_threshold_prefers_clean_separation() -> None:
    y_true = np.array(["CREATE_TASK", "PROMOTE_TASK", "ABSTAIN", "ABSTAIN"])
    predicted = np.array(["CREATE_TASK", "PROMOTE_TASK", "CREATE_TASK", "PROMOTE_TASK"])
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)

    result = select_threshold(
        y_true,
        predicted,
        scores,
        labels=["ABSTAIN", "CREATE_TASK", "PROMOTE_TASK"],
    )

    assert result.executable_accept_rate == 1.0
    assert result.hard_negative_reject_rate == 1.0
    assert result.objective > 0.9
