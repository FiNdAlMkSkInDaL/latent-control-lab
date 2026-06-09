from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_native.bridge.router import LinearProbeRouter, RouterThresholds, thresholds_from_json
from neural_native.vectorbot.state import LABEL_TO_ACTION, VectorBotAction
from neural_native.vectorbot.vector_port import VectorBotRouteDecision


class VectorBotLinearProbeRouter:
    """Adapter that maps probe labels to VectorBot actions."""

    def __init__(
        self,
        bundle_path: str | Path,
        thresholds: RouterThresholds | None = None,
    ) -> None:
        self.base = LinearProbeRouter(
            str(bundle_path),
            thresholds=thresholds,
            label_to_action=LABEL_TO_ACTION,
            abstain_action=VectorBotAction.ABSTAIN,
        )
        self.classes = self.base.classes
        self.metadata = self.base.metadata

    def predict(self, z: np.ndarray) -> VectorBotRouteDecision:
        decision = self.base.predict(z)
        return VectorBotRouteDecision(
            action=decision.action,
            label=decision.label,
            confidence=decision.confidence,
            margin=decision.margin,
            ood_score=decision.ood_score,
            accepted=decision.accepted,
            top_probabilities=self.base.top_probabilities(z, k=3),
        )


__all__ = ["RouterThresholds", "VectorBotLinearProbeRouter", "thresholds_from_json"]
