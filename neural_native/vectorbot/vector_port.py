from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.state import VectorBotAction, VectorBotActionContext


@dataclass(slots=True)
class VectorBotRouteDecision:
    action: VectorBotAction
    label: str
    confidence: float
    margin: float
    ood_score: float
    accepted: bool
    top_probabilities: list[dict[str, float]] = field(default_factory=list)
    full_probabilities: list[dict[str, float]] = field(default_factory=list)  # all classes, sums to ~1.0


class VectorBotRouter(Protocol):
    def predict(self, z: np.ndarray) -> VectorBotRouteDecision:
        """Map one latent vector to a VectorBot route decision."""


class VectorBotVectorPort:
    """
    Vector-facing boundary between latent routing and VectorBot.

    Raw text is kept only in the action context for audit/logging. The action is
    selected exclusively by the router decision over the supplied vector.
    """

    def __init__(self, router: VectorBotRouter, app: VectorBotKernel) -> None:
        self.router = router
        self.app = app

    def ingest(self, z: np.ndarray, raw_text: str) -> dict:
        decision = self.router.predict(z)
        ctx = VectorBotActionContext(
            raw_text=raw_text,
            confidence=decision.confidence,
            vector_norm=float(np.linalg.norm(z)),
        )
        action = decision.action if decision.accepted else VectorBotAction.ABSTAIN
        result = self.app.execute(action, ctx)
        result["route"] = {
            "label": decision.label,
            "action": decision.action.name,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "ood_score": decision.ood_score,
            "accepted": decision.accepted,
            "top_probabilities": list(decision.top_probabilities),
            "full_probabilities": list(decision.full_probabilities),
        }
        return result
