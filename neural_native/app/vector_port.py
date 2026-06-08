from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from neural_native.app.kernel import TaskFlowKernel
from neural_native.app.state import Action, ActionContext


@dataclass(slots=True)
class RouteDecision:
    action: Action
    label: str
    confidence: float
    margin: float
    ood_score: float
    accepted: bool


class VectorRouter(Protocol):
    def predict(self, z: np.ndarray) -> RouteDecision:
        """Map one latent vector to a route decision."""


class VectorActionPort:
    """
    Vector-facing boundary between the latent router and the app kernel.

    This class accepts activation vectors. It does not parse text to decide the action.
    """

    def __init__(self, router: VectorRouter, app: TaskFlowKernel) -> None:
        self.router = router
        self.app = app

    def ingest(self, z: np.ndarray, raw_text: str) -> dict:
        decision = self.router.predict(z)
        ctx = ActionContext(
            raw_text=raw_text,
            confidence=decision.confidence,
            vector_norm=float(np.linalg.norm(z)),
        )

        action = decision.action if decision.accepted else Action.ABSTAIN
        result = self.app.execute(action, ctx)
        result["route"] = {
            "label": decision.label,
            "action": decision.action.name,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "ood_score": decision.ood_score,
            "accepted": decision.accepted,
        }
        return result
