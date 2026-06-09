import numpy as np

from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.state import VectorBotAction
from neural_native.vectorbot.vector_port import (
    VectorBotRouteDecision,
    VectorBotVectorPort,
)


class FixedRouter:
    def __init__(self, decision: VectorBotRouteDecision) -> None:
        self.decision = decision

    def predict(self, z: np.ndarray) -> VectorBotRouteDecision:
        assert z.shape == (3,)
        return self.decision


def test_vectorbot_port_dispatches_accepted_vector_action() -> None:
    decision = VectorBotRouteDecision(
        action=VectorBotAction.MOVE_RIGHT,
        label="MOVE_RIGHT",
        confidence=0.95,
        margin=0.4,
        ood_score=0.0,
        accepted=True,
    )
    kernel = VectorBotKernel()
    port = VectorBotVectorPort(FixedRouter(decision), kernel)
    result = port.ingest(np.ones(3), raw_text="this is audit text only")
    assert result["status"] == "ok"
    assert kernel.state.x == 3
    assert result["route"]["label"] == "MOVE_RIGHT"


def test_vectorbot_port_abstains_when_router_rejects() -> None:
    decision = VectorBotRouteDecision(
        action=VectorBotAction.MOVE_RIGHT,
        label="MOVE_RIGHT",
        confidence=0.2,
        margin=0.01,
        ood_score=99.0,
        accepted=False,
    )
    kernel = VectorBotKernel()
    before = kernel.snapshot()
    port = VectorBotVectorPort(FixedRouter(decision), kernel)
    result = port.ingest(np.ones(3), raw_text="delete my files")
    assert result["status"] == "abstained"
    assert kernel.snapshot() == before
