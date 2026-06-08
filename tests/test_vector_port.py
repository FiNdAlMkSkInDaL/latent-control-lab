import numpy as np

from neural_native.app.kernel import TaskFlowKernel
from neural_native.app.state import Action
from neural_native.app.vector_port import RouteDecision, VectorActionPort


class FixedRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    def predict(self, z: np.ndarray) -> RouteDecision:
        assert z.shape == (4,)
        return self.decision


def test_vector_port_dispatches_accepted_action() -> None:
    decision = RouteDecision(
        action=Action.CREATE_TASK,
        label="CREATE_TASK",
        confidence=0.95,
        margin=0.5,
        ood_score=0.0,
        accepted=True,
    )
    kernel = TaskFlowKernel()
    port = VectorActionPort(FixedRouter(decision), kernel)
    result = port.ingest(np.ones(4), raw_text="add a task")
    assert result["status"] == "ok"
    assert len(kernel.state.backlog) == 1


def test_vector_port_abstains_when_rejected() -> None:
    decision = RouteDecision(
        action=Action.CREATE_TASK,
        label="CREATE_TASK",
        confidence=0.2,
        margin=0.01,
        ood_score=99.0,
        accepted=False,
    )
    kernel = TaskFlowKernel()
    port = VectorActionPort(FixedRouter(decision), kernel)
    result = port.ingest(np.ones(4), raw_text="nonsense")
    assert result["status"] == "abstained"
    assert len(kernel.state.backlog) == 0
