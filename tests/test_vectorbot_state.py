from neural_native.vectorbot.kernel import VectorBotKernel
from neural_native.vectorbot.render import render_ascii
from neural_native.vectorbot.state import VectorBotAction, VectorBotActionContext


def ctx() -> VectorBotActionContext:
    return VectorBotActionContext(raw_text="audit only", confidence=0.9, vector_norm=1.2)


def test_vectorbot_moves_and_reports_diff() -> None:
    kernel = VectorBotKernel(width=5, height=5)
    result = kernel.execute(VectorBotAction.MOVE_UP, ctx())
    assert result["status"] == "ok"
    assert kernel.state.y == 1
    assert result["diff"]["y"] == {"before": 2, "after": 1}
    assert kernel.state.action_history[-1]["action"] == "MOVE_UP"


def test_vectorbot_boundary_movement_does_not_crash_or_leave_grid() -> None:
    kernel = VectorBotKernel(width=5, height=5)
    for _ in range(10):
        kernel.execute(VectorBotAction.MOVE_UP, ctx())
        kernel.execute(VectorBotAction.MOVE_LEFT, ctx())
    assert kernel.state.x == 0
    assert kernel.state.y == 0


def test_vectorbot_toggle_light_and_reset_behavior() -> None:
    kernel = VectorBotKernel(width=5, height=5)
    kernel.execute(VectorBotAction.TOGGLE_LIGHT, ctx())
    assert kernel.state.light_on is True
    assert kernel.state.mode == "lit"
    kernel.execute(VectorBotAction.MOVE_RIGHT, ctx())
    result = kernel.execute(VectorBotAction.RESET, ctx())
    assert result["status"] == "ok"
    assert kernel.state.x == 2
    assert kernel.state.y == 2
    assert kernel.state.light_on is False
    assert kernel.state.step_count == 0
    assert kernel.state.action_history[-1]["action"] == "RESET"


def test_vectorbot_abstain_is_noop() -> None:
    kernel = VectorBotKernel()
    before = kernel.snapshot()
    result = kernel.execute(VectorBotAction.ABSTAIN, ctx())
    assert result["status"] == "abstained"
    assert kernel.snapshot() == before


def test_vectorbot_ascii_renderer_contains_grid_and_bot() -> None:
    rendered = render_ascii(VectorBotKernel().snapshot())
    assert "VectorBot" in rendered
    assert " B " in rendered
    assert "+---+---+---+---+---+" in rendered
