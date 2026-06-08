from neural_native.app.kernel import TaskFlowKernel
from neural_native.app.state import Action, ActionContext


def ctx() -> ActionContext:
    return ActionContext(raw_text="test", confidence=1.0, vector_norm=1.0)


def test_create_task_changes_backlog() -> None:
    kernel = TaskFlowKernel()
    result = kernel.execute(Action.CREATE_TASK, ctx())
    assert result["status"] == "ok"
    assert len(kernel.state.backlog) == 1
    assert kernel.state.backlog[0].status == "backlog"


def test_promote_next_task_moves_backlog_to_active() -> None:
    kernel = TaskFlowKernel()
    kernel.execute(Action.CREATE_TASK, ctx())
    result = kernel.execute(Action.PROMOTE_TASK, ctx())
    assert result["status"] == "ok"
    assert len(kernel.state.backlog) == 0
    assert kernel.state.active is not None
    assert kernel.state.active.status == "active"


def test_complete_active_task_moves_to_completed() -> None:
    kernel = TaskFlowKernel()
    kernel.execute(Action.CREATE_TASK, ctx())
    kernel.execute(Action.PROMOTE_TASK, ctx())
    result = kernel.execute(Action.COMPLETE_ACTIVE, ctx())
    assert result["status"] == "ok"
    assert kernel.state.active is None
    assert len(kernel.state.completed) == 1
    assert kernel.state.completed[0].status == "completed"


def test_archive_completed_moves_items_to_archive() -> None:
    kernel = TaskFlowKernel()
    kernel.execute(Action.CREATE_TASK, ctx())
    kernel.execute(Action.PROMOTE_TASK, ctx())
    kernel.execute(Action.COMPLETE_ACTIVE, ctx())
    result = kernel.execute(Action.ARCHIVE_COMPLETED, ctx())
    assert result["status"] == "ok"
    assert result["moved"] == 1
    assert len(kernel.state.completed) == 0
    assert len(kernel.state.archive) == 1
    assert kernel.state.archive[0].status == "archived"


def test_toggle_focus_mode_flips_boolean() -> None:
    kernel = TaskFlowKernel()
    assert kernel.state.focus_mode is False
    kernel.execute(Action.TOGGLE_FOCUS_MODE, ctx())
    assert kernel.state.focus_mode is True
    kernel.execute(Action.TOGGLE_FOCUS_MODE, ctx())
    assert kernel.state.focus_mode is False


def test_abstain_is_noop() -> None:
    kernel = TaskFlowKernel()
    result = kernel.execute(Action.ABSTAIN, ctx())
    assert result["status"] == "abstained"
    assert kernel.state.next_id == 1
