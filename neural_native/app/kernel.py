from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from neural_native.app.state import Action, ActionContext, AppState, Task


class TaskFlowKernel:
    """
    Sandboxed task-controller state machine.

    This class deliberately accepts typed action enums. It does not parse natural
    language, JSON, commands, or model-generated text.
    """

    def __init__(self) -> None:
        self.state = AppState()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.state)

    def execute(self, action: Action, ctx: ActionContext) -> dict[str, Any]:
        dispatch: dict[Action, Callable[[ActionContext], dict[str, Any]]] = {
            Action.CREATE_TASK: self.create_task,
            Action.PROMOTE_TASK: self.promote_next_task,
            Action.COMPLETE_ACTIVE: self.complete_active_task,
            Action.ARCHIVE_COMPLETED: self.archive_completed,
            Action.TOGGLE_FOCUS_MODE: self.toggle_focus_mode,
        }

        if action == Action.ABSTAIN:
            return {
                "status": "abstained",
                "reason": "router rejected input",
                "confidence": ctx.confidence,
            }

        handler = dispatch.get(action)
        if handler is None:
            return {"status": "rejected", "reason": f"unknown action: {action}"}

        return handler(ctx)

    def create_task(self, ctx: ActionContext) -> dict[str, Any]:
        task = Task(
            id=self.state.next_id,
            title=f"Task {self.state.next_id}",
            source_utterance=ctx.raw_text,
        )
        self.state.next_id += 1
        self.state.backlog.append(task)
        return {"status": "ok", "action": Action.CREATE_TASK.name, "task_id": task.id}

    def promote_next_task(self, ctx: ActionContext) -> dict[str, Any]:
        del ctx
        if self.state.active is not None:
            return {"status": "blocked", "reason": "active task already exists"}
        if not self.state.backlog:
            return {"status": "blocked", "reason": "no backlog tasks"}

        task = self.state.backlog.pop(0)
        task.status = "active"
        self.state.active = task
        return {"status": "ok", "action": Action.PROMOTE_TASK.name, "task_id": task.id}

    def complete_active_task(self, ctx: ActionContext) -> dict[str, Any]:
        del ctx
        if self.state.active is None:
            return {"status": "blocked", "reason": "no active task"}

        task = self.state.active
        task.status = "completed"
        self.state.completed.append(task)
        self.state.active = None
        return {"status": "ok", "action": Action.COMPLETE_ACTIVE.name, "task_id": task.id}

    def archive_completed(self, ctx: ActionContext) -> dict[str, Any]:
        del ctx
        moved = len(self.state.completed)
        for task in self.state.completed:
            task.status = "archived"
        self.state.archive.extend(self.state.completed)
        self.state.completed.clear()
        return {"status": "ok", "action": Action.ARCHIVE_COMPLETED.name, "moved": moved}

    def toggle_focus_mode(self, ctx: ActionContext) -> dict[str, Any]:
        del ctx
        self.state.focus_mode = not self.state.focus_mode
        return {
            "status": "ok",
            "action": Action.TOGGLE_FOCUS_MODE.name,
            "focus_mode": self.state.focus_mode,
        }
