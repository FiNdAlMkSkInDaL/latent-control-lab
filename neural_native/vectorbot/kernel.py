from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neural_native.vectorbot.state import (
    VectorBotAction,
    VectorBotActionContext,
    initial_state,
)


class VectorBotKernel:
    """
    Deterministic sandboxed grid-world state machine.

    The kernel accepts only typed enum actions. It never reads natural language
    to decide behavior and exposes no filesystem, shell, network, or OS actions.
    """

    def __init__(self, *, width: int = 5, height: int = 5) -> None:
        self.state = initial_state(width=width, height=height)

    def snapshot(self) -> dict[str, Any]:
        return self.state.to_summary()

    def execute(self, action: VectorBotAction, ctx: VectorBotActionContext) -> dict[str, Any]:
        before = self.snapshot()
        dispatch: dict[VectorBotAction, Callable[[], None]] = {
            VectorBotAction.MOVE_UP: self._move_up,
            VectorBotAction.MOVE_DOWN: self._move_down,
            VectorBotAction.MOVE_LEFT: self._move_left,
            VectorBotAction.MOVE_RIGHT: self._move_right,
            VectorBotAction.TOGGLE_LIGHT: self._toggle_light,
            VectorBotAction.RESET: self._reset,
        }

        if action in {VectorBotAction.ABSTAIN, VectorBotAction.NO_OP}:
            return {
                "status": "abstained",
                "action": action.name,
                "accepted": False,
                "reason": "router rejected input",
                "confidence": ctx.confidence,
                "vector_norm": ctx.vector_norm,
                "before": before,
                "after": before,
                "diff": {},
            }

        handler = dispatch.get(action)
        if handler is None:
            return {
                "status": "rejected",
                "action": str(action),
                "accepted": False,
                "reason": "unknown VectorBot action",
                "before": before,
                "after": before,
                "diff": {},
            }

        handler()
        self._record(action, ctx)
        after = self.snapshot()
        return {
            "status": "ok",
            "action": action.name,
            "accepted": True,
            "before": before,
            "after": after,
            "diff": _state_diff(before, after),
        }

    def _move_up(self) -> None:
        self.state.y = max(0, self.state.y - 1)
        self.state.mode = "moving"
        self._append_trail()

    def _move_down(self) -> None:
        self.state.y = min(self.state.height - 1, self.state.y + 1)
        self.state.mode = "moving"
        self._append_trail()

    def _move_left(self) -> None:
        self.state.x = max(0, self.state.x - 1)
        self.state.mode = "moving"
        self._append_trail()

    def _move_right(self) -> None:
        self.state.x = min(self.state.width - 1, self.state.x + 1)
        self.state.mode = "moving"
        self._append_trail()

    def _toggle_light(self) -> None:
        self.state.light_on = not self.state.light_on
        self.state.mode = "lit" if self.state.light_on else "idle"

    def _reset(self) -> None:
        width = self.state.width
        height = self.state.height
        self.state = initial_state(width=width, height=height)

    def _append_trail(self) -> None:
        pos = (self.state.x, self.state.y)
        if not self.state.trail or self.state.trail[-1] != pos:
            self.state.trail.append(pos)

    def _record(self, action: VectorBotAction, ctx: VectorBotActionContext) -> None:
        if action is not VectorBotAction.RESET:
            self.state.step_count += 1
        entry = {
            "step": self.state.step_count,
            "action": action.name,
            "x": self.state.x,
            "y": self.state.y,
            "light_on": self.state.light_on,
            "mode": self.state.mode,
            "confidence": ctx.confidence,
            "vector_norm": ctx.vector_norm,
        }
        self.state.action_history.append(entry)


def _state_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key, before_value in before.items():
        after_value = after[key]
        if before_value != after_value:
            if key == "action_history":
                diff["action_history_length"] = {
                    "before": len(before_value),
                    "after": len(after_value),
                }
            else:
                diff[key] = {"before": before_value, "after": after_value}
    return diff
