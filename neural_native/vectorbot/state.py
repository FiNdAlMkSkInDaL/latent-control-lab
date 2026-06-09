from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VectorBotAction(str, Enum):
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    TOGGLE_LIGHT = "toggle_light"
    RESET = "reset"
    ABSTAIN = "abstain"
    NO_OP = "no_op"


LABEL_TO_ACTION: dict[str, VectorBotAction] = {
    "MOVE_UP": VectorBotAction.MOVE_UP,
    "MOVE_DOWN": VectorBotAction.MOVE_DOWN,
    "MOVE_LEFT": VectorBotAction.MOVE_LEFT,
    "MOVE_RIGHT": VectorBotAction.MOVE_RIGHT,
    "TOGGLE_LIGHT": VectorBotAction.TOGGLE_LIGHT,
    "RESET": VectorBotAction.RESET,
    "ABSTAIN": VectorBotAction.ABSTAIN,
    "NO_OP": VectorBotAction.NO_OP,
}

ACTION_TO_LABEL: dict[VectorBotAction, str] = {
    action: label for label, action in LABEL_TO_ACTION.items()
}


@dataclass(slots=True)
class VectorBotActionContext:
    raw_text: str
    confidence: float
    vector_norm: float


@dataclass(slots=True)
class VectorBotState:
    width: int = 5
    height: int = 5
    x: int = 2
    y: int = 2
    light_on: bool = False
    mode: str = "idle"
    step_count: int = 0
    action_history: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "x": self.x,
            "y": self.y,
            "light_on": self.light_on,
            "mode": self.mode,
            "step_count": self.step_count,
            "action_history": list(self.action_history),
        }


def initial_state(width: int = 5, height: int = 5) -> VectorBotState:
    if width <= 0 or height <= 0:
        raise ValueError("VectorBot grid dimensions must be positive")
    return VectorBotState(width=width, height=height, x=width // 2, y=height // 2)
