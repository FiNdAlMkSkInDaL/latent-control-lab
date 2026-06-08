from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Action(str, Enum):
    CREATE_TASK = "create_task"
    PROMOTE_TASK = "promote_task"
    COMPLETE_ACTIVE = "complete_active"
    ARCHIVE_COMPLETED = "archive_completed"
    TOGGLE_FOCUS_MODE = "toggle_focus_mode"
    ABSTAIN = "abstain"


LABEL_TO_ACTION: dict[str, Action] = {
    "CREATE_TASK": Action.CREATE_TASK,
    "PROMOTE_TASK": Action.PROMOTE_TASK,
    "COMPLETE_ACTIVE": Action.COMPLETE_ACTIVE,
    "ARCHIVE_COMPLETED": Action.ARCHIVE_COMPLETED,
    "TOGGLE_FOCUS_MODE": Action.TOGGLE_FOCUS_MODE,
    "ABSTAIN": Action.ABSTAIN,
}

ACTION_TO_LABEL: dict[Action, str] = {value: key for key, value in LABEL_TO_ACTION.items()}


@dataclass(slots=True)
class Task:
    id: int
    title: str
    source_utterance: str
    status: str = "backlog"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


@dataclass(slots=True)
class AppState:
    backlog: list[Task] = field(default_factory=list)
    active: Task | None = None
    completed: list[Task] = field(default_factory=list)
    archive: list[Task] = field(default_factory=list)
    focus_mode: bool = False
    next_id: int = 1


@dataclass(slots=True)
class ActionContext:
    raw_text: str
    confidence: float
    vector_norm: float
