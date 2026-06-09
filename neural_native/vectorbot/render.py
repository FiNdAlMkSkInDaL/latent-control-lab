from __future__ import annotations

from typing import Any

from neural_native.vectorbot.state import VectorBotState


def render_ascii(state: VectorBotState | dict[str, Any]) -> str:
    summary = state.to_summary() if isinstance(state, VectorBotState) else dict(state)
    width = int(summary["width"])
    height = int(summary["height"])
    bot_x = int(summary["x"])
    bot_y = int(summary["y"])
    light_on = bool(summary["light_on"])
    mode = str(summary["mode"])
    step_count = int(summary["step_count"])

    horizontal = "+" + "+".join("---" for _ in range(width)) + "+"
    rows = [
        f"VectorBot | step={step_count} | light={'ON' if light_on else 'OFF'} | mode={mode}",
        horizontal,
    ]
    for y in range(height):
        cells = []
        for x in range(width):
            if x == bot_x and y == bot_y:
                cells.append(" * " if light_on else " B ")
            else:
                cells.append(" . ")
        rows.append("|" + "|".join(cells) + "|")
        rows.append(horizontal)
    return "\n".join(rows)
