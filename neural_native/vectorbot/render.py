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


def render_grid_html(state: VectorBotState | dict[str, Any] | Any, *, cell_size: int = 52) -> str:
    """Return a self-contained HTML/CSS grid visualization with bot + optional trail.

    Designed for Streamlit via st.markdown(..., unsafe_allow_html=True).
    Shows visited trail faintly for visual interest.
    Accepts VectorBotState, summary dict, or a kernel object (for convenience).
    """
    # Normalize input: support kernel, state, or dict
    if hasattr(state, "snapshot"):
        state = state.snapshot()
    elif hasattr(state, "state") and hasattr(state.state, "to_summary"):
        state = state.state
    summary = state.to_summary() if isinstance(state, VectorBotState) else dict(state)
    width = int(summary["width"])
    height = int(summary["height"])
    bx = int(summary["x"])
    by = int(summary["y"])
    light_on = bool(summary["light_on"])
    step_count = int(summary.get("step_count", 0))
    trail = summary.get("trail", []) or []
    # Normalize trail to tuples
    trail_set = {tuple(p) for p in trail}

    cells_html = []
    for y in range(height):
        row = []
        for x in range(width):
            is_bot = (x == bx and y == by)
            visited = (x, y) in trail_set and not is_bot
            if is_bot:
                symbol = "✨" if light_on else "🤖"
                bg = "#fde047" if light_on else "#60a5fa"
                border = "#ca8a04" if light_on else "#1e40af"
                extra = 'box-shadow: 0 0 0 4px rgba(250,204,21,0.35);' if light_on else ''
                cell = (
                    f'<div style="width:{cell_size}px;height:{cell_size}px;'
                    f'background:{bg};border:3px solid {border};border-radius:6px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:22px;{extra}">{symbol}</div>'
                )
            elif visited:
                cell = (
                    f'<div style="width:{cell_size}px;height:{cell_size}px;'
                    f'background:#e0e7ff;border:1px solid #64748b;border-radius:4px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:11px;color:#64748b;">·</div>'
                )
            else:
                cell = (
                    f'<div style="width:{cell_size}px;height:{cell_size}px;'
                    f'background:#f1f5f9;border:1px solid #94a3b8;border-radius:4px;"></div>'
                )
            row.append(cell)
        cells_html.append(
            f'<div style="display:flex;gap:3px;justify-content:center;">{"".join(row)}</div>'
        )

    trail_note = f"trail length: {len(trail)}" if trail else ""
    html = (
        '<div style="font-family:ui-monospace,monospace;font-size:12px;color:#334155;margin-bottom:6px;">'
        f'VectorBot | step={step_count} | light={"ON" if light_on else "OFF"} | {trail_note}'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:3px;background:#0f172a;padding:8px;border-radius:8px;width:fit-content;">'
        + "".join(cells_html)
        + "</div>"
    )
    return html
