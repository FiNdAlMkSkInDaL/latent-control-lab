from __future__ import annotations

from neural_native.llm.extractor import PROMPT_TEMPLATE, format_prompt


def test_format_prompt_uses_default_router_template() -> None:
    formatted = format_prompt("  create a task  ")

    assert formatted == PROMPT_TEMPLATE.format(text="create a task")
    assert "create a task" in formatted


def test_format_prompt_accepts_custom_template() -> None:
    assert format_prompt("create a task", "{text}") == "create a task"
    assert (
        format_prompt("archive done", "Request: {text}\nVector:")
        == "Request: archive done\nVector:"
    )
