from __future__ import annotations

from typing import Any


class PreLMHeadActivationTap:
    """
    Captures the hidden state passed into `model.lm_head`.

    For decoder-only causal LMs, this is the representation immediately before
    unembedding into vocabulary logits. Shape is usually [batch, seq, hidden_size].
    """

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "lm_head"):
            raise ValueError("Expected AutoModelForCausalLM-like object with .lm_head")
        self.hidden = None
        self.handle = model.lm_head.register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module, output
        if not inputs:
            raise RuntimeError("lm_head hook received no inputs")
        self.hidden = inputs[0].detach()

    def clear(self) -> None:
        self.hidden = None

    def close(self) -> None:
        self.handle.remove()


class FinalBlockResidualTap:
    """
    Captures final transformer block output, when model.model.layers exists.

    This is useful for comparing raw final-block residual features with the
    post-final-norm pre-lm_head representation.
    """

    def __init__(self, model: Any) -> None:
        try:
            block = model.model.layers[-1]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Expected model.model.layers[-1] to exist") from exc

        self.hidden = None
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        if isinstance(output, tuple):
            self.hidden = output[0].detach()
        else:
            self.hidden = output.detach()

    def clear(self) -> None:
        self.hidden = None

    def close(self) -> None:
        self.handle.remove()


def _block_container(model: Any) -> Any:
    candidates = (
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
    )
    for attrs in candidates:
        current = model
        try:
            for attr in attrs:
                current = getattr(current, attr)
            if len(current) > 0:
                return current
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(
        "Could not find transformer blocks. Tried model.transformer.h, "
        "model.model.layers, model.gpt_neox.layers, and model.transformer.blocks."
    )


def resolve_transformer_block(model: Any, index: int | str) -> tuple[str, Any]:
    blocks = _block_container(model)
    n_blocks = len(blocks)
    if isinstance(index, str):
        if index == "early":
            resolved = 0
        elif index == "middle":
            resolved = n_blocks // 2
        elif index == "final":
            resolved = n_blocks - 1
        else:
            raise ValueError(f"Unknown layer alias: {index}")
    else:
        resolved = index

    if resolved < 0:
        resolved = n_blocks + resolved
    if resolved < 0 or resolved >= n_blocks:
        raise ValueError(f"Layer index {index!r} is outside model depth {n_blocks}")
    return f"block_{resolved}", blocks[resolved]


class TransformerBlockActivationTap:
    """Captures a transformer block output for layer-sweep evaluation."""

    def __init__(self, model: Any, index: int | str) -> None:
        self.layer_name, block = resolve_transformer_block(model, index)
        self.hidden = None
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        if isinstance(output, tuple):
            self.hidden = output[0].detach()
        else:
            self.hidden = output.detach()

    def clear(self) -> None:
        self.hidden = None

    def close(self) -> None:
        self.handle.remove()
