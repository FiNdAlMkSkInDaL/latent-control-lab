from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PROMPT_TEMPLATE = """You are a latent action router for a sandboxed task controller.

User request:
{text}

Represent the intended controller action:"""


def format_prompt(text: str, prompt_template: str = PROMPT_TEMPLATE) -> str:
    return prompt_template.format(text=text.strip())


def get_input_device(model: Any):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_vectors(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    tap: Any,
    *,
    batch_size: int = 8,
    max_length: int = 160,
    prompt_template: str = PROMPT_TEMPLATE,
) -> np.ndarray:
    """
    Extract one final-token activation vector per input text.

    The function performs only forward passes and never calls model.generate().
    """

    import torch

    if not texts:
        raise ValueError("texts must contain at least one example")

    device = get_input_device(model)
    vectors: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = [
                format_prompt(t, prompt_template)
                for t in texts[start : start + batch_size]
            ]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}

            tap.clear()
            _ = model(**enc, use_cache=False)

            if tap.hidden is None:
                raise RuntimeError("Activation hook did not fire")

            hidden = tap.hidden
            attention_mask = enc["attention_mask"]
            last_token_idx = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

            z = hidden[batch_idx, last_token_idx, :]
            vectors.append(z.float().cpu().numpy())

    return np.concatenate(vectors, axis=0)
