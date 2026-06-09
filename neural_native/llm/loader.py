from __future__ import annotations

import os
from typing import Any


def load_causal_lm(
    model_id: str = "distilgpt2",
    *,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """
    Load a Hugging Face causal LM and tokenizer.

    Heavy imports are kept inside this function so app and probe unit tests do not
    require torch/transformers unless LLM functionality is exercised.
    """

    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # pragma: no cover - optional dependency edge case
        BitsAndBytesConfig = None

    token = os.getenv("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "token": token,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    dtype_key = "dtype" if Version(transformers.__version__).major >= 5 else "torch_dtype"

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs[dtype_key] = torch.float16
        if use_4bit and BitsAndBytesConfig is not None:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    else:
        kwargs[dtype_key] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return tokenizer, model
