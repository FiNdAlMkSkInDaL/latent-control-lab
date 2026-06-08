from __future__ import annotations

import os

import pytest


def llm_dependencies_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not llm_dependencies_available(), reason="torch/transformers not installed")
@pytest.mark.real_model
def test_tiny_hf_causal_lm_pre_lm_head_hook_without_generate(monkeypatch) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neural_native.llm.extractor import extract_vectors
    from neural_native.llm.hooks import PreLMHeadActivationTap

    model_id = os.getenv("NN_TEST_MODEL_ID", "sshleifer/tiny-gpt2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except OSError as exc:
        pytest.skip(f"could not load Hugging Face test model {model_id}: {exc}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def forbidden_generate(*args, **kwargs):
        raise AssertionError("real-model routing test must not call model.generate()")

    monkeypatch.setattr(model, "generate", forbidden_generate)

    tap = PreLMHeadActivationTap(model)
    try:
        enc = tokenizer(
            ["create a task", "finish the active task"],
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            _ = model(**enc, use_cache=False)
    finally:
        tap.close()

    assert tap.hidden is not None
    assert tuple(tap.hidden.shape[:2]) == tuple(enc["input_ids"].shape)
    assert tap.hidden.ndim == 3
    assert tap.hidden.shape[-1] == model.config.hidden_size

    tap = PreLMHeadActivationTap(model)
    try:
        vectors = extract_vectors(
            ["create a task"],
            tokenizer,
            model,
            tap,
            batch_size=1,
            max_length=64,
        )
    finally:
        tap.close()

    assert vectors.shape == (1, model.config.hidden_size)
