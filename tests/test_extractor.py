import numpy as np
import pytest


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_extract_vectors_uses_forward_hook_final_token() -> None:
    import torch
    from torch import nn

    from neural_native.llm.extractor import extract_vectors
    from neural_native.llm.hooks import PreLMHeadActivationTap

    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
            assert padding is True
            assert truncation is True
            assert return_tensors == "pt"

            lengths = [min(max(1, len(text.split()) // 4), max_length) for text in texts]
            width = max(lengths)
            input_ids = torch.zeros((len(texts), width), dtype=torch.long)
            attention_mask = torch.zeros((len(texts), width), dtype=torch.long)
            for row, length in enumerate(lengths):
                input_ids[row, :length] = torch.arange(1, length + 1)
                attention_mask[row, :length] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, attention_mask, use_cache=False):
            assert use_cache is False
            token_value = input_ids.float().unsqueeze(-1)
            mask_value = attention_mask.float().unsqueeze(-1)
            hidden = torch.cat(
                [
                    token_value,
                    token_value + 10.0,
                    mask_value,
                    token_value * 2.0,
                ],
                dim=-1,
            )
            return self.lm_head(hidden)

        def generate(self, *args, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError("extract_vectors must not call generate()")

    model = FakeLM()
    tap = PreLMHeadActivationTap(model)
    try:
        vectors = extract_vectors(
            ["short request", "a longer request with several more tokens"],
            FakeTokenizer(),
            model,
            tap,
            batch_size=2,
        )
    finally:
        tap.close()

    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert vectors[0, 0] == 5.0
    assert vectors[1, 0] == 6.0
