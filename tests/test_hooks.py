import pytest


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_pre_lm_head_hook_captures_input() -> None:
    import torch
    from torch import nn

    from neural_native.llm.hooks import PreLMHeadActivationTap

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(4, 8)

        def forward(self, x):
            return self.lm_head(x)

    model = TinyLM()
    tap = PreLMHeadActivationTap(model)
    x = torch.randn(2, 3, 4)
    _ = model(x)
    assert tap.hidden is not None
    assert tuple(tap.hidden.shape) == (2, 3, 4)
    tap.close()


def test_pre_lm_head_hook_requires_lm_head() -> None:
    from neural_native.llm.hooks import PreLMHeadActivationTap

    class NotLM:
        pass

    with pytest.raises(ValueError):
        PreLMHeadActivationTap(NotLM())


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_final_block_residual_hook_captures_output() -> None:
    import torch
    from torch import nn

    from neural_native.llm.hooks import FinalBlockResidualTap

    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

        def forward(self, x):
            return self.layers[-1](x)

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyBackbone()

        def forward(self, x):
            return self.model(x)

    model = TinyLM()
    tap = FinalBlockResidualTap(model)
    x = torch.randn(2, 3, 4)
    _ = model(x)
    assert tap.hidden is not None
    assert tuple(tap.hidden.shape) == (2, 3, 4)
    tap.close()


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_transformer_block_hook_resolves_gpt2_style_blocks() -> None:
    import torch
    from torch import nn

    from neural_native.llm.hooks import TransformerBlockActivationTap

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])

        def forward(self, x):
            for block in self.h:
                x = block(x)
            return x

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = TinyTransformer()

        def forward(self, x):
            return self.transformer(x)

    model = TinyLM()
    tap = TransformerBlockActivationTap(model, "middle")
    x = torch.randn(2, 3, 4)
    _ = model(x)
    assert tap.layer_name == "block_1"
    assert tap.hidden is not None
    assert tuple(tap.hidden.shape) == (2, 3, 4)
    tap.close()
