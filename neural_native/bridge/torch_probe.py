from __future__ import annotations


def build_linear_intent_probe(d_model: int, n_actions: int):
    """Factory to avoid importing torch unless this optional probe is used."""

    import torch
    from torch import nn

    class LinearIntentProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(d_model, n_actions)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            z = torch.nn.functional.layer_norm(z, z.shape[-1:])
            return self.proj(z)

    return LinearIntentProbe()
