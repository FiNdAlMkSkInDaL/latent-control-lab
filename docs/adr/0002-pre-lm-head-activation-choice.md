# ADR 0002: Pre-lm_head activation choice

## Decision

The default feature vector is the final non-padding token representation captured as the input to `model.lm_head`.

## Rationale

This vector is immediately before vocabulary unembedding and is available with a simple PyTorch forward hook on Hugging Face causal LMs.

## Consequences

- Different checkpoints may expose different module names.
- Optional layer sweeps should compare this feature space with final-block residual activations.
