# ADR 0001: Zero-generation routing

## Decision

The app-action route must use hidden-state activations and a linear projection layer, not generated tool-call text.

## Rationale

The project thesis is that an application can consume LLM latent vectors directly, reducing reliance on text generation and parser robustness.

## Consequences

- The LLM does not need to generate action names.
- The probe and prompt template become important artifacts.
- OOD rejection is mandatory because there is no parser-level validation step.
