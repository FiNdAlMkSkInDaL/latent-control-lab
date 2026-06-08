# Repository instructions for coding agents

## Mission

This repository demonstrates **zero-generation latent action routing**. Maintain the core invariant: the runtime route from user request to app action must use a frozen LLM activation vector and a lightweight projection layer, not generated commands or parsed tool-call text.

## Hard constraints

- Do not call `model.generate()` in the production inference path.
- Do not parse user text with regexes, keyword matching, generated JSON, generated SQL, or generated shell commands to decide actions.
- Do not add arbitrary filesystem, shell, network, or OS-control actions to the toy application.
- Keep the task kernel sandboxed and deterministic.
- Keep the LLM frozen. Train only the probe/projection layer.
- Preserve the `ABSTAIN` route and OOD gates.

## Coding standards

- Prefer small, typed modules with clear boundaries.
- Add tests for state mutations and routing behavior.
- Put executable demos in `scripts/` or `neural_native/cli.py`.
- Keep notebooks optional; do not make tests depend on notebooks.
- Use `pytest`, `ruff`, and type hints.
- Document model id, feature space, prompt template, and metrics in saved artifacts.
- For delegated verification tasks, Codex must execute the required commands itself and must not ask the user to run local commands. If execution is blocked by the sandbox, record the blocker and implement the closest runnable fallback.

## Architecture boundary

```text
Text input -> tokenizer -> frozen LLM forward pass -> hook -> vector -> probe -> gate -> enum action -> TaskFlowKernel
```

Anything outside this path should be marked as experimental or evaluation-only.

## Preferred tests

- App state machine tests should be pure Python and fast.
- Hook tests may use fake modules or a tiny local model.
- Integration tests should verify that a router decision reaches the app without text parsing.
- Add a guard test or code comment ensuring `model.generate()` is not used for action routing.
