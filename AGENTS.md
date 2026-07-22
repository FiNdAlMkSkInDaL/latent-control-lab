# Notes for coding agents in this repo

This project demos **zero-generation** action routing: frozen LM activations → probe → typed action. The kernel never sees free-form model text.

## Hard rules

- Do **not** call `model.generate()` on the action path.
- Do **not** parse user text with keywords/JSON/SQL/shell to choose actions.
- Keep the grid kernel sandboxed and deterministic.
- Keep the LM frozen; train only the probe.
- Keep the `ABSTAIN` route.

## Path

```text
text → tokenizer → frozen LM → hook → vector → probe → gate → enum → kernel
```

Anything outside that for demos should be labeled analysis-only.

## Tests

```bash
pytest
```

Prefer small modules, type hints, and tests for state + routing.
