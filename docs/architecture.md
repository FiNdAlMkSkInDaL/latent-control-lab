# Architecture

The current repository centers VectorBot, a sandboxed 2D grid-world controlled
by hidden-state vectors from a frozen tiny transformer.

```text
Text input -> tokenizer -> frozen LLM forward pass -> pre-lm-head hook
-> final non-padding token vector -> probe -> gate -> enum action
-> VectorBotKernel.execute()
```

## Runtime Boundary

`VectorBotVectorPort` accepts an activation vector and a raw text audit string.
The route decision comes from the router's vector prediction. Raw text is not
parsed to select actions.

## Core modules

- `neural_native/llm/loader.py` loads CPU-friendly causal LMs, defaulting to
  `distilgpt2`.
- `neural_native/llm/hooks.py` captures the hidden state passed into `lm_head`.
- `neural_native/llm/extractor.py` selects the final non-padding token vector.
- `neural_native/bridge/router.py` loads a scikit-learn probe bundle and applies
  confidence/OOD gates.
- `neural_native/vectorbot/kernel.py` executes only typed VectorBot enum actions.

## Invariants

- No `model.generate()` in the action route.
- No generated JSON or tool-call parsing.
- No regex or keyword action parser.
- No arbitrary shell, filesystem, network, or OS control actions.
- The model is frozen; only the probe is trained.
- `ABSTAIN` remains a first-class route.
