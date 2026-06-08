# Codex Task: Build Neural-Native Software MVP

You are implementing a portfolio-grade Python MVP named **Neural-Native Software: A Zero-API Paradigm**.

## Goal

Build a system that routes natural-language user requests to a sandboxed Python state machine by:

1. running a frozen Hugging Face causal language model in a forward pass,
2. capturing the pre-`lm_head` hidden state using a PyTorch forward hook,
3. mapping that vector to an application action using a lightweight linear probe,
4. applying confidence / margin / OOD gates, and
5. executing a typed state-machine transition.

Do **not** use generated JSON, generated SQL, generated shell commands, regex command parsing, or `model.generate()` in the core inference path.

## Deliverables

### Phase 1: Toy application

Implement and test a zero-GUI task-controller kernel with these actions:

- `CREATE_TASK`
- `PROMOTE_TASK`
- `COMPLETE_ACTIVE`
- `ARCHIVE_COMPLETED`
- `TOGGLE_FOCUS_MODE`
- `ABSTAIN`

The app must expose a vector-facing port that accepts `np.ndarray` activations and dispatches only typed enum actions.

### Phase 2: Activation extraction

Implement Hugging Face loading helpers and PyTorch forward hooks:

- `PreLMHeadActivationTap` captures `inputs[0]` to `model.lm_head`.
- `FinalBlockResidualTap` optionally captures `model.model.layers[-1]` output.
- `extract_vectors()` returns one final-token vector per input text.

Use `torch.inference_mode()` and keep the base LLM frozen.

### Phase 3: Probe training

Generate a small synthetic dataset with approximately:

- 50 examples per executable action,
- 100 `ABSTAIN` examples,
- train/validation/test split metadata.

Train a scikit-learn logistic-regression probe over frozen activations. Save a `joblib` bundle containing:

- trained probe,
- label classes,
- model id,
- prompt template,
- feature space metadata,
- metrics.

### Phase 4: Inference and evaluation

Implement:

- CLI demo: text -> vector -> probe -> gate -> app state transition.
- OOD evaluation with max probability, top-1/top-2 margin, and centroid distance.
- Unit tests for app state, hooks, probe routing, and integration.

### Phase 5: Portfolio polish

Update README with:

- architecture diagram,
- quickstart,
- metrics table,
- limitations,
- safety boundary,
- CV bullet points.

## Acceptance criteria

- `make dataset` creates `data/intent_dataset.csv`.
- `make test` passes without downloading a large model.
- `python scripts/train_probe.py --features ...` trains and serializes a probe.
- Core inference path does not call `model.generate()`.
- All state mutations go through `TaskFlowKernel.execute()`.
- No text parser determines the action.
- Code is typed, modular, and suitable for a software engineering portfolio.

## Suggested implementation order

1. Finish and test `neural_native/app/*`.
2. Finish and test hook classes with a tiny fake model or tiny HF model.
3. Generate synthetic dataset.
4. Extract features with `sshleifer/tiny-gpt2` for a smoke test.
5. Train and evaluate logistic-regression probe.
6. Add real Gemma/Llama model support with quantization.
7. Add OOD evaluation and plots.
8. Polish README and docs.
