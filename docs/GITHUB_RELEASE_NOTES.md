# GitHub Release Notes

## Project Summary

This release packages Neural Native Software as a portfolio-ready research MVP for zero-generation latent action routing. The system routes natural-language task requests through a frozen Hugging Face causal LM forward pass, captures a hidden activation before `lm_head`, classifies that vector with a lightweight probe, applies OOD gates, and executes only a bounded enum action in a deterministic toy task kernel.

The key claim is narrow and testable: a small end-to-end app can route actions from frozen LLM activations without generating commands, parsing JSON tool calls, or using text regexes to decide the action.

## Major Implemented Features

- Deterministic task application with a bounded `Action` enum and `ABSTAIN` route.
- Latent router interface that receives vectors and returns action decisions with confidence, margin, and OOD scores.
- Hugging Face activation extraction for causal LMs using a pre-`lm_head` forward hook.
- Dataset generation, feature extraction, probe training, OOD evaluation, calibration, prompt augmentation, and baseline comparison scripts.
- CLI/demo path showing natural language routed through hidden-state features into app actions.
- Tiny real-model integration test that verifies captured activation shape and guards against `model.generate()` usage.
- Portfolio artifacts for metrics, confusion matrices, thresholds, routes, calibration, baseline comparison, and demo outputs.
- Public-facing documentation for reproduction, real-model execution, demo presentation, and final project summary.

## Verified Invariants

- Production routing follows:

  ```text
  Text input -> tokenizer -> frozen LLM forward pass -> hook -> vector -> probe -> gate -> enum action -> TaskFlowKernel
  ```

- The production routing path does not call `model.generate()`.
- The action decision is not made by generated JSON, generated commands, keyword matching, regex parsing, or text parsers.
- The LLM remains frozen; only probe/projection-style classifiers are trained.
- The task kernel remains sandboxed and deterministic.
- The `ABSTAIN` route and OOD gates are preserved.
- Tests cover state mutations, bridge routing behavior, CLI behavior, real-model hook behavior, and no-generate guards.

## Results

| Evaluation | Model / Method | Accuracy | Macro-F1 | Notes |
| --- | --- | ---: | ---: | --- |
| V2 in-distribution test | `distilgpt2` latent probe | 1.000 | 1.000 | Balanced held-out synthetic test split |
| V2 hard eval | latent probe, default prompt | 0.607 | 0.349 | Stress set with paraphrases, refusals, negations, and OOD |
| V2 hard eval | calibrated latent probe | 0.647 | 0.483 | Uses calibrated thresholds |
| Prompt-augmented hard eval | best prompt variant | 0.676 | 0.553 | Terse prompt variant |
| Baseline hard eval | TF-IDF logistic regression | 0.665 | 0.587 | Useful non-neural reference point |
| Baseline hard eval | random | 0.222 | 0.130 | Sanity baseline |
| Baseline hard eval | majority | 0.545 | 0.118 | Mostly predicts `ABSTAIN` |
| OOD validation | negative Mahalanobis score | AUROC 0.990 | AUPRC 0.982 | Strong separation on the evaluated OOD validation setup |

Demo artifacts show five accepted app actions and one `ABSTAIN` example routed through the latent pipeline:

- `CREATE_TASK`
- `PROMOTE_TASK`
- `COMPLETE_ACTIVE`
- `ARCHIVE_COMPLETED`
- `TOGGLE_FOCUS_MODE`
- `ABSTAIN`

## Limitations

- The task application is intentionally small; this is a bounded research MVP, not a general automation agent.
- The strongest classifier baseline remains competitive with or better than the latent probe on the hard evaluation set.
- V2 in-distribution metrics are synthetic and should not be interpreted as broad natural-language understanding.
- OOD calibration is promising but dataset-specific and should be revalidated before expanding the action space.
- GPU-scale Gemma execution is documented as a reproduction path, but this packaged release does not claim local Gemma benchmark results.
- The current system demonstrates routing and safety boundaries, not production reliability.

## Release Readiness

This repository is suitable for publication as a research portfolio project when presented with the conservative claim above. The code, tests, documentation, and small artifacts support the milestone narrative while avoiding claims of general agentic ability or production-grade robustness.
