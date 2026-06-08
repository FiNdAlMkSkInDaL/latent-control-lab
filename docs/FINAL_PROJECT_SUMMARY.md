# Final Project Summary

## One-Paragraph Summary

Neural-Native Software is a research-engineering MVP that routes natural
language into a bounded task application without generated commands. Instead of
asking an LLM to emit JSON or a tool call, the system runs a frozen Hugging Face
causal LM in a forward pass, captures the pre-`lm_head` hidden state with a
PyTorch hook, classifies that activation with a lightweight linear probe, applies
OOD gates, and executes only typed enum actions in a sandboxed `TaskFlowKernel`.

## Architecture

```text
user text -> tokenizer -> frozen causal LM -> pre-lm_head hook
-> final-token vector -> linear probe -> OOD gate -> Action enum
-> TaskFlowKernel.execute(Action, ctx)
```

The core invariant is that no generated text decides the action.

## Technical Stack

| Area | Stack |
|---|---|
| Language/runtime | Python 3.12 locally, Python 3.11 in CI |
| LLM interface | Hugging Face `transformers`, `torch`, `accelerate` |
| Feature extraction | PyTorch forward hooks, frozen causal LM forward passes |
| Probe | scikit-learn `StandardScaler + LogisticRegression` |
| Data/artifacts | pandas, numpy, joblib, JSON/CSV artifacts |
| Tests/lint | pytest, ruff, GitHub Actions |

## Experiments Run

- V1 real-model MVP with `distilgpt2`.
- Dataset leakage and near-duplicate audit.
- Human-style hard evaluation set.
- V2 leakage-reduced dataset generation and audit.
- V2 `distilgpt2` feature extraction and probe training.
- Hard-eval routing with default and calibrated gates.
- Prompt augmentation across five prompt templates.
- OOD scoring and calibration, including Mahalanobis and entropy.
- TF-IDF, random, and majority baselines.
- Small-model comparison with `distilgpt2`, attempted `gpt2`, and
  `sshleifer/tiny-gpt2` plumbing baseline.

## Final Metrics

| Metric | Result |
|---|---:|
| V2 feature matrix | `X=(480, 768)` |
| V2 dataset normalized duplicates | 0 |
| V2 template-family split overlap | 0 |
| V2 synthetic test accuracy / macro-F1 | 1.000 / 1.000 |
| V2 hard eval accuracy / macro-F1 | 0.607 / 0.349 |
| V2 calibrated hard eval accuracy / macro-F1 | 0.647 / 0.483 |
| Prompt-augmented hard eval accuracy / macro-F1 | 0.651 / 0.516 |
| V2 calibrated ABSTAIN precision / recall | 0.633 / 0.907 |
| Best advanced V2 OOD AUROC | 0.990 Mahalanobis |
| TF-IDF hard eval accuracy / macro-F1 | 0.665 / 0.587 |

## Limitations

- The app is intentionally tiny and sandboxed.
- `distilgpt2` is a small semantic model.
- The datasets are curated and still small.
- TF-IDF remains competitive on hard eval.
- Calibration improves rejection but does not solve executable hard-paraphrase
  recall.
- Gemma is documented for Colab but not claimed as executed locally.

## Future Work

1. Run the V2 pipeline with `google/gemma-2-2b-it` in Colab.
2. Collect larger human-authored hard negatives and paraphrases.
3. Validate Mahalanobis OOD on independent data.
4. Compare richer probes while keeping the LLM frozen.
5. Explore activation patching and counterfactual vector swaps.

## Recommended CV Bullets

- Built a zero-generation action router that maps frozen Hugging Face hidden
  states into typed app actions via PyTorch hooks and a linear probe, avoiding
  generated JSON/tool-call parsing.
- Rebuilt a leakage-prone synthetic benchmark into a V2 dataset with zero
  normalized duplicates and zero template-family split overlap, then verified
  the full `distilgpt2` route end to end.
- Added robust evaluation artifacts: hard negatives, prompt augmentation,
  calibrated OOD gates, text baselines, model comparison, and CI-backed tests.
