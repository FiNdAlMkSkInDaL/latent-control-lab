# Portfolio Narrative

## Project Pitch

Neural-Native Software is a prototype action router that controls a sandboxed
task state machine from frozen LLM hidden states instead of generated commands.
The core idea is to replace the usual `LLM -> tool JSON -> parser -> API` path
with `LLM forward pass -> activation hook -> linear probe -> typed action`.

## System Architecture

```text
user text -> tokenizer -> frozen causal LM -> pre-lm_head hook -> vector
-> linear probe -> OOD gate -> Action enum -> TaskFlowKernel
```

The app has only five executable actions plus `ABSTAIN`; no arbitrary shell,
filesystem, network, or OS-control action exists.

## What Makes It Unusual

Most agent demos ask a model to generate structured text and then parse it. This
project routes directly from a latent activation vector into a typed state
transition. That makes the experiment closer to a neural-native interface than a
text-command API wrapper.

## What Was Verified

- Real `distilgpt2` hidden-state extraction: `X=(480, 768)` for V2.
- No `model.generate()` in the production routing path.
- End-to-end demo: natural language to hook to probe to `TaskFlowKernel`.
- V2 dataset audit: zero normalized duplicates, zero split overlap, zero
  template-family leakage, zero cross-split near-duplicate pairs.
- Prompt augmentation across five prompt templates.
- Calibrated OOD gates selected from validation data only.

## What Failed Or Was Weak

- V1 synthetic accuracy was inflated by dataset leakage.
- Prompt transfer was poor before prompt augmentation.
- V2 default gates became overly cautious on executable hard paraphrases.
- TF-IDF remains competitive on the small synthetic/hard benchmark.
- Gemma was documented for Colab but not run locally.

## How Weaknesses Were Addressed

- Built `intent_dataset_v2.csv` with stronger hard negatives and cleaner splits.
- Added hard eval, prompt augmentation, model comparison, calibration, and
  refreshed baselines.
- Reported tradeoffs instead of hiding them: calibration improves hard accuracy
  but still does not solve executable recall.

## Final Technical Highlights

- PyTorch forward hooks over Hugging Face causal LMs.
- Frozen-LM feature extraction with no token generation.
- scikit-learn linear probes and serialized routing bundles.
- Validation-selected OOD calibration with documented objective.
- CI-friendly tests that avoid large model downloads by default.

## Limitations

This is a bounded research prototype, not a deployable agent. The action space
is tiny, the dataset is small, and text baselines are strong. The strongest claim
is architectural feasibility with honest evaluation, not benchmark supremacy.

## Future Work

Run Gemma in Colab, collect larger human-authored hard negatives, validate
Mahalanobis OOD on independent data, and compare richer probes while preserving
the zero-generation invariant.
