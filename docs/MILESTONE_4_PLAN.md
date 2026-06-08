# Milestone 4 Plan

## What Milestone 3 Proved

Milestone 3 proved that the zero-generation route works end to end with a real
Hugging Face causal LM:

```text
natural language -> tokenizer -> frozen LM forward pass -> pre-lm_head hook
-> vector -> linear probe -> gate -> VectorActionPort -> TaskFlowKernel
```

It also proved that the project can produce honest evaluation artifacts:
dataset leakage audit, hard eval, prompt robustness, layer sweep, data
efficiency, OOD comparison, and text baselines.

## Weaknesses Remaining

- The v1 synthetic dataset has normalized duplicates, split overlap,
  template-family leakage, and cross-split near-duplicates.
- Hard-eval ABSTAIN recall is weak, especially for negation, compound commands,
  and task-adjacent near misses.
- Prompt transfer is weak: probes trained on the default router prompt collapse
  toward `ABSTAIN` under alternate prompt templates.
- OOD methods need calibration before they should influence the runtime router.
- TF-IDF is competitive on the current small synthetic benchmark, so the
  project should claim architectural novelty rather than benchmark dominance.

## What Milestone 4 Will Improve

1. Generate a cleaner v2 dataset with split-isolated template families,
   stronger hard negatives, and no normalized duplicates.
2. Train and evaluate a v2 latent router with `distilgpt2` while preserving v1
   artifacts for comparison.
3. Train a prompt-augmented probe across several prompt templates and measure
   prompt robustness directly.
4. Add validation-selected router calibration, including Mahalanobis and
   entropy-based scoring, and only recommend gates supported by metrics.
5. Refresh baselines on v2 and compare latent, prompt-augmented, calibrated,
   TF-IDF, random, and majority classifiers.
6. Package the work for a reviewer: Colab Gemma instructions, portfolio
   narrative, interview talk track, reproducibility notes, CI, and README polish.

## What Will Not Be Attempted

- No GUI or front-end.
- No arbitrary filesystem, shell, network, browser, or OS-control actions.
- No expansion of the TaskFlow app beyond the existing bounded action enum.
- No claim that Gemma was run locally unless it is actually executed.
- No claim that latent routing beats text baselines on every small synthetic
  benchmark.
