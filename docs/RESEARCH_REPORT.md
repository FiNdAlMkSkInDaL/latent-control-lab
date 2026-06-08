# Research Report

This is a historical Milestone 3 report. For the final publication summary and
V2 calibrated results, see `README.md`, `docs/FINAL_PROJECT_SUMMARY.md`, and
`docs/DEMO_RESULTS_V2.md`.

## Abstract

This repository demonstrates zero-generation latent action routing: text is
tokenized, a frozen causal LM is run in a forward pass, a PyTorch hook captures a
hidden state, and a lightweight probe maps that vector into a sandboxed task
controller. The MVP works end to end with `distilgpt2`, but Milestone 3 shows
that the original perfect synthetic score is inflated by dataset leakage and
template simplicity.

## System Architecture

```text
Text input -> tokenizer -> frozen LLM forward pass -> hook -> vector
-> linear probe -> gate -> enum action -> TaskFlowKernel
```

The production app mutation path is:

```text
VectorActionPort -> router.predict(z) -> TaskFlowKernel.execute(Action, ctx)
```

## Zero-Generation Invariant

The route path does not call `model.generate()`. It does not parse generated
JSON, generated tool calls, SQL, shell commands, regexes, or text keywords to
select app actions. The raw text is stored only as action context for audit
metadata.

## Model And Activation Site

Verified local model: `distilgpt2`.

Primary feature space: `pre_lm_head_last_token`.

Prompt template:

```text
You are a latent action router for a sandboxed task controller.

User request:
{text}

Represent the intended controller action:
```

The pre-`lm_head` hook captures `[batch, seq, hidden_size]`; for `distilgpt2` the
final vector dimension is 768.

## Dataset Construction

The synthetic dataset contains 350 rows:

- 50 examples for each executable action.
- 100 `ABSTAIN` examples.
- Stratified `train`, `validation`, and `test` metadata.

The hard eval set contains 275 rows:

- 25 human-style paraphrases per executable class.
- 150 `ABSTAIN` rows across negation, compound, near-miss, gibberish,
  unrelated, unsafe, and ambiguous subsets.

## Dataset Leakage Audit

The audit found:

- 25 normalized duplicate texts.
- 8 normalized text overlaps across train/validation/test.
- 39 template families present across multiple splits.
- 69 near-duplicate pairs at token Jaccard `>= 0.82`.
- 31 cross-split near-duplicate pairs.

This explains why `1.000` synthetic test accuracy is not credible evidence of
general language robustness.

## Probe Training Setup

The probe is a scikit-learn pipeline:

- `StandardScaler`
- `LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")`

The LLM is frozen. Only the probe is trained.

## Normal Test Results

From `artifacts/metrics.json`:

| Metric | Value |
|---|---:|
| Synthetic test accuracy | 1.000 |
| Synthetic macro F1 | 1.000 |
| ABSTAIN precision | 1.000 |
| ABSTAIN recall | 1.000 |

Because of audit findings, these numbers should be read as a successful
software-path smoke milestone, not as a robust benchmark.

## Hard Eval Results

From `artifacts/hard_eval_metrics.json`:

| Metric | Value |
|---|---:|
| Overall accuracy | 0.575 |
| Macro F1 | 0.587 |
| Executable-only accuracy | 0.736 |
| ABSTAIN precision | 0.702 |
| ABSTAIN recall | 0.440 |

Hard negatives are the main failure mode. Negation rejection is `0.267`,
compound-command rejection is `0.133`, and near-miss rejection is `0.400`.

## OOD Evaluation

The first OOD run was weak: max probability AUROC `0.41696`, top-2 margin AUROC
`0.41508`, and centroid distance AUROC `0.49636`.

The advanced OOD run compared additional signals:

| Score | AUROC | AUPRC |
|---|---:|---:|
| Negative Mahalanobis distance | 0.9977 | 0.9932 |
| LOF novelty score | 0.7771 | 0.5054 |
| Top-2 margin | 0.7106 | 0.3307 |
| Max softmax probability | 0.7050 | 0.3237 |
| Nearest executable centroid distance | 0.4917 | 0.2928 |

The Mahalanobis result is promising but should be treated as preliminary because
the dataset is small and synthetic activations may have unusually clean geometry.

## Prompt Robustness

The default router prompt works best. Alternate prompts do not transfer:

| Prompt | Synthetic accuracy | Hard accuracy | Hard macro F1 |
|---|---:|---:|---:|
| Router prompt | 0.927 | 0.575 | 0.587 |
| Minimal | 0.273 | 0.545 | 0.118 |
| Instruction | 0.273 | 0.545 | 0.118 |
| Noisy | 0.273 | 0.545 | 0.118 |

The alternate prompts mostly collapse into `ABSTAIN`, producing high hard
ABSTAIN recall but no executable routing. Prompt template is therefore part of
the feature space.

## Layer Sweep

Fast sampled `distilgpt2` sweep:

| Layer | Synthetic accuracy | Hard accuracy | Hard macro F1 |
|---|---:|---:|---:|
| `block_0` | 0.889 | 0.419 | 0.465 |
| `block_3` | 0.852 | 0.463 | 0.511 |
| `block_5` | 0.963 | 0.481 | 0.519 |
| `pre_lm_head` | 0.963 | 0.475 | 0.512 |

Later layers are more linearly separable on synthetic data, but hard-eval
generalization remains limited.

## Data Efficiency

Synthetic accuracy reaches `0.927` with 20 examples per class and `1.000` with
30 examples per class in this run. Hard eval reaches only `0.513` at 30 to 50
examples per class, with hard ABSTAIN recall below `0.22`.

More synthetic examples do not substitute for hard negatives.

## Baselines

TF-IDF + logistic regression matches the latent probe on synthetic test accuracy:
`1.000` vs `1.000`. On hard eval, TF-IDF gets accuracy `0.545` and macro-F1
`0.598`; the latent router gets accuracy `0.575` and macro-F1 `0.587`.

The latent route is architecturally interesting, but this dataset does not show
clear superiority over a text baseline.

## Limitations

- The synthetic dataset leaks template families across splits.
- The hard eval is hand-authored but still small.
- Prompt transfer is poor.
- Runtime threshold calibration was added in the later V2 milestone; this report
  predates those calibrated artifacts.
- Mahalanobis OOD needs a larger validation set before becoming a default gate.
- The app is intentionally tiny and sandboxed.

## Future Work

1. Group split by template family.
2. Train with hard negatives and evaluate on held-out negative families.
3. Treat prompt template and layer as explicit model-card metadata.
4. Validate Mahalanobis and LOF OOD on larger independent data.
5. Add activation-geometry plots and calibration curves.

## CV-Ready Summary

Built and evaluated a zero-generation action-routing prototype that maps frozen
LLM hidden states into typed application actions through a linear probe, with
forward hooks, hard-negative evaluation, prompt robustness tests, layer sweeps,
data-efficiency curves, OOD scoring, and honest text-baseline comparisons.

## Milestone 4 Addendum

Milestone 4 rebuilt the synthetic dataset as `data/intent_dataset_v2.csv`.
Compared with V1, V2 has zero normalized duplicate texts, zero split text
overlap, zero template-family overlap, and zero cross-split near-duplicate pairs
at the audit threshold.

V2 `distilgpt2` feature extraction produced `X=(480, 768)`. The default V2
latent router keeps synthetic test accuracy at `1.000` and raises hard-eval
accuracy to `0.607`, but macro-F1 drops to `0.349` because the router becomes
cautious about executable hard paraphrases. Calibration improves hard-eval
accuracy to `0.647` and macro-F1 to `0.483`; prompt augmentation gives the best
latent hard macro-F1 in this run at `0.516`.

The refreshed TF-IDF baseline remains competitive with hard-eval accuracy
`0.665` and macro-F1 `0.587`. The final safe claim is architectural: the project
demonstrates and evaluates zero-generation latent-state routing, while making
clear that small text baselines remain strong on this benchmark.
