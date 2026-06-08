# Evaluation Audit

Generated from the Milestone 3 artifacts on 2026-06-08.

## Summary

The original `distilgpt2` run proves the software path works, but its `1.000`
synthetic test accuracy is not strong evidence of language robustness. The
dataset audit found split leakage signals:

| Check | Result |
|---|---:|
| Rows | 350 |
| Exact duplicate rows | 0 |
| Normalized duplicate texts | 25 |
| Normalized text overlaps across splits | 8 |
| Template families appearing across splits | 39 |
| Near-duplicate pairs at Jaccard >= 0.82 | 69 |
| Cross-split near-duplicate pairs | 31 |

Artifacts:

- `artifacts/dataset_audit.json`
- `artifacts/dataset_near_duplicates.csv`

## Why 1.000 Accuracy Can Mislead

The synthetic generator composes a small set of seed phrases with fixed prefixes
and suffixes. Even when exact rows are not duplicated, normalized variants and
near-duplicates cross split boundaries. The `template_family` field also appears
in multiple splits for most families, so the test split is not a true held-out
template-family evaluation.

The original score therefore supports "the probe can recover labels from a
familiar synthetic distribution", not "the latent route generalizes to new human
phrasing."

## Train/Test Near-Duplicates

There are 31 cross-split near-duplicate pairs at token Jaccard similarity
`>= 0.82`, plus 8 normalized text overlaps across train/validation/test. Examples
include repeated normalized task and ABSTAIN requests. This makes the synthetic
test set easier than it should be.

## Template Leakage

There are 39 template families that appear in more than one split. This is a
direct leakage channel because the generator assigns families by label and seed
index, then splits rows afterward. A stronger split should hold out whole
template families or use hand-authored examples that do not share the generator's
surface form.

## ABSTAIN Difficulty

The synthetic ABSTAIN set is too easy. It mostly contains unrelated requests,
gibberish, unsafe instructions, or generic knowledge questions. The hard eval
adds negation, compound commands, and near-miss task requests. Results show the
weakness clearly:

| Hard subset | Accuracy or rejection rate |
|---|---:|
| Negation rejection | 0.267 |
| Compound rejection | 0.133 |
| Near-miss rejection | 0.400 |
| Gibberish rejection | 0.600 |
| Unsafe rejection | 0.867 |
| Unrelated rejection | 0.867 |

The model rejects obvious unsafe/unrelated rows better than it rejects
task-adjacent negatives.

## OOD AUROC Interpretation

The first OOD artifact was weak:

| Score | AUROC |
|---|---:|
| Max probability | 0.41696 |
| Top-2 margin | 0.41508 |
| Executable centroid distance | 0.49636 |

The advanced OOD run improved some signals on a harder ID/OOD definition:

| Score | AUROC | False accept | False reject |
|---|---:|---:|---:|
| Negative Mahalanobis distance | 0.9977 | 0.018 | 0.025 |
| LOF novelty score | 0.7771 | 0.139 | 0.500 |
| Top-2 margin | 0.7106 | 0.012 | 1.000 |
| Max softmax probability | 0.7050 | 0.012 | 1.000 |
| Nearest executable centroid distance | 0.4917 | 0.188 | 0.750 |

The Mahalanobis score is promising in this small feature geometry, but it should
not be overclaimed until evaluated on a larger, independently authored dataset.
The confidence and margin thresholds were calibrated too aggressively on the
validation split and reject all normal ID examples at their selected thresholds.

## Prompt Robustness

The trained probe is prompt-bound. With the original router prompt, hard eval
accuracy is `0.575`. With minimal, instruction, and noisy prompts, hard eval
accuracy is `0.545` only because the router mostly predicts `ABSTAIN`, giving
perfect ABSTAIN recall and zero executable recall. Synthetic test accuracy drops
from `0.927` through the gated router to `0.273` for those alternate prompts.

## What The MVP Proves

- A real Hugging Face causal LM can be run with forward passes only.
- A pre-`lm_head` forward hook captures `[batch, seq, hidden_size]` activations.
- A lightweight probe can route frozen hidden states into typed app actions.
- The app mutation path is zero-generation: no generated JSON, no tool-call text,
  no regex action parser, and no `model.generate()` in routing.

## What It Does Not Yet Prove

- It does not prove robust language understanding.
- It does not prove prompt-invariant latent intent geometry.
- It does not prove OOD reliability outside this tiny setup.
- It does not beat a TF-IDF text baseline on the synthetic distribution.
- It does not justify deployment beyond the sandboxed toy task kernel.

## Recommended Next Evaluation Work

1. Rebuild the synthetic set with group splits by `template_family`.
2. Train with hard negatives: negation, compound commands, near misses, and
   ambiguous task-adjacent requests.
3. Evaluate prompt templates as separate feature spaces rather than assuming
   prompt transfer.
4. Validate Mahalanobis OOD on a larger set before updating runtime gates.

## Milestone 4 V2 Dataset Audit

Milestone 4 generated `data/intent_dataset_v2.csv` and reran the same audit.

| Check | V1 | V2 |
|---|---:|---:|
| Rows | 350 | 480 |
| Exact duplicate rows | 0 | 0 |
| Normalized duplicate texts | 25 | 0 |
| Normalized text overlaps across splits | 8 | 0 |
| Template-family split overlaps | 39 | 0 |
| Near-duplicate pairs at Jaccard >= 0.82 | 69 | 7 |
| Cross-split near-duplicate pairs | 31 | 0 |

V2 fixes the known leakage channels by assigning template families to one split
only, adding stronger hard-negative ABSTAIN examples, and avoiding normalized
duplicates. The v2 synthetic test score is still `1.000`, so it should still be
reported alongside hard eval, not by itself.

V2 hard-eval behavior shifted: default V2 gates improve hard-negative rejection
but reject many executable paraphrases. Calibrated V2 gates improve hard eval
accuracy to `0.647` and macro-F1 to `0.483`, while keeping ABSTAIN recall at
`0.907`.
