# V2 Results

Milestone 4 introduced `data/intent_dataset_v2.csv`, a leakage-reduced synthetic
dataset with hard negatives in train/validation and split-isolated template
families.

## Dataset Audit

| Audit check | V1 | V2 |
|---|---:|---:|
| Rows | 350 | 480 |
| Normalized duplicate texts | 25 | 0 |
| Split text overlap | 8 | 0 |
| Template-family overlap | 39 | 0 |
| Near-duplicate pairs | 69 | 7 |
| Cross-split near-duplicate pairs | 31 | 0 |

## Main Metrics

| Run | Synthetic/test accuracy | Synthetic/test macro F1 | Hard accuracy | Hard macro F1 | ABSTAIN precision | ABSTAIN recall |
|---|---:|---:|---:|---:|---:|---:|
| V1 latent router | 1.000 | 1.000 | 0.575 | 0.587 | 0.702 | 0.440 |
| V2 latent router | 1.000 | 1.000 | 0.607 | 0.349 | 0.587 | 0.967 |
| V2 calibrated router | 1.000 | 1.000 | 0.647 | 0.483 | 0.633 | 0.907 |
| V2 prompt-augmented default prompt | 1.000 | 1.000 | 0.651 | 0.516 | 0.668 | 0.847 |

## OOD Metrics

Simple v2 OOD AUROC:

| Score | AUROC |
|---|---:|
| Max probability | 0.551 |
| Top-2 margin | 0.557 |
| Negative entropy | 0.547 |
| Negative executable centroid distance | 0.415 |

Advanced v2 OOD:

| Score | AUROC | AUPRC | False accept | False reject |
|---|---:|---:|---:|---:|
| Negative Mahalanobis distance | 0.990 | 0.982 | 0.000 | 0.067 |
| LOF novelty score | 0.668 | 0.327 | 0.444 | 0.267 |
| Top-2 margin | 0.602 | 0.222 | 0.545 | 0.222 |
| Max softmax probability | 0.592 | 0.218 | 0.567 | 0.200 |

Mahalanobis remains the strongest geometry score in this experiment, but the
calibrated runtime selected max-softmax because the validation objective tied
across several methods and max-softmax appeared first. This is not proof that
max-softmax is inherently better.

## Baseline Comparison

| Method | Hard accuracy | Hard macro F1 |
|---|---:|---:|
| V2 prompt-augmented latent | 0.651 | 0.516 |
| V2 calibrated latent | 0.647 | 0.483 |
| TF-IDF + logistic regression | 0.665 | 0.587 |
| Majority class | 0.545 | 0.118 |
| Random stratified | 0.222 | 0.130 |

TF-IDF remains competitive and slightly stronger on hard macro-F1. The portfolio
claim should therefore focus on the zero-generation latent-state architecture,
not benchmark dominance.

## Interpretation

V2 substantially improves dataset hygiene and hard-negative rejection. It also
exposes a tradeoff: adding hard negatives makes the router more cautious, which
improves ABSTAIN recall but hurts executable paraphrase recall. Prompt
augmentation gives the best latent hard macro-F1 in this run and greatly reduces
prompt brittleness.
