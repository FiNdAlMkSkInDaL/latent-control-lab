# Baselines

Artifacts:

- `artifacts/baseline_metrics.json`
- `artifacts/baseline_metrics.csv`

The text baseline is evaluation-only. It is not used by the zero-generation app
route.

| Method | Dataset | Accuracy | Macro F1 | ABSTAIN precision | ABSTAIN recall |
|---|---|---:|---:|---:|---:|
| Latent probe router | Synthetic test | 1.000 | 1.000 | 1.000 | 1.000 |
| Latent probe router | Hard eval | 0.575 | 0.587 | 0.702 | 0.440 |
| TF-IDF + logistic regression | Synthetic test | 1.000 | 1.000 | 1.000 | 1.000 |
| TF-IDF + logistic regression | Hard eval | 0.545 | 0.598 | 0.971 | 0.227 |
| Random stratified | Synthetic test | 0.200 | 0.185 | 0.357 | 0.333 |
| Random stratified | Hard eval | 0.207 | 0.141 | 0.547 | 0.273 |
| Majority class | Synthetic test | 0.273 | 0.071 | 0.273 | 1.000 |
| Majority class | Hard eval | 0.545 | 0.118 | 0.545 | 1.000 |

## Interpretation

TF-IDF matches the latent probe on the synthetic split. That confirms the
synthetic dataset is surface-form easy and should not be used as evidence that
latent routing beats text classifiers.

On hard eval, the latent router has slightly higher accuracy than TF-IDF and a
much better macro-F1 than the majority baseline, but the difference is modest.
The stronger claim is architectural: the app route is zero-generation and
vector-facing. The current experiments do not establish benchmark superiority
over text baselines.

## V2 Refresh

Artifacts:

- `artifacts/baseline_metrics_v2.json`
- `artifacts/baseline_metrics_v2.csv`

| Method | Dataset | Accuracy | Macro F1 | ABSTAIN precision | ABSTAIN recall |
|---|---|---:|---:|---:|---:|
| V2 latent router | V2 test | 1.000 | 1.000 | 1.000 | 1.000 |
| V2 latent router | Hard eval | 0.607 | 0.349 | 0.587 | 0.967 |
| V2 prompt-augmented latent | V2 test | 1.000 | 1.000 | 1.000 | 1.000 |
| V2 prompt-augmented latent | Hard eval | 0.651 | 0.516 | 0.668 | 0.847 |
| V2 calibrated latent | Hard eval | 0.647 | 0.483 | 0.633 | 0.907 |
| TF-IDF + logistic regression | V2 test | 1.000 | 1.000 | 1.000 | 1.000 |
| TF-IDF + logistic regression | Hard eval | 0.665 | 0.587 | 0.672 | 0.793 |
| Majority class | Hard eval | 0.545 | 0.118 | 0.545 | 1.000 |
| Random stratified | Hard eval | 0.222 | 0.130 | 0.480 | 0.327 |

V2 improves the latent router's hard-eval safety behavior, especially ABSTAIN
recall. Prompt augmentation gives the strongest latent hard macro-F1. TF-IDF is
still competitive and wins hard macro-F1 in this small benchmark, so the
portfolio claim remains about the zero-generation latent-state architecture and
evaluation discipline.
