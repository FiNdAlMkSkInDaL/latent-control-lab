# Probe Card

## Feature space

Default: `pre_lm_head_last_token`.

## Model assumptions

- Frozen causal LM.
- Prompt template is fixed during feature extraction and inference.
- One vector per user request.

## Probe

Default: scikit-learn pipeline with `StandardScaler` and regularized multinomial `LogisticRegression`.

## Metrics to report

- Held-out accuracy.
- Macro F1.
- Confusion matrix.
- ABSTAIN precision/recall.
- OOD AUROC with max probability and top-2 margin.
