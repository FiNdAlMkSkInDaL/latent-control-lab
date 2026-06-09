# VectorBot Final Polish

## Fast-run baseline

The first CPU smoke run used `--fast` and produced 108 feature vectors.

| Metric | Value |
|---|---:|
| Model id | `distilgpt2` |
| Feature shape | `(108, 768)` |
| Test accuracy | 0.5833 |
| Macro F1 | 0.5176 |
| ABSTAIN precision / recall | 0.8571 / 1.0000 |
| Scripted demo | 6 executable actions accepted; 2 ABSTAIN examples rejected |

The fast run was modest because the held-out split was tiny: only 24 test rows,
with three examples per executable class. That made individual movement
confusions swing the macro score sharply.

## Full CPU run

The publication pass uses a fuller CPU-friendly dataset:

- 60 examples per executable class
- 180 ABSTAIN examples
- 540 total rows
- 377 train / 81 validation / 82 test
- `distilgpt2`, `pre_lm_head_last_token`, max length 128

Full-run metrics are stored in `artifacts/vectorbot_metrics_full.json`.

## Artifact naming policy

Small publication artifacts are committed:

- `artifacts/vectorbot_metrics_full.json`
- `artifacts/vectorbot_confusion_matrix_full.csv`
- `artifacts/vectorbot_thresholds_full.json`
- `artifacts/vectorbot_routes_full.jsonl`
- `artifacts/vectorbot_projection_full.csv`
- `artifacts/vectorbot_dataset_audit.json`
- `docs/assets/vectorbot_*.png`
- `docs/vectorbot_demo.html`

Large local artifacts stay ignored:

- `artifacts/vectorbot_features_distilgpt2_full.npz`
- `artifacts/vectorbot_probe_distilgpt2_full.joblib`
- model and Hugging Face cache directories

## What will be published

The README should lead with the visual VectorBot composite, a short explanation
of the hidden-state control path, the zero-generation invariant, the full
metrics table, and the exact demo transcript. Older TaskFlow and larger-model
material is historical context, not the public story.
