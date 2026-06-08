# OOD Calibration

Milestone 4 added validation-selected router calibration. The selection
objective was:

```text
0.4 * executable_accept_rate
+ 0.4 * hard_negative_reject_rate
+ 0.2 * macro_f1
```

Calibration used only the v2 validation split. Hard eval was held out for final
measurement.

Artifacts:

- `artifacts/router_calibration_v2.json`
- `artifacts/router_thresholds_calibrated_v2.json`
- `artifacts/hard_eval_calibrated_v2.json`
- `artifacts/hard_eval_predictions_calibrated_v2.csv`

## Selected Runtime Gate

| Field | Value |
|---|---:|
| Selected method | `max_softmax_probability` |
| Threshold | 0.7045 |
| Validation objective | 1.000 |
| Validation AUROC | 0.487 |
| Validation AUPRC | 0.621 |

Several methods tied on the validation objective. Max-softmax was selected by
tie order, not because it had the best AUROC.

## Hard Eval Effect

| Run | Hard accuracy | Hard macro F1 | Executable-only accuracy | ABSTAIN precision | ABSTAIN recall |
|---|---:|---:|---:|---:|---:|
| V2 default gates | 0.607 | 0.349 | 0.176 | 0.587 | 0.967 |
| V2 calibrated gate | 0.647 | 0.483 | 0.336 | 0.633 | 0.907 |

Calibration improves hard accuracy, macro-F1, executable acceptance, and
ABSTAIN precision. It reduces ABSTAIN recall from `0.967` to `0.907`, which is
the expected tradeoff.

## Advanced OOD Scores

| Score | AUROC | AUPRC | False accept | False reject |
|---|---:|---:|---:|---:|
| Negative Mahalanobis distance | 0.990 | 0.982 | 0.000 | 0.067 |
| LOF novelty score | 0.668 | 0.327 | 0.444 | 0.267 |
| Top-2 margin | 0.602 | 0.222 | 0.545 | 0.222 |
| Max softmax probability | 0.592 | 0.218 | 0.567 | 0.200 |

Mahalanobis is the strongest score in the advanced OOD artifact, but it is not
installed as the default runtime gate because the validation-threshold objective
did not choose it. It should be validated on a larger independent hard-negative
set before being promoted.

## Honest Bottom Line

OOD gating is improved but not solved. Calibration helps recover executable
actions while retaining strong rejection, but prompt-augmented training is still
the stronger robustness improvement in this run.
