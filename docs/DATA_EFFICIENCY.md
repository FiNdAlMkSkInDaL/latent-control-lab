# Data Efficiency

Artifacts:

- `artifacts/data_efficiency.json`
- `artifacts/data_efficiency.csv`

The data-efficiency curve trains probes on saved `distilgpt2`
pre-`lm_head` features with balanced examples per class from the synthetic train
split. It evaluates on the normal synthetic test split and the full hard eval.

| Examples/class | Train rows | Synthetic accuracy | Hard accuracy | Hard macro F1 | Hard ABSTAIN recall |
|---:|---:|---:|---:|---:|---:|
| 5 | 30 | 0.691 | 0.302 | 0.303 | 0.087 |
| 10 | 60 | 0.745 | 0.407 | 0.428 | 0.127 |
| 20 | 120 | 0.927 | 0.484 | 0.524 | 0.187 |
| 30 | 180 | 1.000 | 0.513 | 0.553 | 0.220 |
| 50 | 225 | 0.982 | 0.513 | 0.560 | 0.193 |

## Interpretation

The synthetic task is learnable with few examples. Around 20 examples per class
already reaches `0.927` synthetic test accuracy, and 30 examples per class
reaches `1.000` in this deterministic sample.

Hard-eval performance improves much more slowly. The best hard accuracy in this
curve is only `0.513`, and hard ABSTAIN recall remains below `0.22`. More
synthetic examples do not fix the missing hard-negative coverage. The next useful
data improvement is not simply more of the same templates; it is new negative
families and group-held-out evaluation.
