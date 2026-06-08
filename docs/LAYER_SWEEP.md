# Layer Sweep

Command artifact:

- `artifacts/layer_sweep_metrics.json`
- `artifacts/layer_sweep_metrics.csv`

This fast sweep used `distilgpt2`, `sample_size=180` from the synthetic dataset,
and `hard_sample_size=160` from `data/hard_eval.csv`.

| Layer | Synthetic test accuracy | Macro F1 | Hard eval accuracy | Hard macro F1 |
|---|---:|---:|---:|---:|
| `block_0` | 0.889 | 0.885 | 0.419 | 0.465 |
| `block_3` | 0.852 | 0.821 | 0.463 | 0.511 |
| `block_5` | 0.963 | 0.948 | 0.481 | 0.519 |
| `pre_lm_head` | 0.963 | 0.948 | 0.475 | 0.512 |

## Interpretation

Action labels become more linearly separable in later `distilgpt2` layers on the
synthetic distribution. The final transformer block and the pre-`lm_head`
representation tie on synthetic test accuracy in this fast run.

Hard-eval transfer remains weak for every layer. The best hard-eval accuracy in
this sweep is `0.481` from `block_5`, and the best hard macro-F1 is `0.519`.
That means depth improves separability on familiar synthetic phrasing, but it
does not solve robustness to negation, compounds, or task-adjacent near misses.

## Caveat

This is a fast sampled run, not a full-depth benchmark. It is still useful
portfolio evidence because it measures where intent labels become linearly
separable across transformer depth without changing the zero-generation runtime
path.
