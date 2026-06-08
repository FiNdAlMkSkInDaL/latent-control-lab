# Model Comparison

Artifact:

- `artifacts/model_comparison.json`
- `artifacts/model_comparison.csv`

This was a fast sampled run on `data/intent_dataset_v2.csv` with
`sample_size=120` and `hard_sample_size=80`.

| Model | Status | Semantic evidence | Feature shape | Test accuracy | Hard accuracy | Notes |
|---|---|---:|---|---:|---:|---|
| `distilgpt2` | ok | yes | `[120, 768]` | 0.846 | 0.625 | Real small-model semantic run. |
| `gpt2` | failed | no | n/a | n/a | n/a | Hugging Face network access was blocked in the sandbox. |
| `sshleifer/tiny-gpt2` | ok | no | `[120, 2]` | 0.231 | 0.100 | Plumbing baseline only. |

## Interpretation

`distilgpt2` remains the verified local semantic model. `sshleifer/tiny-gpt2`
is useful for fast hook and pipeline tests, but its 2-dimensional hidden state
is not meaningful semantic evidence. `gpt2` was attempted, but the Codex
environment blocked access to Hugging Face, so no result is claimed.

## Gemma Status

`google/gemma-2-2b-it` remains the portfolio target for Colab/GPU reproduction.
This repository does not claim Gemma metrics unless that Colab path is actually
run and artifacts are saved.
