# Run Real Model

This document records the real-model milestone that Codex executed in this
workspace and gives reproducible CPU and Colab/GPU paths.

## Verified Path

The verified route is:

```text
text -> tokenizer -> frozen Hugging Face causal LM forward pass -> pre-lm_head hook -> vector -> linear probe -> gates -> enum action -> TaskFlowKernel
```

The route does not call `model.generate()`. It also does not parse generated
JSON, tool-call text, SQL, shell commands, regexes, keywords, or user text to
choose an app action.

## What Codex Executed

Codex executed the real semantic small-model path with `distilgpt2` on CPU:

```powershell
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -e ".[dev,llm]"
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check .
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest --basetemp=.pytest_tmp
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\generate_dataset.py --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\extract_features.py --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\train_probe.py --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\evaluate_ood.py --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\run_scripted_demo.py --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m neural_native.cli --help
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\generate_dataset.py --output data\intent_dataset.csv --seed 42
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\extract_features.py --model-id distilgpt2 --batch-size 4 --no-4bit --output artifacts\features_distilgpt2_pre_lm_head.npz
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\train_probe.py --features artifacts\features_distilgpt2_pre_lm_head.npz --output artifacts\probe.joblib --metrics artifacts\metrics.json --confusion-matrix artifacts\confusion_matrix.csv --thresholds artifacts\thresholds.json --random-state 42
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\evaluate_ood.py --probe artifacts\probe.joblib --features artifacts\features_distilgpt2_pre_lm_head.npz --output artifacts\ood_metrics.json --seed 42
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\run_scripted_demo.py --model-id distilgpt2 --probe artifacts\probe.joblib --output artifacts\example_routes.jsonl --summary-output docs\DEMO_RESULTS.md --batch-size 6 --no-4bit --seed 42
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check .
& 'C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest --basetemp=.pytest_tmp
```

Observed feature shape: `X=(350, 768)` for `distilgpt2` pre-`lm_head`
final-token activations.

## CPU Small-Model Path

`distilgpt2` is the local verification model because it is small enough for CPU
execution while still exposing a real 768-dimensional hidden state. `gpt2` is
also valid but slower. `sshleifer/tiny-gpt2` is used for hook/CI plumbing only
because its hidden size is too small for meaningful routing metrics.

```bash
python -m pip install -e ".[dev,llm]"
python scripts/generate_dataset.py --output data/intent_dataset.csv --seed 42
python scripts/extract_features.py \
  --model-id distilgpt2 \
  --batch-size 4 \
  --max-length 160 \
  --no-4bit \
  --output artifacts/features_distilgpt2_pre_lm_head.npz
python scripts/train_probe.py \
  --features artifacts/features_distilgpt2_pre_lm_head.npz \
  --output artifacts/probe.joblib \
  --metrics artifacts/metrics.json \
  --confusion-matrix artifacts/confusion_matrix.csv \
  --thresholds artifacts/thresholds.json \
  --random-state 42
python scripts/evaluate_ood.py \
  --probe artifacts/probe.joblib \
  --features artifacts/features_distilgpt2_pre_lm_head.npz \
  --output artifacts/ood_metrics.json \
  --seed 42
python scripts/run_scripted_demo.py \
  --model-id distilgpt2 \
  --probe artifacts/probe.joblib \
  --output artifacts/example_routes.jsonl \
  --summary-output docs/DEMO_RESULTS.md \
  --batch-size 6 \
  --no-4bit \
  --seed 42
```

Expected runtime on a laptop CPU is a few minutes for `distilgpt2` feature
extraction. Peak memory is typically a few GB. The generated `.npz` and `.joblib`
files are intentionally ignored by git.

## Colab / GPU Gemma Path

Gemma is the portfolio target because it is a modern instruction-tuned
open-weight model with a richer representation than GPT-2-family CPU smoke
models. In Colab, a T4/A100/L4 runtime can run the same pipeline with
`google/gemma-2-2b-it`.

For gated model access, set `HF_TOKEN` before extraction:

```bash
export HF_TOKEN=hf_...
```

Then execute:

```bash
python -m pip install -e ".[dev,llm]"
python scripts/generate_dataset.py --output data/intent_dataset.csv --seed 42
python scripts/extract_features.py \
  --model-id google/gemma-2-2b-it \
  --batch-size 8 \
  --max-length 160 \
  --output artifacts/features_gemma2_2b_pre_lm_head.npz
python scripts/train_probe.py \
  --features artifacts/features_gemma2_2b_pre_lm_head.npz \
  --output artifacts/probe.joblib \
  --metrics artifacts/metrics.json \
  --confusion-matrix artifacts/confusion_matrix.csv \
  --thresholds artifacts/thresholds.json
python scripts/evaluate_ood.py \
  --probe artifacts/probe.joblib \
  --features artifacts/features_gemma2_2b_pre_lm_head.npz \
  --output artifacts/ood_metrics.json
python scripts/run_scripted_demo.py \
  --model-id google/gemma-2-2b-it \
  --probe artifacts/probe.joblib \
  --output artifacts/example_routes.jsonl \
  --summary-output docs/DEMO_RESULTS.md
```

On CUDA, the loader uses 4-bit quantization when `bitsandbytes` is available.
Pass `--no-4bit` to force full precision.

## Fast Verification Modes

The scripts include fast/sample knobs for sandbox checks:

```bash
python scripts/generate_dataset.py --fast --seed 42
python scripts/extract_features.py --model-id sshleifer/tiny-gpt2 --sample-size 24 --batch-size 8 --no-4bit
python scripts/evaluate_ood.py --probe artifacts/probe.joblib --features artifacts/features_distilgpt2_pre_lm_head.npz --sample-size 80 --seed 42
python scripts/run_scripted_demo.py --fast --model-id distilgpt2 --probe artifacts/probe.joblib --no-4bit
```

Fast tiny-model output should be labeled as plumbing-only, not semantic proof.

## Troubleshooting

- Hugging Face download blocked: rerun in an environment with Hub access or use
  a locally cached model path as `--model-id`. If this blocks verification,
  record it in `docs/BLOCKERS.md`.
- Gated Gemma access denied: set `HF_TOKEN` and ensure the account has accepted
  the model license.
- CPU run too slow: use `distilgpt2 --batch-size 2`, then move the Gemma run to
  Colab/GPU.
- Windows has no `bitsandbytes`: this is expected. CPU verification uses
  `--no-4bit`.
- Large artifacts: keep `artifacts/*.npz`, `artifacts/*.joblib`, model weights,
  and Hugging Face cache files out of git.
