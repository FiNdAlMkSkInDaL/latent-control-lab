# Colab Gemma Real-Model Run

This is the reproducible Colab/GPU path for the portfolio target model
`google/gemma-2-2b-it`. The local verified model is `distilgpt2`; do not claim
Gemma results unless this notebook/path has actually been run.

## 1. Runtime

In Colab, select a GPU runtime. A T4 should work with small batches and 4-bit
loading; L4/A100 is more comfortable.

## 2. Clone And Install

```bash
git clone <REPO_URL> neural-native-software
cd neural-native-software
python -m pip install --upgrade pip
python -m pip install -e ".[dev,llm]"
```

If bitsandbytes fails on the selected runtime, continue without 4-bit by adding
`--no-4bit` to model-loading scripts.

## 3. Hugging Face Token

Gemma may require accepting the license on Hugging Face and setting `HF_TOKEN`.

```python
import os
from getpass import getpass

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("HF_TOKEN: ")
```

Optional login:

```bash
huggingface-cli login
```

## 4. Generate V2 Dataset

```bash
python scripts/generate_dataset_v2.py \
  --output data/intent_dataset_v2.csv \
  --seed 42 \
  --examples-per-class 60 \
  --abstain-examples 180 \
  --strict
```

## 5. Audit V2 Dataset

```bash
python scripts/audit_dataset.py \
  --dataset data/intent_dataset_v2.csv \
  --output-json artifacts/dataset_v2_audit.json \
  --output-csv artifacts/dataset_v2_near_duplicates.csv
```

## 6. Extract Gemma Features

```bash
python scripts/extract_features.py \
  --dataset data/intent_dataset_v2.csv \
  --model-id google/gemma-2-2b-it \
  --batch-size 2 \
  --max-length 160 \
  --output artifacts/features_gemma2_2b_v2_pre_lm_head.npz
```

If memory is tight, reduce `--batch-size` to `1`. If 4-bit loading causes
environment trouble, rerun with `--no-4bit`.

## 7. Train Probe

```bash
python scripts/train_probe.py \
  --features artifacts/features_gemma2_2b_v2_pre_lm_head.npz \
  --output artifacts/probe_gemma2_2b_v2.joblib \
  --metrics artifacts/metrics_gemma2_2b_v2.json \
  --confusion-matrix artifacts/confusion_matrix_gemma2_2b_v2.csv \
  --thresholds artifacts/thresholds_gemma2_2b_v2.json \
  --random-state 42
```

## 8. Evaluate Hard Eval

```bash
python scripts/evaluate_hard_eval.py \
  --model-id google/gemma-2-2b-it \
  --probe artifacts/probe_gemma2_2b_v2.joblib \
  --dataset data/hard_eval.csv \
  --output-metrics artifacts/hard_eval_metrics_gemma2_2b_v2.json \
  --output-predictions artifacts/hard_eval_predictions_gemma2_2b_v2.csv \
  --output-confusion artifacts/hard_eval_confusion_matrix_gemma2_2b_v2.csv \
  --features-output artifacts/hard_eval_features_gemma2_2b_v2_pre_lm_head.npz \
  --batch-size 2
```

## 9. Calibrate Router

```bash
python scripts/calibrate_router.py \
  --probe artifacts/probe_gemma2_2b_v2.joblib \
  --features artifacts/features_gemma2_2b_v2_pre_lm_head.npz \
  --output-calibration artifacts/router_calibration_gemma2_2b_v2.json \
  --output-thresholds artifacts/router_thresholds_calibrated_gemma2_2b_v2.json
```

Rerun hard eval with calibrated thresholds:

```bash
python scripts/evaluate_hard_eval.py \
  --model-id google/gemma-2-2b-it \
  --probe artifacts/probe_gemma2_2b_v2.joblib \
  --dataset data/hard_eval.csv \
  --output-metrics artifacts/hard_eval_calibrated_gemma2_2b_v2.json \
  --output-predictions artifacts/hard_eval_predictions_calibrated_gemma2_2b_v2.csv \
  --output-confusion artifacts/hard_eval_confusion_matrix_calibrated_gemma2_2b_v2.csv \
  --thresholds-json artifacts/router_thresholds_calibrated_gemma2_2b_v2.json \
  --batch-size 2
```

## 10. Scripted Demo

```bash
python scripts/run_scripted_demo.py \
  --model-id google/gemma-2-2b-it \
  --probe artifacts/probe_gemma2_2b_v2.joblib \
  --thresholds-json artifacts/router_thresholds_calibrated_gemma2_2b_v2.json \
  --example-set v2 \
  --output artifacts/example_routes_gemma2_2b_v2.jsonl \
  --summary-output docs/DEMO_RESULTS_GEMMA2_2B_V2.md \
  --batch-size 6
```

## 11. Expected Output Files

Small portfolio artifacts:

- `artifacts/dataset_v2_audit.json`
- `artifacts/metrics_gemma2_2b_v2.json`
- `artifacts/confusion_matrix_gemma2_2b_v2.csv`
- `artifacts/thresholds_gemma2_2b_v2.json`
- `artifacts/hard_eval_metrics_gemma2_2b_v2.json`
- `artifacts/router_calibration_gemma2_2b_v2.json`
- `artifacts/router_thresholds_calibrated_gemma2_2b_v2.json`
- `artifacts/hard_eval_calibrated_gemma2_2b_v2.json`
- `artifacts/example_routes_gemma2_2b_v2.jsonl`

Large local artifacts to keep out of git:

- `artifacts/features_gemma2_2b_v2_pre_lm_head.npz`
- `artifacts/hard_eval_features_gemma2_2b_v2_pre_lm_head.npz`
- `artifacts/probe_gemma2_2b_v2.joblib`
- Hugging Face model cache files

## Troubleshooting

- `401` or gated model error: accept the Gemma license and set `HF_TOKEN`.
- CUDA out of memory: lower `--batch-size`, restart runtime, or use 4-bit.
- bitsandbytes import error: rerun with `--no-4bit`.
- Slow extraction: start with `--sample-size` for smoke checks, then run full.
- Do not claim Gemma metrics until the full commands above have produced saved
  artifacts.
